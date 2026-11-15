"""Dense samplers: a field's draws at chosen DoFs, or under a linear observation, as small dense matrix products.

    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    A = d.dense(device="cuda")                     # the FE coefficients at the trace DoFs (bnd=), or at dofs=[...]
    B = d.dense(observe=E, device="cuda", dtype="float32")    # E @ the coefficients, E any (m x ndof) matrix
    c = A.sample(n, generator=gen)                 # (n, len(A.dofs)); correlated fields: (n, k, m)
    A.save("inflow.npz")
    A2 = ngsrng.DenseSampler.load("inflow.npz", device="cuda")   # numpy and torch only: no NGSolve needed

``dense()`` exists on every distribution (``rng.distribution(...)``), on ``ngsrng.correlated(...)`` and on the
Gaussians (``Gaussian``, ``LowRankGaussian``, ``MatheronPosterior``, ``StationaryMatern``: their coefficients). The
set-up is made once per object and arguments and kept there; draws are matrix products, on any device.

What is dense, and why it is exact
----------------------------------
Every family it covers is a pointwise map of one latent Gaussian, followed by linear maps::

    z ~ N(0, I_r)  ->  w = m + F z     latent values at the points S that determine the outputs
                   ->  v = T(w)        the family's pointwise map (an affine map for Gaussian fields)
                   ->  y = A v         the representation's linear map to the outputs (FE coefficients at the chosen
                                       DoFs, composed with bnd= and observe=)

- **The points S.** Only the latent values that the outputs depend on: the columns of A that are not zero. For
  ``representation="bernstein"`` and a trace (``bnd=``) these are the Bernstein domain points on the boundary (the
  trace of a Bernstein function depends only on its domain points there); for values at points (the default
  ``"quadrature"`` representation, ``points=``) the chosen points; for Gaussian FE fields the chosen DoFs, or the
  observations themselves when there are fewer of them than DoFs they depend on. A is found by solving with the
  representation's mass matrix (one solve per output), and its entries below ``ZERO`` times the largest of their
  row are rounding (exact zeros of the conversion): they are dropped (``info["zero_drop"]`` records the largest).
- **The latent covariance** K = G Sigma G^T at S (G the evaluation at S, standardised as the family standardises
  its latent; Sigma the coefficient covariance, conditioning included) is the exact discrete covariance of what
  ``d.sample()`` draws: it comes from |S| applications of the factor L^T (:func:`ngsrng.observed.operator_covariance`).
- **The factor** F: the eigendecomposition of K, over the eigenvalues above ``rtol`` times the largest (exact up to
  that cutoff: ||K - F F^T||_2 <= rtol max eig K), in decreasing order: the columns of F are the Karhunen-Loeve modes
  at S, so ``rank=`` / ``tol=`` truncate it optimally (``info["exact"]`` and ``info["captured_variance"]`` say what
  was kept). **Pinned points**, whose variance is at most ``rtol`` times the largest (a Dirichlet DoF, a point
  conditioned to the noise level, a point a family standardises to a constant), get a zero row: they equal their
  mean exactly, and the rest is factorised without any jitter.
- **Correlated components** (``ngsrng.correlated``) share K: the joint factor is L_R (x) F (R the cross-correlation),
  and each component keeps its own mean, map and output map.
- **The Bernstein range guarantee** holds as in ``d.sample()``: the map comes before the linear conversion, and the
  values are pulled inside the family's range by a relative margin of at least 1000 times the conversion's measured
  rounding (summed over S, so it bounds every sample) before it.

The law is that of ``d.sample()`` (its marginal at the chosen DoFs, or its image under ``observe``), up to the
eigenvalue cutoff and rounding; the draws themselves differ (another random stream). Data drawn by either route can
be mixed. Not covered: ``bnd_values=`` (conditioning on a trace), families built from several latents (chi-square
sums, Student-t, Wishart, Dirichlet, ...), and Karhunen-Loeve fields; Gaussian FE fields and every pointwise
family of one latent (closed-form quantiles, numerical inversion with constant parameters, ``transformed``,
``indicator``) are. ``save`` stores the closed-form maps (``uniform``, ``lognormal``, ``normal``, ...,
``indicator``) and Gaussian fields; numerically inverted families (``gamma``, ``beta``, ...) sample in the process.

Size: up to ``MAX_POINTS`` latent points by default (an 800 MB float64 factor at the limit; ``max_points=`` moves
it). The set-up costs |S| applications of L^T and an |S| x |S| eigendecomposition; a draw costs |S| r + nnz(A)
multiply-adds per sample and component.

Randomness
----------
As every sampler (:mod:`ngsrng.seeding`): ``seed=None`` reads torch's global generator at each draw, an int is an own
stream, and ``sample(n, generator=g)`` reads ``g`` for that draw (one key per draw). The draws are sample-major
standard normals z (n, k, r); every sample's z is fixed by its position in the stream, so the draws do not depend
on how a batch is split (``sample(4)`` then ``sample(6)`` is ``sample(10)``: the same z, and the same values up to the
rounding of products over other batch sizes). ``noise="host"`` (the CPU's default) draws z with NumPy and copies it
to the device: the same samples on every device. ``noise="device"`` (a GPU's default) draws z on the device with
torch, keyed per block of ``block`` samples (fast; the samples then depend on the device, as torch's generators do).

Stored samplers
---------------
``save(path)`` writes one ``.npz`` (no pickles): the factor, the means, the maps (family, parameters, range), the
output maps, the DoF indices and latent points, and provenance (ngsrng version and commit, space, mesh hash,
representation, stream convention). ``DenseSampler.load(path, device=...)`` needs NumPy and torch only. Without
ngsrng at all, a stored sampler of a closed-form family draws with NumPy and SciPy::

    import json, numpy as np
    from scipy.special import ndtr
    s = np.load("sampler.npz")
    meta = json.loads(str(s["meta"]))
    z = np.random.default_rng(0).standard_normal((n, *s["mix"].shape[:1], s["factor"].shape[1]))
    z = np.einsum("ij,njr->nir", s["mix"], z)                      # correlate the components
    for i, c in enumerate(meta["components"]):
        w = s[f"mean{i}"] + z[:, i] @ s["factor"].T               # latent values (n, |S|)
        p = {k: s[v] if isinstance(v, str) else v for k, v in c["params"].items()}
        v = {"uniform": lambda w, low, high: low + (high - low) * ndtr(w),
             "lognormal": lambda w, mean, sigma: np.exp(mean + sigma * w),
             "affine": lambda w, shift, scale: shift + scale * w}[c["map"]](w, **p)
        if c["out_shape"]:                                         # the output map (none: the values themselves)
            a = np.zeros(c["out_shape"])
            a[s[f"out{i}_rows"], s[f"out{i}_cols"]] = s[f"out{i}_vals"]
            v = v @ a.T                                            # (n, m): component i

(the other closed forms are in :mod:`ngsrng.quantiles`; ``DenseSampler.from_noise(z)`` gives the same values from the
same z, up to the range margins of bounded families, which move values within ~1e-9 of a bound by at most that.)
"""

import hashlib
import json
import math
import numbers
import time

import numpy as np
import scipy.sparse as sp
import torch

from .backend import DeviceCache, resolve_device, resolve_dtype
from .inplace import InPlace, as_tensor, default_work_values
from .quantiles import MAPS, NAMES, XP
from .seeding import Source, as_source, one_draw

__all__ = ["DenseSampler"]

FORMAT = "ngsrng.dense/1"
MAX_POINTS = 10_000  # latent points of an exact dense factor by default (|S|^2 float64: 0.8 GB at the limit)
RTOL = 1e-12  # eigenvalues below rtol * max are dropped; points with variance below rtol * max are pinned
ZERO = 1e-11  # entries of the output map below ZERO * (their row's largest) are rounding of exact zeros
MARGIN = 1e-9  # the Bernstein representation's smallest relative range margin (as PointwiseField.MARGIN)
BLOCK_NORMALS = 2**22  # standard normals per keyed block of device noise
DENSE_OUT = 2**22  # output maps up to this many entries are kept dense on the device, larger ones sparse


def _affine(w, xp, shift, scale):
    return shift + scale * w


MAPS = {**MAPS, "affine": _affine}


class DenseSampler(InPlace):
    """Draws of a field at chosen DoFs or under a linear observation, by dense products (see :mod:`ngsrng.dense`).

    Made by ``dense()`` (on a distribution, ``ngsrng.correlated(...)`` or a Gaussian) or :meth:`load`. ``sample(n)``
    gives (n, m), or (n, k, m) for k correlated components (a tuple when their output sizes differ); ``dofs`` (the
    DoF indices of the outputs, None under ``observe=``), ``rank``, ``info``, :meth:`save`, :meth:`to`.
    """

    def __init__(self, law, *, device="cpu", dtype=None, noise=None, seed=None, output=None, work_values=None):
        self.law = law
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype) if dtype is not None else torch.float64
        self.noise = noise or ("host" if self.device.type == "cpu" else "device")
        if self.noise not in ("host", "device"):
            raise ValueError("noise must be 'host' or 'device'")
        self.source = as_source(seed)
        self.output = output
        self.work_values = default_work_values(self.device) if work_values is None else int(work_values)
        self._device = DeviceCache()
        k, r = len(law["components"]), law["factor"].shape[1]
        self.block = max(1, BLOCK_NORMALS // max(1, k * r))  # samples per keyed block of device noise

    # ------------------------------------------------------------------ description

    @property
    def components(self) -> int:
        """Number of jointly drawn components (1 for a single field)."""
        return len(self.law["components"])

    @property
    def rank(self) -> int:
        """Columns of the latent factor (standard normals per sample and component)."""
        return int(self.law["factor"].shape[1])

    @property
    def latent_points(self) -> int:
        return int(self.law["factor"].shape[0])

    @property
    def dofs(self):
        """The indices of the outputs among the values ``d.sample()`` returns (FE DoFs for coefficient
        representations), or None for an observation (``observe=``)."""
        return self.law.get("dofs")

    @property
    def nout(self):
        sizes = [c["nout"] for c in self.law["components"]]
        return sizes[0] if len(set(sizes)) == 1 else tuple(sizes)

    @property
    def info(self) -> dict:
        return {**self.law["info"], "device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""),
                "noise": self.noise, "block": self.block, "randomness": self.source.describe()}

    def __repr__(self):
        i = self.law["info"]
        return (f"DenseSampler({i.get('law', '')}: {self.components} x {self.nout} outputs, {self.latent_points} "
                f"latent points, rank {self.rank}, {'exact' if i.get('exact') else 'truncated'}, device={self.device})")

    def to(self, device=None, dtype=None, noise=None, seed=None, output=None) -> "DenseSampler":
        """The same law with other sampling settings. What is not given is kept: the set-up, and the randomness (the
        same stream, continued by both samplers); ``noise`` follows a new device's default."""
        if noise is None and device is None:
            noise = self.noise
        return DenseSampler(self.law, device=self.device if device is None else device,
                            dtype=self.dtype if dtype is None else dtype, noise=noise,
                            seed=self.source if seed is None else seed,
                            output=self.output if output is None else output)

    # ------------------------------------------------------------------ draws

    def sample(self, n=None, generator=None, output=None, out=None):
        """n draws: (n, m), or (n, k, m) for correlated components; ``n=None``: one, without the sample axis.

        ``generator``: a torch or NumPy generator read for this draw (one key), instead of the sampler's own
        randomness. ``output``: "numpy" or "torch" (default: NumPy on the CPU, torch tensors on a GPU). ``out``: a
        buffer of that shape (any float dtype and device, or one per component) filled in place and returned."""
        source = self.source if generator is None else Source(generator)
        with one_draw():
            if out is not None:
                return self._sample_out(n, out, source)
            ns = 1 if n is None else int(n)
            rows = [torch.empty((ns, c["nout"]), dtype=self.dtype, device=self.device)
                    for c in self.law["components"]]
            self._fill(rows, source)
        return self._deliver(rows, n, output)

    rvs = sample

    def sample_into(self, rows):
        """Fill (n, m) rows in place (one buffer per component, or one (n, k, m) buffer)."""
        with one_draw():
            self._sample_out(None, rows, self.source)
        return rows

    def from_noise(self, z, output=None):
        """The samples made from given standard normals ``z``: (n, k, r) (or (n, r) for one component), as
        ``sample`` returns them: the deterministic part of a draw (for antithetic or quasi-random z, or to check a
        stored sampler against a reference implementation)."""
        z = torch.as_tensor(z)
        if z.ndim == 2:
            z = z[:, None, :]
        k, r = self.components, self.rank
        if tuple(z.shape[1:]) != (k, r):
            raise ValueError(f"z must have shape (n, {k}, {r}), got {tuple(z.shape)}")
        rows = self._transform(z.to(self.device, self._work_dtype()).reshape(z.shape[0], k * r))
        return self._deliver([y.to(self.dtype) for y in rows], z.shape[0], output)

    def _sample_out(self, n, out, source):
        bufs = list(out) if isinstance(out, (list, tuple)) else [out]
        k = self.components
        if len(bufs) == 1 and k > 1:  # one (n, k, m) buffer
            t = as_tensor(bufs[0])
            rows = [t[:, i] for i in range(k)]
        elif len(bufs) == k:
            rows = [as_tensor(b) for b in bufs]
        else:
            raise ValueError(f"out: one buffer, or one per component ({k})")
        for r, c in zip(rows, self.law["components"]):
            if r.ndim != 2 or r.shape[1] != c["nout"] or (n is not None and r.shape[0] != int(n)):
                raise ValueError(f"out holds {tuple(r.shape)} values per component, expected (n, {c['nout']})")
        self._fill(rows, source)
        return out

    def _fill(self, rows, source):
        n = rows[0].shape[0]
        if n == 0:
            return
        k, r = self.components, self.rank
        per = k * (r + 2 * self.latent_points) + sum(c["nout"] for c in self.law["components"])
        step = max(1, int(self.work_values // max(per, 1)))
        streams = source.streams()
        for k0 in range(0, n, step):
            k1 = min(n, k0 + step)
            z = self._standard_normal(k1 - k0, k * r, streams)
            for row, y in zip(rows, self._transform(z)):
                row[k0:k1].copy_(y)

    def _work_dtype(self):
        return torch.float64 if self.law["work64"] or self.dtype == torch.float64 else self.dtype

    def _standard_normal(self, n, width, streams):
        """(n, width) standard normals of the next n samples of the stream, on the device (see the module
        docstring: host noise from NumPy, device noise keyed per block of samples)."""
        wd = self._work_dtype()
        if self.noise == "host":
            z = torch.from_numpy(streams.numpy().standard_normal((n, width)))
            return z.to(self.device, wd)
        seq, first = streams.reserve(n)
        out = torch.empty((n, width), dtype=wd, device=self.device)
        gen, b = self._generator(), self.block
        for blk in range(first // b, (first + n - 1) // b + 1):
            lo, hi = max(first, blk * b), min(first + n, (blk + 1) * b)
            gen.manual_seed(_block_seed(seq, blk))
            z = torch.randn((b, width), generator=gen, dtype=wd, device=self.device)
            out[lo - first:hi - first] = z[lo - blk * b:hi - blk * b]
        return out

    def _generator(self):
        key = ("generator", str(self.device))
        if key not in self._device:
            self._device[key] = torch.Generator(device=self.device)
        return self._device[key]

    def _transform(self, z):
        """Samples (one (n, m) tensor per component) from standard normals z (n, k r) on the device."""
        k, r = self.components, self.rank
        n = z.shape[0]
        ops = self._ops(z.dtype)
        z = z.reshape(n, k, r)
        if k > 1:
            z = torch.einsum("ij,njr->nir", ops["mix"], z)
        out = []
        for i, c in enumerate(ops["components"]):
            w = torch.addmm(c["mean"], z[:, i], ops["factor"].T) if r else c["mean"].expand(n, -1).clone()
            v = self._map(i, c, w)
            a = c["out"]
            if a is None:
                y = v
            elif a.layout == torch.strided:
                y = v @ a.T
            else:
                y = (a @ v.T).T
            out.append(y)
        return out

    def _map(self, i, c, w):
        spec = self.law["components"][i]
        if spec["map"] == "object":  # a transform of the process (numerical inversion, transformed)
            t = spec["transform"]
            if getattr(t, "torch_capable", False):
                v = t(w, XP(True))
            else:
                v = torch.as_tensor(t(w.double().cpu().numpy(), XP(False)), device=w.device).to(w.dtype)
        else:
            v = MAPS[spec["map"]](w, XP(True), **c["params"])
        for bound, op in c["clamp"]:  # strictly inside the support in the output precision
            v = torch.maximum(v, bound) if op == "max" else torch.minimum(v, bound)
        if c["pull"] is not None:  # Bernstein: inside the range by a margin safe against the conversion's rounding
            margin = spec["pull"]["margin"] * v.abs().amax(dim=1, keepdim=True).clamp_min(1e-300)
            for bound, sign in c["pull"]:
                v = torch.maximum(v, bound + margin) if sign > 0 else torch.minimum(v, bound - margin)
        return v

    def _ops(self, wd):
        """The law's arrays on the device in the work dtype (made once per device and dtype)."""
        key = (str(self.device), wd)
        if key in self._device:
            return self._device[key]
        dev = self.device

        def t(x):
            return torch.as_tensor(np.asarray(x, dtype=np.float64)).to(dev, wd)

        comps = []
        for spec in self.law["components"]:
            a = spec["out"]
            if a is not None:
                a = sp.csr_matrix(a)
                if a.shape[0] * a.shape[1] <= DENSE_OUT:
                    a = t(a.toarray())
                else:
                    from .backend import sparse_to_torch

                    a = sparse_to_torch(a, dev, wd)
            params = {k: (t(v) if isinstance(v, np.ndarray) else float(v)) for k, v in spec["params"].items()}
            comps.append({"mean": t(spec["mean"]), "params": params, "out": a,
                          "clamp": _clamp_bounds(spec.get("support"), self.dtype, t),
                          "pull": _pull_bounds(spec.get("pull"), t)})
        self._device[key] = {"factor": t(self.law["factor"]), "mix": t(self.law["mix"]), "components": comps}
        return self._device[key]

    def _deliver(self, rows, n, output):
        output = output or self.output or ("numpy" if self.device.type == "cpu" else "torch")
        if output not in ("numpy", "torch"):
            raise ValueError("output must be 'numpy' or 'torch'")
        if len(rows) == 1:
            res = rows[0]
        elif len({r.shape[1] for r in rows}) == 1:
            res = torch.stack(rows, dim=1)
        else:
            res = tuple(rows)
        if n is None:
            res = tuple(r[0] for r in res) if isinstance(res, tuple) else res[0]
        if output == "numpy":
            return tuple(r.cpu().numpy() for r in res) if isinstance(res, tuple) else res.cpu().numpy()
        return res

    # ------------------------------------------------------------------ storage

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_device"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._device = DeviceCache()

    def save(self, path):
        """Write the sampler's law to one ``.npz`` file (no pickles; see the module docstring for its layout and a
        NumPy reference sampler). Closed-form families and Gaussian fields; numerically inverted families (gamma,
        beta, ...) and ``transformed`` sample only in the process that built them."""
        law = self.law
        arrays = {"factor": law["factor"], "mix": law["mix"]}
        for key in ("dofs", "latent_index", "latent_coordinates"):
            if law.get(key) is not None:
                arrays[key] = np.asarray(law[key])
        comps = []
        for i, spec in enumerate(law["components"]):
            if spec["map"] == "object":
                raise NotImplementedError(
                    f"component {spec['name']}: its map ({spec['label']}) is a numerical inversion or a user "
                    "function, which is not stored; save works for closed-form families (uniform, lognormal, normal, "
                    "exponential, ...), indicator and Gaussian fields")
            arrays[f"mean{i}"] = spec["mean"]
            params = {}
            for name, v in spec["params"].items():
                if isinstance(v, np.ndarray):
                    arrays[f"param{i}_{name}"] = v
                    params[name] = f"param{i}_{name}"
                else:
                    params[name] = float(v)
            entry = {"name": spec["name"], "map": spec["map"], "params": params, "nout": int(spec["nout"]),
                     "support": _store_pair(spec.get("support"), arrays, f"support{i}"),
                     "pull": None if spec.get("pull") is None else
                     {"range": _store_pair(spec["pull"]["range"], arrays, f"range{i}"),
                      "margin": float(spec["pull"]["margin"])},
                     "out_shape": None}
            if spec["out"] is not None:
                a = sp.coo_matrix(spec["out"])
                arrays[f"out{i}_rows"], arrays[f"out{i}_cols"], arrays[f"out{i}_vals"] = a.row, a.col, a.data
                entry["out_shape"] = list(a.shape)
            comps.append(entry)
        meta = {"format": FORMAT, "work64": bool(law["work64"]), "components": comps, "info": law["info"],
                "dtype": str(self.dtype).replace("torch.", "")}
        np.savez(path, meta=np.array(json.dumps(meta, default=_json_default)), **arrays)

    @classmethod
    def load(cls, path, device="cpu", dtype=None, noise=None, seed=None, output=None) -> "DenseSampler":
        """A sampler from a file written by :meth:`save` (NumPy and torch only; no NGSolve). ``dtype`` defaults to
        the dtype it was saved with."""
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        meta = json.loads(str(arrays.pop("meta")))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: not a stored ngsrng dense sampler ({FORMAT})")
        comps = []
        for i, c in enumerate(meta["components"]):
            params = {k: arrays[v] if isinstance(v, str) else v for k, v in c["params"].items()}
            pull = c.get("pull")
            out = None
            if c["out_shape"] is not None:
                out = sp.csr_matrix((arrays[f"out{i}_vals"], (arrays[f"out{i}_rows"], arrays[f"out{i}_cols"])),
                                    shape=tuple(c["out_shape"]))
            comps.append({"name": c["name"], "map": c["map"], "params": params, "nout": c["nout"],
                          "mean": arrays[f"mean{i}"], "out": out,
                          "support": _load_pair(c.get("support"), arrays),
                          "pull": None if pull is None else {"range": _load_pair(pull["range"], arrays),
                                                             "margin": pull["margin"]}})
        law = {"factor": arrays["factor"], "mix": arrays["mix"], "components": comps, "work64": meta["work64"],
               "dofs": arrays.get("dofs"), "latent_index": arrays.get("latent_index"),
               "latent_coordinates": arrays.get("latent_coordinates"), "info": meta["info"]}
        saved = meta.get("dtype")
        return cls(law, device=device, dtype=dtype if dtype is not None else saved, noise=noise, seed=seed,
                   output=output)


def _block_seed(seq, block) -> int:
    """The torch seed of one block of device noise: a child of the draw's normal stream (its key and the block)."""
    child = np.random.SeedSequence(seq.entropy, spawn_key=tuple(seq.spawn_key) + (int(block),))
    return int(child.generate_state(1, np.uint64)[0] >> np.uint64(1))


def _clamp_bounds(support, out_dtype, t):
    """Bounds just inside a finite support in the output precision (as PointwiseField.clamp_open)."""
    if support is None:
        return []
    dt = torch.empty(0, dtype=out_dtype).numpy().dtype
    bounds = []
    for bound, toward, op in ((support[0], np.inf, "max"), (support[1], -np.inf, "min")):
        if bound is None:
            continue
        b = np.asarray(bound, dtype=float)
        if not np.all(np.isfinite(b)):
            continue
        inner = np.nextafter(b.astype(dt), np.asarray(toward, dtype=dt)).astype(np.float64)
        bounds.append((t(inner), op))
    return bounds


def _pull_bounds(pull, t):
    if pull is None:
        return None
    out = []
    for b, sign in ((pull["range"][0], 1.0), (pull["range"][1], -1.0)):
        if b is None:
            continue
        b = np.asarray(b, dtype=float)
        if np.isfinite(b).any():
            out.append((t(np.where(np.isfinite(b), b, -sign * np.inf)), sign))
    return out


def _store_pair(pair, arrays, key):
    if pair is None:
        return None
    out = []
    for j, b in enumerate(pair):
        if isinstance(b, np.ndarray):
            arrays[f"{key}_{j}"] = b
            out.append(f"{key}_{j}")
        else:
            out.append(None if b is None or not math.isfinite(b) else float(b))
    return out


def _load_pair(pair, arrays):
    if pair is None:
        return None
    return tuple(arrays[b] if isinstance(b, str) else b for b in pair)


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    return str(x)


# ====================================================================== set-up (needs NGSolve)


class DenseOf:
    """Mixin of the Gaussians: ``dense()``, the dense sampler of their coefficients."""

    def dense(self, dofs=None, observe=None, **kwargs) -> DenseSampler:
        """A :class:`ngsrng.DenseSampler` of this Gaussian's coefficients: at ``dofs`` (default all) or under
        ``observe`` (an m x ndof matrix), as dense products on any device (:mod:`ngsrng.dense`; arguments as
        :func:`ngsrng.dense.dense`). Set up once per arguments and kept with the Gaussian; ``save`` stores it."""
        return dense(self, dofs, observe, **kwargs)


def dense(source, dofs=None, observe=None, *, rank=None, tol=None, rtol=RTOL, max_points=MAX_POINTS, device=None,
          dtype=None, noise=None, seed=None, output=None) -> DenseSampler:
    """The dense sampler of ``source`` (a distribution, ``ngsrng.correlated(...)`` or a Gaussian); see
    :mod:`ngsrng.dense`. The set-up is made once per source and law arguments and kept on the source; the sampler
    once per further arguments.

    Parameters
    ----------
    dofs : indices of the outputs among the values a draw returns (FE DoFs for coefficient representations). Default:
        the trace DoFs with ``bnd=``, else all.
    observe : a linear map of those values (m x nvalues; SciPy sparse, NumPy or torch): the outputs are
        ``observe @ values`` (after ``bnd=``). Not with ``dofs``.
    rank, tol : truncate the factor to its leading ``rank`` modes, or to the fewest modes that keep a fraction
        1 - ``tol`` of the latent variance (Karhunen-Loeve; ``info["exact"]`` is then False).
    rtol : the eigenvalue cutoff of the exact factor and the pinning threshold (relative to the largest).
    max_points : the largest number of latent points (a clear error beyond).
    device, dtype, noise, seed, output : where and how it samples (default: as the source's own draws; ``noise``
        "host" or "device", see the module docstring; ``seed`` default: a child of the source's randomness).
    """
    from .correlated import CorrelatedFields
    from .generator import FieldDistribution

    if isinstance(source, FieldDistribution):
        owner, parts, mix = source.field, [_Part.of_field(source.field, source.generator, source.name)], np.eye(1)
        parent, keep64 = source.generator, source.generator._output_kind(source.field, source.name)[1]
        out_dtype, out_device = (torch.float64 if keep64 else source.generator.dtype), source.generator.device
    elif isinstance(source, CorrelatedFields):
        owner = source
        parts = [_Part.of_field(p.field, p.gen, p.name) for p in source._parts]
        mix, parent = source._mix, source._parts[0].gen
        out_dtype = torch.float64 if any(p.keep64 for p in source._parts) else parent.dtype
        out_device = parent.device
    elif hasattr(source, "apply_t") and hasattr(source, "space") and hasattr(source, "mean"):
        owner, parts, mix, parent = source, [_Part.of_gaussian(source)], np.eye(1), source
        out_dtype = getattr(source, "dtype", torch.float64)
        out_device = getattr(source, "torch_device", None) or getattr(source, "device", "cpu")
    else:
        raise TypeError(f"dense samplers are made from distributions, correlated fields or Gaussians, not "
                        f"{type(source).__name__}")
    if dofs is not None and observe is not None:
        raise ValueError("give dofs= or observe=, not both")
    law_key = (_digest(dofs), _digest(observe), rank, tol, float(rtol), int(max_points))
    cache = _cache_of(owner)
    if law_key not in cache["laws"]:
        cache["laws"][law_key] = _build_law(parts, mix, dofs, observe, rank, tol, float(rtol), int(max_points))
    law = cache["laws"][law_key]
    device = resolve_device(out_device if device is None else device)
    dtype = out_dtype if dtype is None else resolve_dtype(dtype)
    key = law_key + (str(device), str(dtype), noise, _digest(seed), output)
    if key not in cache["samplers"]:
        src = seed if seed is not None else _child_source(parent)
        cache["samplers"][key] = DenseSampler(law, device=device, dtype=dtype, noise=noise, seed=src, output=output)
    return cache["samplers"][key]


def _child_source(parent) -> Source:
    src = getattr(parent, "source", None)
    return src.spawn() if isinstance(src, Source) else Source()


def _cache_of(owner) -> dict:
    """The dense set-ups kept on ``owner``, tied to its identity (shallow copies, e.g. derived posteriors, must not
    share them)."""
    stored = owner.__dict__.get("_ngsrng_dense")
    if stored is None or stored[0] != id(owner):
        stored = (id(owner), {"laws": {}, "samplers": {}})
        owner.__dict__["_ngsrng_dense"] = stored
    return stored[1]


def _digest(x):
    """A hashable stand-in for an argument (arrays and matrices by content)."""
    if x is None or isinstance(x, (numbers.Number, str)):
        return x
    if isinstance(x, (np.random.SeedSequence, np.random.Generator, torch.Generator, Source)):
        return ("id", id(x))
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    h = hashlib.sha256()
    if sp.issparse(x):
        x = sp.csr_matrix(x)
        for a in (np.asarray(x.shape), x.indptr, x.indices, x.data):
            h.update(np.ascontiguousarray(a).tobytes())
        return ("sparse", h.hexdigest())
    a = np.asarray(x)
    h.update(str((a.shape, a.dtype.str)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return ("array", h.hexdigest())


class _Part:
    """One component, as the set-up needs it: its latent Gaussian, how its values come from it, and its map."""

    @classmethod
    def of_field(cls, field, gen, name):
        from .distributions.transforms import MarginalField
        from .generator import _NormalField

        p = cls()
        p.name, p.field = name, field
        bnd = getattr(field, "bnd", None)
        if bnd is not None and bnd["values"] is not None:
            raise NotImplementedError("bnd_values= conditions the field on a trace; dense samplers cover bnd= alone "
                                      "(the trace restriction)")
        p.bnd = None if bnd is None else bnd["sampler"]
        if isinstance(field, MarginalField):
            p.kind, p.gauss, p.points = "points", field.points.gauss, field.points
            p.representation = field.representation
            p.nvalues = field.space.ndof if p.representation in ("projection", "bernstein") else field.points.npoints
        elif isinstance(field, _NormalField):
            p.kind, p.gauss, p.points, p.representation = "coefficients", field.gauss, None, "coefficients"
            p.loc, p.scale, p.evaluation = np.asarray(field.loc, dtype=float), field.scale, field.evaluation
            p.nvalues = field.space.ndof if field.evaluation is None else field.evaluation.shape[0]
        else:
            raise NotImplementedError(
                f"{name}: dense samplers cover pointwise maps of one latent Gaussian (uniform, lognormal, gamma, beta, "
                "..., normal with a representation=, marginal, indicator, transformed) and Gaussian fields; this "
                "family is built from several latents or is not a function of point values")
        return p

    @classmethod
    def of_gaussian(cls, g):
        p = cls()
        p.name, p.field, p.bnd, p.kind, p.gauss, p.points = type(g).__name__, None, None, "coefficients", g, None
        p.representation = "coefficients"
        p.loc, p.scale, p.evaluation, p.nvalues = np.asarray(g.mean, dtype=float), 1.0, None, g.space.ndof
        return p

    @property
    def space(self):
        return self.gauss.space

    def output_rows(self, dofs, observe):
        """The output map on the values a draw returns (m x nvalues, CSR), with ``bnd=`` (DoFs off the trace are
        zero in a draw) folded in, and the DoF indices of the outputs (None for an observation)."""
        n = self.nvalues
        if observe is not None:
            o = observe.detach().cpu().numpy() if isinstance(observe, torch.Tensor) else observe
            o = sp.csr_matrix(o, dtype=np.float64)
            if o.shape[1] != n:
                raise ValueError(f"observe has {o.shape[1]} columns; a draw of {self.name} has {n} values")
            out_dofs = None
        else:
            if dofs is None:
                out_dofs = (np.setdiff1d(np.arange(n), self.bnd._off) if self.bnd is not None else np.arange(n))
            else:
                out_dofs = np.asarray(dofs, dtype=np.int64).ravel()
                if out_dofs.size and (out_dofs.min() < 0 or out_dofs.max() >= n):
                    raise ValueError(f"dofs must lie in [0, {n})")
            o = sp.csr_matrix((np.ones(out_dofs.size), (np.arange(out_dofs.size), out_dofs)), shape=(out_dofs.size, n))
        if self.bnd is not None:
            keep = np.ones(n)
            keep[self.bnd._off] = 0.0
            o = (o @ sp.diags(keep)).tocsr()
            o.eliminate_zeros()
        return o, out_dofs

    # -------------------------------------------------------------- the representation's linear map

    def latent_rows(self, rows):
        """``rows`` (m x nvalues) composed with the map from the latent vector to the values: the coefficients of a
        Gaussian field (with ``points=``, their evaluation), or the representation's map from the latent point values
        (Bernstein conversion, L2 projection, identity). Returns (CSR m x nlatent, largest dropped entry, relative)."""
        if self.kind == "coefficients":
            return (rows @ self.evaluation).tocsr() if self.evaluation is not None else rows, 0.0
        if self.representation in ("quadrature", "points"):
            return rows, 0.0
        return _solve_rows(self, rows)

    def forward_columns(self, rows, cols):
        """For the Bernstein representation: ``rows`` applied to the library's own conversion of the unit vectors at
        ``cols`` (dense m x |cols|), and the sum of their conversion residuals (the rounding bound of any sample)."""
        bern = self.points.bernstein
        npts = bern.npoints
        out = np.empty((rows.shape[0], len(cols)))
        resid = 0.0
        step = 256
        for j0 in range(0, len(cols), step):
            blk = cols[j0:j0 + step]
            e = np.zeros((npts, len(blk)))
            e[blk, np.arange(len(blk))] = 1.0
            c, res = bern.to_coefficients(e)
            out[:, j0:j0 + len(blk)] = rows @ c
            resid += float(np.sum(res))
        return out, resid

    # -------------------------------------------------------------- the latent

    def latent_operator(self, idx):
        """G (|idx| x nfree, CSR) with latent = G g + const for the Gaussian's coefficients g, and the latent mean."""
        s = self.gauss.space
        free = s.free
        if self.kind == "coefficients":
            pos = np.full(s.ndof, -1)
            pos[free] = np.arange(free.size)
            rows = np.flatnonzero(pos[idx] >= 0)
            g = sp.csr_matrix((np.ones(rows.size), (rows, pos[idx][rows])), shape=(len(idx), free.size))
            return g, np.asarray(self.gauss.mean, dtype=float)[idx]
        d = self.field._latent_scale()[idx]
        d = np.where(np.isfinite(d), d, 0.0)
        e = self.points.eval_full.tocsr()[idx][:, free]
        return (sp.diags(d) @ e).tocsr(), self.field._latent_mean()[idx]

    def mapping(self, idx, mean):
        """The pointwise map of the latent at ``idx`` as stored by the sampler: name, parameters, support, pull."""
        if self.kind == "coefficients":  # loc + scale (g - mean_g)
            return {"map": "affine", "params": {"shift": self.loc[idx] - self.scale * mean, "scale": float(self.scale)},
                    "support": None, "pull": None, "label": "affine"}
        t = self.field.transform
        support = getattr(t, "support", None)
        pull_range = support or getattr(t, "closed_range", None)
        spec = {"label": self.field.label, "support": _restrict_pair(support, idx, self.points.npoints),
                "pull": None if self.representation != "bernstein" or pull_range is None else
                {"range": _restrict_pair(pull_range, idx, self.points.npoints), "margin": MARGIN}}
        func = getattr(t, "func", None)
        if func is not None and func in NAMES and hasattr(t, "params"):  # ClosedForm
            spec.update(map=NAMES[func], params={k: _restrict(v, idx, self.points.npoints)
                                                 for k, v in t.params.items()})
            return spec
        from .distributions.inversion import Inversion

        if isinstance(t, Inversion):
            if t.varying or np.ndim(t.loc) or np.ndim(t.scale):
                raise NotImplementedError(f"{self.name}: a numerical inversion with parameters that vary in space has "
                                          "no dense sampler (constant parameters are supported)")
            t._resolve(10**9, True)  # the table (as d.sample() on a GPU or for large draws), or ppf
        spec.update(map="object", params={}, transform=t)
        return spec


def _restrict(v, idx, npoints):
    if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == npoints:
        return np.ascontiguousarray(v.reshape(npoints, -1)[idx, 0], dtype=np.float64)
    return float(np.asarray(v, dtype=float))


def _restrict_pair(pair, idx, npoints):
    if pair is None:
        return None
    out = []
    for b in pair:
        b = _restrict(b, idx, npoints)
        out.append(b if isinstance(b, np.ndarray) else (b if math.isfinite(b) else None))
    return tuple(out)


def _solve_rows(part, rows):
    """rows C for the representation's map C from latent point values to FE coefficients, from the transposes (one
    mass solve per row): Bernstein C = M^-1 R (same-rule mass on all dofs), projection C = M_free^-1 B_free (zero at
    fixed dofs). Entries below ZERO times their row's largest are rounding of exact zeros and dropped."""
    field, m = part.field, rows.shape[0]
    space = part.space
    if part.representation == "bernstein":
        bern = field.points.bernstein
        r = bern._operators()[1]  # ndof x npoints

        def block(rt):  # (ndof x b) -> (b x npoints)
            return np.asarray((r.T @ bern._mass.solve(rt)).T)
    else:  # projection
        from .solvers import NGSolveDirect
        from .space import mass_form

        if field._mass_solver is None:
            field._mass_solver = NGSolveDirect(space, mass_form)
        free = space.free
        bf = field.points.rule.b.tocsr()[free]

        def block(rt):
            x = field._mass_solver.apply(torch.from_numpy(np.ascontiguousarray(rt[free]))).numpy()
            return np.asarray((bf.T @ x).T)

    rt_all = rows.T.tocsc()
    pieces, dropped = [], 0.0
    for i0 in range(0, m, 256):
        rt = rt_all[:, i0:i0 + 256].toarray()
        b = block(rt)
        top = np.abs(b).max(axis=1, keepdims=True)
        small = np.abs(b) <= ZERO * top
        if small.any():
            dropped = max(dropped, float((np.abs(b) * small / np.maximum(top, 1e-300)).max()))
        b[small] = 0.0
        pieces.append(sp.csr_matrix(b))
    return sp.vstack(pieces).tocsr(), dropped


def _build_law(parts, mix, dofs, observe, rank, tol, rtol, max_points) -> dict:
    t0 = time.perf_counter()
    k = len(parts)
    p0 = parts[0]
    outs = [p.output_rows(dofs, observe) for p in parts]
    lat = [p.latent_rows(o) for p, (o, _) in zip(parts, outs)]
    support = np.unique(np.concatenate([np.unique(sp.csr_matrix(a).indices) for a, _ in lat]))
    m0 = outs[0][0].shape[0]
    at_outputs = k == 1 and p0.kind == "coefficients" and m0 < support.size
    nlat = m0 if at_outputs else support.size
    if nlat > max_points:
        raise ValueError(
            f"the outputs depend on {nlat} latent values, more than max_points={max_points} (a dense factor of "
            f"{8 * nlat**2 / 2**30:.1f} GiB): choose fewer DoFs (dofs=, bnd=) or a representation whose outputs "
            "depend on fewer values (representation='bernstein' rather than 'projection'), or raise max_points")
    comps = []
    for i, (p, (o, _), (a, _)) in enumerate(zip(parts, outs, lat)):
        if at_outputs:  # a Gaussian field observed: the observations are the latent
            a_full = sp.csr_matrix(a)
            free = p.space.free
            g_op = a_full[:, free].tocsr()
            mean = np.asarray(a_full @ np.asarray(p.gauss.mean, dtype=float)).ravel()
            spec = {"map": "affine", "label": "affine",
                    "params": {"shift": np.asarray(a_full @ p.loc).ravel() - p.scale * mean, "scale": float(p.scale)},
                    "support": None, "pull": None}
            out = None
        else:
            g_op, mean = p.latent_operator(support)
            spec = p.mapping(support, mean)
            a_s = sp.csr_matrix(a)[:, support].tocsr()
            if p.kind == "points" and p.representation == "bernstein":
                fwd, resid = p.forward_columns(o, support)
                fwd[sp.csr_matrix(a_s).toarray() == 0] = 0.0  # the pattern of the exact conversion
                a_s = sp.csr_matrix(fwd)
                if spec["pull"] is not None:
                    spec["pull"] = {**spec["pull"], "margin": max(MARGIN, 1e3 * resid)}
            out = None if _is_identity(a_s) else a_s
        spec.update(name=p.name, mean=mean, out=out, nout=o.shape[0], g=g_op)
        comps.append(spec)
    # the latent covariance (shared by the components) and its factor
    k_lat, route = _covariance(p0, comps[0]["g"])
    factor, finfo = _factor(k_lat, rtol, rank, tol)
    for c in comps:
        del c["g"]
    work64 = any(p.kind == "points" and p.representation in ("projection", "bernstein") for p in parts)
    info = {
        "law": _law_label(parts, k),
        "components": [{"name": p.name, "map": c["label"], "representation": p.representation,
                        "bnd": None if p.bnd is None else p.bnd.name,
                        **({"pull_margin": c["pull"]["margin"]} if c.get("pull") else {})}
                       for p, c in zip(parts, comps)],
        "outputs": int(m0) if len({o.shape[0] for o, _ in outs}) == 1 else [int(o.shape[0]) for o, _ in outs],
        "selection": ("observe" if observe is not None else "dofs" if dofs is not None else
                      f"trace of '{p0.bnd.name}'" if p0.bnd is not None else "all"),
        "latent": "outputs" if at_outputs else ("coefficients" if p0.kind == "coefficients" else "points"),
        "latent_points": int(nlat),
        **finfo,
        "covariance": route,
        "zero_drop": max(d for _, d in lat),
        "memory_bytes": int(factor.nbytes + sum(_nbytes(c) for c in comps)),
        "setup_s": round(time.perf_counter() - t0, 3),
        "provenance": _provenance(p0),
    }
    return {"factor": factor, "mix": np.asarray(mix, dtype=float), "components": comps, "work64": work64,
            "dofs": outs[0][1], "latent_index": None if at_outputs else support,
            "latent_coordinates": None if at_outputs else _coordinates(p0, support), "info": info}


def _is_identity(a) -> bool:
    a = sp.csr_matrix(a)
    if a.shape[0] != a.shape[1] or a.nnz != a.shape[0]:
        return False
    a.sort_indices()
    return bool(np.array_equal(a.indices, np.arange(a.shape[0])) and np.all(a.data == 1.0))


def _nbytes(c) -> int:
    n = c["mean"].nbytes + sum(v.nbytes for v in c["params"].values() if isinstance(v, np.ndarray))
    if c["out"] is not None:
        a = sp.csr_matrix(c["out"])
        n += a.data.nbytes + a.indices.nbytes + a.indptr.nbytes
    return int(n)


def _covariance(part, g):
    """K = G Sigma G^T (dense float64 torch, on the Gaussian's device) and how it was computed."""
    from .observed import operator_covariance

    if g.shape[0] == 0:
        return torch.zeros((0, 0), dtype=torch.float64), "empty"
    return operator_covariance(part.gauss, g, part.space)


def _factor(k, rtol, rank, tol):
    """F (n x r, NumPy float64) with F F^T = K up to the cutoff, in decreasing eigenvalue order; pinned rows zero."""
    n = k.shape[0]
    k = 0.5 * (k + k.T)
    diag = torch.diagonal(k).clamp_min(0.0)
    top = float(diag.max()) if n else 0.0
    pinned = diag <= rtol * top if top > 0 else torch.ones(n, dtype=torch.bool, device=k.device)
    free = torch.nonzero(~pinned).ravel()
    total = float(diag.sum())
    f = np.zeros((n, 0))
    lam = np.zeros(0)
    if free.numel():
        w, v = torch.linalg.eigh(k[free][:, free])
        w, v = w.flip(0), v.flip(1)  # decreasing
        keep = w > rtol * float(w[0]) if float(w[0]) > 0 else torch.zeros_like(w, dtype=torch.bool)
        w, v = w[keep], v[:, keep]
        exact = int(w.numel())
        if rank is not None:
            w, v = w[:int(rank)], v[:, :int(rank)]
        if tol is not None and w.numel():
            frac = torch.cumsum(w, 0) / max(total, 1e-300)
            r = int(torch.searchsorted(frac, torch.tensor(1.0 - float(tol), dtype=frac.dtype, device=frac.device))) + 1
            w, v = w[:r], v[:, :r]
        # a fixed sign per mode (largest entry positive): the factor does not depend on the eigensolver's choice
        big = v.abs().argmax(dim=0)
        sign = torch.sign(v[big, torch.arange(v.shape[1], device=v.device)])
        v = v * torch.where(sign == 0, torch.ones_like(sign), sign)
        f = np.zeros((n, w.numel()))
        f[free.cpu().numpy()] = (v * torch.sqrt(w)).cpu().numpy()
        lam = w.cpu().numpy()
    else:
        exact = 0
    captured = float(lam.sum()) / total if total > 0 else 1.0
    info = {"rank": int(f.shape[1]), "exact": f.shape[1] == exact, "captured_variance": captured,
            "dropped_trace": max(0.0, 1.0 - captured), "rtol": rtol, "pinned": int(pinned.sum()),
            "truncation": None if rank is None and tol is None else {"rank": rank, "tol": tol}}
    return f, info


def _law_label(parts, k):
    p = parts[0]
    what = "Gaussian field" if p.kind == "coefficients" else f"{p.field.label} ({p.representation})"
    return what if k == 1 else f"{k} correlated components: " + ", ".join(q.name for q in parts)


def _coordinates(part, idx):
    """Physical coordinates of the latent points (None for coefficients)."""
    if part.kind != "points":
        return None
    pts = part.points
    if pts.bernstein is not None:
        import ngsolve as ngs

        return np.column_stack([pts.bernstein.cf_values(c)[idx] for c in (ngs.x, ngs.y, ngs.z)[:part.space.dim]])
    if pts.rule is not None:
        return pts.rule.points[idx]
    return pts.coords[idx]


def _provenance(part) -> dict:
    import ngsolve as ngs

    space = part.space
    mesh = space.mesh
    h = hashlib.sha256()
    h.update(np.array([v.point for v in mesh.vertices], dtype=np.float64).tobytes())
    for el in mesh.Elements(ngs.VOL):
        h.update(np.array([v.nr for v in el.vertices], dtype=np.int64).tobytes())
    version, commit = _version()
    return {"ngsrng": version, "commit": commit, "ngsolve": getattr(ngs, "__version__", "?"),
            "space": space.describe(), "fes": space.fes.type, "order": int(space.order), "ndof": int(space.ndof),
            "mesh_sha256": h.hexdigest(), "representation": part.representation,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stream": "z ~ N(0, I) sample-major (n, k, r); host: numpy standard_normal of the sampler's stream; "
                      "device: torch.randn per block of `block` samples, seeded by the block's child key"}


def _version():
    import importlib.metadata
    import os
    import subprocess

    try:
        dist = importlib.metadata.distribution("ngsrng")
        version = dist.version
        direct = dist.read_text("direct_url.json")
        commit = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    except importlib.metadata.PackageNotFoundError:
        version, commit = "?", None
    if commit is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isdir(os.path.join(root, ".git")):
            try:
                commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                                        timeout=10).stdout.strip() or None
            except (OSError, subprocess.SubprocessError):
                commit = None
    return version, commit
