"""Dense samplers: a field's draws at chosen DoFs, or under a linear observation, as small dense matrix products.

    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    A = d.dense(device="cuda")                     # the FE coefficients at the trace DoFs (bnd=), or at dofs=[...]
    B = (E @ d).dense(device="cuda")               # E @ the coefficients, E any (m x ndof) matrix; = d.dense(observe=E)
    C = E @ A                                      # a dense sampler composed with E (no new set-up)
    c = A.sample(n, generator=gen)                 # (n, len(A.dofs)); correlated fields: (n, k, m)
    A.save("inflow.npz")
    A2 = ngsrng.DenseSampler.load("inflow.npz", device="cuda")   # the stored law: no mesh, space or assembly

``dense()`` exists on every distribution (``rng.distribution(...)``), on ``ngsrng.correlated(...)``, on observed laws
(``E @ d``: an :class:`ObservedField`, or an :class:`ngsrng.ObservedGaussian` for Gaussian fields), on the
Gaussians (``Gaussian``, ``LowRankGaussian``, ``MatheronPosterior``, ``StationaryMatern``: their coefficients) and on
mixtures (``ngsrng.mixture(...)``: one dense law per law of the mixture, drawn together). The set-up is made once per
object and arguments and kept there (it survives pickling, and ``Generator.spawn`` children share it); draws are
matrix products, on any device.

What is dense, and why it is exact
----------------------------------
Every family it covers is a pointwise map of one latent Gaussian, followed by linear maps::

    z ~ N(0, I_r)  ->  w = m + F z     latent values at the points S that determine the outputs
                   ->  v = T(w)        the family's pointwise map (an affine map for Gaussian fields)
                   ->  y = A v         the representation's linear map to the outputs (FE coefficients at the chosen
                                       DoFs, composed with bnd= and the observation)

- **The points S** are those the outputs depend on, found exactly, without a threshold. For
  ``representation="bernstein"`` an NGSolve coefficient on a mesh node (vertex, edge, face, cell) depends exactly on
  the Bernstein domain points in that node's closure (a trace: the domain points on the boundary). For values at
  points (the default ``"quadrature"`` representation, ``points=``) and for Gaussian FE fields, S is the set the output
  map reads; the L2 projection (``"projection"``) reads every point. A Gaussian field observed by fewer observations
  than the DoFs they read has the observations themselves as its latent. Entries of the output map (an observation,
  an evaluation, projection rows) at most ``zero`` (default 1e-15) times their row's largest are taken as rounding of
  exact zeros (a point evaluation gives ~1e-17 to basis functions that vanish at the point; ``info["zero_drop"]``
  records the largest dropped); ``zero=None`` keeps every stored entry.
- **The latent covariance** K = G Sigma G^T at S (G the evaluation at S, standardised as the family standardises its
  latent; Sigma the coefficient covariance, conditioning included) is the exact discrete covariance of what
  ``d.sample()`` draws. It is never formed: a QR of Y = L^T G^T (|S| applications of the factor L^T) gives R with
  R^T R = K to the rounding of Y (:func:`ngsrng.observed.covariance_factor`).
- **The factor** F: with ``rtol`` (default 1e-12) the SVD of R, cut at eigenvalues below rtol times the largest
  (||K - F F^T||_2 <= rtol max eig K; each direction's standard deviation off by at most sqrt(rtol) of the largest),
  as one thin matrix whose columns are the Karhunen-Loeve modes at S, in decreasing order, so ``rank=`` / ``tol=``
  truncate it optimally (``info["exact"]`` and ``info["captured_variance"]`` say what was kept). ``rtol=None`` (or 0)
  without rank / tol: no truncation, F = R^T. Points with exactly zero variance (Dirichlet DoFs, points a family
  standardises to a constant) get exactly zero rows: they equal their mean, without any jitter; ``pin=`` also pins
  points whose variance is at most pin times the largest.
- **Gaussian fields fold**: their map is affine, so y = A m' + (A F) z; when one m x r' matrix (A F, recompressed by an
  SVD to r' <= min(m, r) columns) costs less than F and A (|S| r + nnz(A) per sample), the output map is folded into
  the factor. Decided at set-up and after each composition. Pointwise families keep F and A apart (T is in between).
- **Correlated components** (``ngsrng.correlated``) share K: the joint factor is L_R (x) F (R the cross-correlation),
  and each component keeps its own mean, map and output map.
- **Mixtures** (``ngsrng.mixture``): K laws with weights, each set up as above (its own S, factor, maps; ``rank=`` and
  ``tol=`` relative to its own variance) with the same components and output sizes. A draw's label l (law l with
  probability weights[l]) chooses its law; a batch costs one product per law, for the draws that follow it.
- **The Bernstein range guarantee** holds as in ``d.sample()``: the map comes before the linear conversion, and the
  values are pulled inside the family's range by a relative margin of at least 1000 times the conversion's measured
  rounding (summed over S, so it bounds every sample) before it.
- **Precision**: the noise z, the dense product w = m + F z (the cost of a draw) and the closed-form maps run in the
  sampler's dtype. Numerical inversions and pointwise functions run in float64 for point values in the projection and
  Bernstein representations, whose conversions to coefficients are ill-conditioned (the margin above is measured in
  float64), and so do the clamps, the pull and the conversion there; in the sampler's dtype otherwise. With host
  noise a float32 sampler's draws are the float64 sampler's (the same z) up to float32 rounding.
- **Host synchronisation**: a draw on a GPU into torch tensors only queues work on the device, so the caller can queue
  its own behind it: the output maps are applied by gathers (torch's sparse products wait for the device on some
  GPUs), host noise and labels are copied without waiting. NumPy output and functions without a torch form wait.

The law is that of ``d.sample()`` (its marginal at the chosen DoFs, or its image under the observation), up to the
cut and rounding; the draws themselves differ (another random stream). Data drawn by either route can be mixed. Not
covered: ``bnd_values=`` (conditioning on a trace), families built from several latents (chi-square sums, Student-t,
Wishart, Dirichlet, ...), Karhunen-Loeve fields, and spatially varying shape parameters of numerically inverted
families.

Observations: two syntaxes, both exact
--------------------------------------
``(E @ d).dense()`` (or ``d.dense(observe=E)``) sets up for the observations: only what E reads. ``E @ d.dense()``
composes an existing dense sampler with E, without a new set-up (also a loaded one): the latent points E no longer
reads are dropped (a Gaussian's marginal: rows of its factor), the factor is recompressed when it has more columns
than rows, and Gaussian components fold as above. Without truncation the two give the same law at the same cost per
draw; with ``rank=`` / ``tol=`` they differ, both correctly: the first keeps the best modes for the observations, the
second those of the field.

Size: up to ``MAX_POINTS`` latent points by default (``max_points=`` moves it). The set-up costs |S| applications of
L^T, a QR and an SVD of |S| columns; a draw costs |S| r + nnz(A) (folded: m r') multiply-adds per sample and
component (of its law, for a mixture).

Randomness
----------
As every sampler (:mod:`ngsrng.seeding`): ``seed=None`` reads torch's global generator at each draw, an int is an own
stream, and ``sample(n, generator=g)`` reads ``g`` for that draw (one key per draw). The draws are sample-major
standard normals z (n, k, r), r = ``rank`` (for a mixture, of its widest law: a draw following a law of rank r_l uses
z[:, :, :r_l]), and for a mixture the labels, from a stream of their own (NumPy, on the host: the same labels on every
device, and the sizes of the laws' groups are known there without waiting for the device); every sample's z and label
are fixed by its position in the streams, so the draws do not depend on how a batch is split (``sample(4)`` then
``sample(6)`` is ``sample(10)``: the same z, and the same values up to the rounding of products over other batch
sizes). ``noise="host"`` (the CPU's default) draws z with NumPy (float64) and
copies it to the device: the same samples on every device. ``noise="device"`` (a GPU's default) draws z on the device
with torch in the sampler's dtype, keyed per block of ``block`` samples, a power of two (fast, and a batch of a
power-of-two size draws no normals it does not use; the samples then depend on the device and the dtype, as torch's
generators do).

Stored samplers
---------------
``save(path)`` writes one ``.npz`` (format ``ngsrng.dense/2``) without pickles or code. ``meta`` (JSON) holds the
format, the dtype, the laws' ``weights`` and per law its description (``laws[l]``: the components' maps, output sizes
and the keys of their arrays, and ``info``, with provenance: ngsrng version and commit, space, mesh hash,
representation, stream convention). The arrays of law l have keys ``law{l}/...``: the factor, the mixing of correlated
components, the means, the maps' parameters and ranges (closed forms: family, parameters and range; numerical
inversions: the scipy.stats distribution's name and parameters; ``transformed`` with a NumPy or torch function: its
name), the output maps (COO), the DoF indices and the latent points. One law is K = 1.
``DenseSampler.load(path, device=...)`` rebuilds the sampler from these arrays alone. Without ngsrng, a stored sampler
draws with NumPy and SciPy::

    import json, numpy as np, scipy.stats as st
    from scipy.special import ndtr
    s = np.load("sampler.npz")
    meta = json.loads(str(s["meta"]))
    gen = np.random.default_rng(0)
    labels = gen.choice(len(meta["weights"]), size=n, p=meta["weights"])   # the law each draw follows
    out = [np.empty((n, c["nout"])) for c in meta["laws"][0]["components"]]
    for l, law in enumerate(meta["laws"]):
        a = lambda key: s[f"law{l}/{key}"]
        rows = labels == l
        z = gen.standard_normal((rows.sum(), a("mix").shape[0], a("factor").shape[1]))
        z = np.einsum("ij,njr->nir", a("mix"), z)                  # correlate the components
        for i, c in enumerate(law["components"]):
            w = a(f"mean{i}") + z[:, i] @ a("factor").T            # latent values (n_l, |S|)
            p = {k: s[v] if isinstance(v, str) else v for k, v in c["params"].items()}
            if c["map"] == "inversion":                           # e.g. gamma: loc + scale F^-1(Phi(w))
                q = c["stored"]
                v = q["loc"] + q["scale"] * getattr(st, q["scipy"])(*q["args"], **q["kwds"]).ppf(ndtr(w))
            else:
                v = {"uniform": lambda w, low, high: low + (high - low) * ndtr(w),
                     "lognormal": lambda w, mean, sigma: np.exp(mean + sigma * w),
                     "affine": lambda w, shift, scale: shift + scale * w}[c["map"]](w, **p)
            if c["out_shape"]:                                     # the output map (none: the values themselves)
                m = np.zeros(c["out_shape"])
                m[a(f"out{i}_rows"), a(f"out{i}_cols")] = a(f"out{i}_vals")
                v = v @ m.T
            out[i][rows] = v                                       # (n, m): component i

(the other closed forms are ``transforms.CLOSED``; ``DenseSampler.from_noise(z, labels)`` gives the same values from
the same z and labels, up to the range margins of bounded families, which move values within ~1e-9 of a bound by at
most that, and the 1e-10 accuracy of inversion tables.)
"""

import hashlib
import importlib
import json
import math
import numbers
import time

import numpy as np
import scipy.sparse as sp
import torch

from .backend import DeviceCache, owned_cache, resolve_device, resolve_dtype, sparse_to_torch
from .inplace import InPlace, as_tensor, default_work_values
from .seeding import Source, as_source, inherit, one_draw

__all__ = ["DenseSampler", "ObservedField"]

FORMAT = "ngsrng.dense/2"
MAX_POINTS = 10_000  # latent points of a dense factor by default (|S|^2 float64: 0.8 GB at the limit)
RTOL = 1e-12  # the default cut: eigenvalues of K below rtol times the largest
# Entries of an output map at most ZERO times their row's largest are taken as rounding of exact zeros (a point
# evaluation gives ~1e-17 to basis functions that vanish at the point); configurable per call (zero=). To do: decide
# whether a threshold belongs here at all (docs: "Look into later").
ZERO = 1e-15
BLOCK_NORMALS = 2**22  # standard normals per keyed block of device noise at most (performance, not accuracy)
# Output maps on the device: dense up to DENSE_OUT entries unless at most 1 / GATHER_FILL of them is stored; else by
# gathers of their stored entries (rows padded to a common number of entries per bucket; performance)
DENSE_OUT = 2**22
GATHER_FILL = 8


def _affine(w, xp, shift, scale):
    return shift + scale * w


def _maps():
    """name -> pointwise map (the closed forms, the excursion indicator, the affine map of Gaussian fields)."""
    global _MAPS
    if _MAPS is None:
        from .distributions.transforms import CLOSED, _exceeds

        _MAPS = {name: entry[0] for name, entry in CLOSED.items()} | {"indicator": _exceeds, "affine": _affine}
    return _MAPS


_MAPS = None
LIVE = ("inversion", "pointmap", "object")  # maps held as objects of the process (rebuilt on load where stored)


def _xp(torch_mode):
    from .distributions.pointwise import XP

    return XP(torch_mode)


def _margin():
    from .distributions.pointwise import PointwiseField

    return PointwiseField.MARGIN


class DenseSampler(InPlace):
    """Draws of a field at chosen DoFs or under a linear observation, by dense products (see :mod:`ngsrng.dense`).

    Made by ``dense()`` (on a distribution, ``ngsrng.correlated(...)``, an observed law, a Gaussian or
    ``ngsrng.mixture(...)``), by ``E @ sampler`` or by :meth:`load`. ``sample(n)`` gives (n, m), or (n, k, m) for k
    correlated components (a tuple when their output sizes differ); ``dofs`` (the DoF indices of the outputs, None for
    an observation), ``rank``, ``info``, :meth:`save`, :meth:`to`.

    A sampler holds K >= 1 ``laws`` with ``weights``, a finite mixture: each draw follows law l with probability
    weights[l] (the draw's label). The laws have the same components and output sizes. One law: K = 1.
    """

    __array_ufunc__ = None  # E @ sampler: NumPy (and SciPy, torch) leave the product to __rmatmul__

    def __init__(self, laws, weights=None, *, device="cpu", dtype=None, noise=None, seed=None, output=None,
                 work_values=None):
        self.laws = (laws,) if isinstance(laws, dict) else tuple(laws)
        self.weights = _mixture_weights(weights, len(self.laws))
        _check_outputs(self.laws)
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype) if dtype is not None else torch.float64
        self.noise = noise or ("host" if self.device.type == "cpu" else "device")
        if self.noise not in ("host", "device"):
            raise ValueError("noise must be 'host' or 'device'")
        self.source = as_source(seed)
        self.output = output
        self.work_values = default_work_values(self.device) if work_values is None else int(work_values)
        self._device = DeviceCache()
        self.block = _floor_pow2(BLOCK_NORMALS // max(1, self.components * self.rank))  # samples per keyed block
        # laws that share the map, bounds and output maps (drawn by one conversion), in the order of their draws
        stages = {}
        for l, law in enumerate(self.laws):
            stages.setdefault(_stage_key(law), []).append(l)
        self._stages = list(stages.values())
        self._position = np.empty(len(self.laws), dtype=np.int64)
        self._position[[l for stage in self._stages for l in stage]] = np.arange(len(self.laws))

    # ------------------------------------------------------------------ description

    @property
    def components(self) -> int:
        """Number of jointly drawn components (1 for a single field)."""
        return len(self.laws[0]["components"])

    @property
    def rank(self) -> int:
        """Standard normals per draw and component: the columns of the latent factor (of the widest law; a law
        with fewer columns reads the leading ones)."""
        return max(int(law["factor"].shape[1]) for law in self.laws)

    @property
    def latent_points(self) -> int:
        """Latent values (of the law with the most)."""
        return max(int(law["factor"].shape[0]) for law in self.laws)

    @property
    def dofs(self):
        """The indices of the outputs among the values ``d.sample()`` returns (FE DoFs for coefficient
        representations), or None for an observation (or laws with different DoFs)."""
        first = self.laws[0].get("dofs")
        if first is None or any(law.get("dofs") is None or not np.array_equal(law["dofs"], first)
                                for law in self.laws[1:]):
            return None
        return first

    @property
    def nout(self):
        sizes = [c["nout"] for c in self.laws[0]["components"]]
        return sizes[0] if len(set(sizes)) == 1 else tuple(sizes)

    @property
    def info(self) -> dict:
        if len(self.laws) == 1:
            law = self.laws[0]["info"]
        else:
            law = {"law": f"mixture of {len(self.laws)} laws", "weights": self.weights.tolist(),
                   "laws": [law["info"] for law in self.laws],
                   "memory_bytes": sum(int(law["info"].get("memory_bytes", 0)) for law in self.laws)}
        return {**law, "device": str(self.device), "dtype": str(self.dtype).replace("torch.", ""),
                "noise": self.noise, "block": self.block, "randomness": self.source.describe()}

    def __repr__(self):
        if len(self.laws) > 1:
            ranks = [int(law["factor"].shape[1]) for law in self.laws]
            return (f"DenseSampler(mixture of {len(self.laws)} laws: {self.components} x {self.nout} outputs, rank "
                    f"{min(ranks)}..{max(ranks)}, device={self.device})")
        i = self.laws[0]["info"]
        return (f"DenseSampler({i.get('law', '')}: {self.components} x {self.nout} outputs, {self.latent_points} "
                f"latent values, rank {self.rank}, {'exact' if i.get('exact') else 'truncated'}, device={self.device})")

    def to(self, device=None, dtype=None, noise=None, seed=None, output=None) -> "DenseSampler":
        """The same law with other sampling settings. What is not given is kept: the set-up, and the randomness (the
        same stream, continued by both samplers); ``noise`` follows a new device's default."""
        if noise is None and device is None:
            noise = self.noise
        return DenseSampler(self.laws, self.weights, device=self.device if device is None else device,
                            dtype=self.dtype if dtype is None else dtype, noise=noise,
                            seed=self.source if seed is None else seed,
                            output=self.output if output is None else output)

    def __rmatmul__(self, operator):
        """``E @ sampler``: the outputs observed by E (m2 x m), a dense sampler again, without a new set-up: the latent
        values E no longer reads are dropped, the factor recompressed, Gaussian components folded where cheaper (per
        law of a mixture). Same randomness (stream) and sampling settings."""
        from .observed import _as_operator

        if not isinstance(self.nout, int):
            raise ValueError("E @ sampler: the components have outputs of different sizes")
        e = _as_operator(operator, self.nout)
        return DenseSampler([_compose(law, e) for law in self.laws], self.weights, device=self.device,
                            dtype=self.dtype, noise=self.noise, seed=self.source, output=self.output,
                            work_values=self.work_values)

    # ------------------------------------------------------------------ draws

    def sample(self, n=None, generator=None, output=None, out=None, labels=False):
        """n draws: (n, m), or (n, k, m) for correlated components; ``n=None``: one, without the sample axis.

        ``generator``: a torch or NumPy generator read for this draw (one key), instead of the sampler's own
        randomness. ``output``: "numpy" or "torch" (default: NumPy on the CPU, torch tensors on a GPU). ``out``: a
        buffer of that shape (any float dtype and device, or one per component) filled in place and returned.
        ``labels``: also return each draw's label, the index of the law it follows (int64, (n,); zeros for one
        law), as ``(samples, labels)``."""
        source = self.source if generator is None else Source(generator)
        with one_draw():
            if out is not None:
                lab = self._sample_out(n, out, source)
                res, count = out, as_tensor(out[0] if isinstance(out, (list, tuple)) else out).shape[0]
            else:
                count = 1 if n is None else int(n)
                rows = [torch.empty((count, c["nout"]), dtype=self.dtype, device=self.device)
                        for c in self.laws[0]["components"]]
                lab = self._fill(rows, source)
                res = self._deliver(rows, n, output)
        if not labels:
            return res
        lab = np.zeros(count, dtype=np.int64) if lab is None else lab
        if (output or self.output or ("numpy" if self.device.type == "cpu" else "torch")) == "torch":
            lab = _to_device(torch.from_numpy(lab), self.device)
        return res, (lab[0] if n is None and out is None else lab)

    rvs = sample

    def sample_into(self, rows):
        """Fill (n, m) rows in place (one buffer per component, or one (n, k, m) buffer)."""
        with one_draw():
            self._sample_out(None, rows, self.source)
        return rows

    def from_noise(self, z, labels=None, output=None):
        """The samples made from given standard normals ``z``: (n, k, r) (or (n, r) for one component, r =
        ``rank``), and for a mixture the draws' ``labels`` (n,), as ``sample`` returns them: the deterministic part
        of a draw (for antithetic or quasi-random z, or to check a stored sampler against a reference
        implementation). A draw following a law with fewer columns reads the leading ones of its z."""
        z = torch.as_tensor(z)
        if z.ndim == 2:
            z = z[:, None, :]
        k, r, n = self.components, self.rank, z.shape[0]
        if tuple(z.shape[1:]) != (k, r):
            raise ValueError(f"z must have shape (n, {k}, {r}), got {tuple(z.shape)}")
        lab = None
        if len(self.laws) > 1:
            if labels is None:
                raise ValueError(f"a mixture of {len(self.laws)} laws: give the draws' labels (n,)")
            lab = np.asarray(labels, dtype=np.int64).reshape(-1)
            if lab.shape != (n,) or (n and (lab.min() < 0 or lab.max() >= len(self.laws))):
                raise ValueError(f"labels: n = {n} integers in [0, {len(self.laws)})")
        rows = [torch.empty((n, c["nout"]), dtype=self.dtype, device=self.device) for c in self.laws[0]["components"]]
        self._write(rows, 0, n, z.to(self.device, self.dtype).reshape(n, k * r), lab)
        return self._deliver(rows, n, output)

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
        for r, c in zip(rows, self.laws[0]["components"]):
            if r.ndim != 2 or r.shape[1] != c["nout"] or (n is not None and r.shape[0] != int(n)):
                raise ValueError(f"out holds {tuple(r.shape)} values per component, expected (n, {c['nout']})")
        return self._fill(rows, source)

    def _fill(self, rows, source):
        """Fill the rows with draws in chunks of ``work_values`` values; the draws' labels (None for one law)."""
        n, many = rows[0].shape[0], len(self.laws) > 1
        labels = np.empty(n, dtype=np.int64) if many else None
        if n == 0:
            return labels
        k, r = self.components, self.rank
        per = k * r + max(k * 2 * int(law["factor"].shape[0]) for law in self.laws) + \
            sum(c["nout"] for c in self.laws[0]["components"])
        step = max(1, int(self.work_values // max(per, 1)))
        streams = source.streams()
        edges = np.cumsum(self.weights)[:-1]
        for k0 in range(0, n, step):
            k1 = min(n, k0 + step)
            lab = None
            if many:  # the labels from their own stream (on the host: the groups' sizes are known there)
                lab = labels[k0:k1] = np.searchsorted(edges, streams.labels().random(k1 - k0), side="right")
            self._write(rows, k0, k1, self._standard_normal(k1 - k0, k * r, streams), lab)
        return labels

    def _write(self, rows, k0, k1, z, labels):
        """rows[k0:k1] from the standard normals z (k1 - k0, k r) and the draws' labels (None: one law). A mixture's
        draws are sorted by law (one gather of z): one product per law into its slice of the latent values, then the
        map and the conversion once per stage (laws whose map, bounds and output maps are the same: the same family
        and representation under, e.g., another correlation length), and one scatter back to the draws' rows."""
        if labels is None:
            for row, y in zip(rows, self._transform(0, z)):
                row[k0:k1].copy_(y)
            return
        k, r = self.components, self.rank
        order = np.argsort(self._position[labels], kind="stable")
        counts = np.bincount(labels, minlength=len(self.laws))
        perm = _to_device(torch.from_numpy(order), self.device)
        zs = z.view(-1, k, r).index_select(0, perm)
        at = 0
        for stage in self._stages:
            size = int(counts[stage].sum())
            if not size:
                continue
            nlat = int(self.laws[stage[0]]["factor"].shape[0])
            ws = [torch.empty((size, nlat), dtype=self.dtype, device=self.device) for _ in range(k)]
            s0 = 0
            for l in stage:
                c, rl = int(counts[l]), int(self.laws[l]["factor"].shape[1])
                if c:
                    self._latent(l, zs[at + s0:at + s0 + c, :, :rl].reshape(c, k * rl), [w[s0:s0 + c] for w in ws])
                    s0 += c
            idx = perm[at:at + size]
            for row, y in zip(rows, self._convert(stage[0], ws)):
                where = idx if row.device == idx.device else torch.from_numpy(order[at:at + size]).to(row.device)
                row[k0:k1].index_copy_(0, where, y.to(row.device, row.dtype))
            at += size

    def _conversion_dtype(self, l):
        """The dtype of law l's conversion to the outputs: float64 for point values in the projection and Bernstein
        representations (ill-conditioned conversions), the sampler's otherwise."""
        return torch.float64 if self.laws[l]["work64"] else self.dtype

    def _standard_normal(self, n, width, streams):
        """(n, width) standard normals of the next n samples of the stream, on the device in the sampler's dtype (see
        the module docstring: host noise from NumPy, device noise keyed per block of samples)."""
        if self.noise == "host":
            z = torch.from_numpy(streams.numpy().standard_normal((n, width))).to(self.dtype)
            return _to_device(z, self.device)
        seq, first = streams.reserve(n)
        out = torch.empty((n, width), dtype=self.dtype, device=self.device)
        gen, b = self._generator(), self.block
        for blk in range(first // b, (first + n - 1) // b + 1):
            lo, hi = max(first, blk * b), min(first + n, (blk + 1) * b)
            gen.manual_seed(_block_seed(seq, blk))
            dst = out[lo - first:hi - first]
            if hi - lo == b:  # a whole block: drawn in place
                torch.randn((b, width), generator=gen, dtype=self.dtype, device=self.device, out=dst)
            else:
                dst.copy_(torch.randn((b, width), generator=gen, dtype=self.dtype, device=self.device)[lo - blk * b:
                                                                                                    hi - blk * b])
        return out

    def _generator(self):
        key = ("generator", str(self.device))
        if key not in self._device:
            self._device[key] = torch.Generator(device=self.device)
        return self._device[key]

    def _transform(self, l, z):
        """Samples of law l (one (n, m) tensor per component) from standard normals z (n, k r_l) in the sampler's
        dtype on the device: the dense product w = m + F z (the cost of a draw) and the closed-form maps in the
        sampler's dtype; numerical inversions and pointwise functions, the clamps, the Bernstein pulls and the
        conversion to the outputs in the conversion dtype (float64 for point values in the projection and Bernstein
        representations, whose conversions to coefficients are ill-conditioned)."""
        return self._convert(l, self._latent(l, z))

    def _latent(self, l, z, out=None):
        """Law l's latent values w = m + F z per component, (n, |S|) in the sampler's dtype (into ``out``, one
        tensor per component, if given)."""
        law = self.laws[l]
        k, r, n = self.components, int(law["factor"].shape[1]), z.shape[0]
        lin = self._linear(l)
        x = z.reshape(n, k, r)
        if k > 1:
            x = torch.einsum("ij,njr->nir", lin["mix"], x)
        ws = []
        for i in range(k):
            dst, mean = None if out is None else out[i], lin["means"][i]
            if r:
                ws.append(torch.addmm(mean, x[:, i], lin["factor"].T, out=dst))
            else:
                ws.append(mean.expand(n, -1).clone() if dst is None else dst.copy_(mean.expand(n, -1)))
        return ws

    def _convert(self, l, ws):
        """The outputs from latent values ws (per component) with law l's maps and conversions."""
        out = []
        for i, (c, w) in enumerate(zip(self._conversions(l), ws)):
            v = self._map(l, i, c, w)
            out.append(v if c["out"] is None else c["out"](v))
        return out

    def _map(self, l, i, c, w):
        spec = self.laws[l]["components"][i]
        cd = self._conversion_dtype(l)
        if spec["map"] in LIVE:  # a numerical inversion or a pointwise function (an object of the process)
            w = w.to(cd)
            t = spec["transform"]
            if getattr(t, "torch_capable", False):
                v = t(w, _xp(True))
            else:
                v = torch.as_tensor(t(w.double().cpu().numpy(), _xp(False)), device=w.device).to(cd)
        else:
            v = _maps()[spec["map"]](w, _xp(True), **c["params"]).to(cd)
        # strictly inside the support in the output precision, and for Bernstein inside the range by a margin safe
        # against the conversion's rounding (relative to the largest value of the sample): one clamp to the tighter
        # bounds
        low, high = c["low"], c["high"]
        if c["pull"] is not None:
            a, b = torch.aminmax(v, dim=1, keepdim=True)
            margin = spec["pull"]["margin"] * torch.maximum(a.abs(), b.abs()).clamp_min(1e-300)
            for bound, sign in c["pull"]:
                if sign > 0:
                    low = bound + margin if low is None else torch.maximum(low, bound + margin)
                else:
                    high = bound - margin if high is None else torch.minimum(high, bound - margin)
        if low is not None or high is not None:
            v = torch.clamp(v, min=low, max=high)
        return v

    def _tensor(self, dt):
        dev = self.device
        return lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64)).to(dev, dt)

    def _linear(self, l):
        """Law l's dense product on the device in the sampler's dtype (made once per device and dtype): the factor,
        the mixing of correlated components and the components' means."""
        key = ("linear", l, str(self.device), self.dtype)
        if key not in self._device:
            law, t = self.laws[l], self._tensor(self.dtype)
            self._device[key] = {"factor": t(law["factor"]), "mix": t(law["mix"]),
                                 "means": [t(spec["mean"]) for spec in law["components"]]}
        return self._device[key]

    def _conversions(self, l):
        """Law l's maps and conversions on the device (made once per device and dtype): the closed-form maps'
        parameters in the sampler's dtype; the output maps (:class:`_OutputMap`), the clamps and the Bernstein pulls
        in the conversion dtype."""
        key = ("conversions", l, str(self.device), self.dtype)
        if key not in self._device:
            cd = self._conversion_dtype(l)
            t, tc = self._tensor(self.dtype), self._tensor(cd)
            comps = []
            for spec in self.laws[l]["components"]:
                params = {k: (t(v) if isinstance(v, np.ndarray) else v) for k, v in spec["params"].items()}
                low, high = _clamp_bounds(spec.get("support"), self.dtype, tc)
                comps.append({"params": params,
                              "out": None if spec["out"] is None else _OutputMap(spec["out"], self.device, cd),
                              "low": low, "high": high, "pull": _pull_bounds(spec.get("pull"), tc)})
            self._device[key] = comps
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
        """Write the sampler's laws and weights to one ``.npz`` file, without pickles or code (see the module
        docstring for its layout and a NumPy reference sampler). Every map is stored but ``transformed`` with a
        function other than a NumPy or torch one (which would need its code)."""
        arrays, laws = {}, []
        for l, law in enumerate(self.laws):
            laws.append(_store_law(law, arrays, f"law{l}/"))
        meta = {"format": FORMAT, "dtype": str(self.dtype).replace("torch.", ""), "weights": self.weights.tolist(),
                "laws": laws}
        np.savez(path, meta=np.array(json.dumps(meta, default=_json_default)), **arrays)

    @classmethod
    def load(cls, path, device="cpu", dtype=None, noise=None, seed=None, output=None) -> "DenseSampler":
        """A sampler from a file written by :meth:`save` (its arrays alone: no mesh, space or assembly). ``dtype``
        defaults to the dtype it was saved with."""
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        meta = json.loads(str(arrays.pop("meta")))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: not a stored ngsrng dense sampler of format {FORMAT} (found "
                             f"{meta.get('format')!r})")
        laws = [_load_law(entry, arrays, f"law{l}/") for l, entry in enumerate(meta["laws"])]
        saved = meta.get("dtype")
        return cls(laws, meta["weights"], device=device, dtype=dtype if dtype is not None else saved, noise=noise,
                   seed=seed, output=output)


def _mixture_weights(weights, k) -> np.ndarray:
    """The weights of k laws as probabilities (None: equal)."""
    if k < 1:
        raise ValueError("a dense sampler needs at least one law")
    if weights is None:
        return np.full(k, 1.0 / k)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != (k,) or not np.all(np.isfinite(w)) or (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"weights: {k} nonnegative finite numbers, not all zero")
    return w / w.sum()


def _check_outputs(laws):
    """The laws of one sampler draw the same components with the same output sizes."""
    first = [c["nout"] for c in laws[0]["components"]]
    for l, law in enumerate(laws[1:], 1):
        if [c["nout"] for c in law["components"]] != first:
            raise ValueError(f"law {l} has outputs {[c['nout'] for c in law['components']]} per component, law 0 "
                             f"{first}: the laws of a mixture have the same components and output sizes")


def _store_law(law, arrays, prefix) -> dict:
    """Law ``law`` into ``arrays`` (keys with ``prefix``); its description (JSON) with the keys of its arrays."""
    arrays[prefix + "factor"], arrays[prefix + "mix"] = law["factor"], law["mix"]
    for key in ("dofs", "latent_index", "latent_coordinates"):
        if law.get(key) is not None:
            arrays[prefix + key] = np.asarray(law[key])
    comps = []
    for i, spec in enumerate(law["components"]):
        if spec["map"] == "object":
            raise NotImplementedError(
                f"component {spec['name']}: its map ({spec['label']}) is a user function, whose code a file "
                "without pickles cannot hold (NumPy and torch functions are stored by name)")
        arrays[f"{prefix}mean{i}"] = spec["mean"]
        params = {}
        for name, v in spec["params"].items():
            if isinstance(v, np.ndarray):
                arrays[f"{prefix}param{i}_{name}"] = v
                params[name] = f"{prefix}param{i}_{name}"
            else:
                params[name] = v
        entry = {"name": spec["name"], "map": spec["map"], "params": params, "nout": int(spec["nout"]),
                 "support": _store_pair(spec.get("support"), arrays, f"{prefix}support{i}"),
                 "pull": None if spec.get("pull") is None else
                 {"range": _store_pair(spec["pull"]["range"], arrays, f"{prefix}range{i}"),
                  "margin": float(spec["pull"]["margin"])},
                 "out_shape": None, "stored": spec.get("stored") if spec["map"] in LIVE else None}
        if spec["out"] is not None:
            a = sp.coo_matrix(spec["out"])
            arrays[f"{prefix}out{i}_rows"], arrays[f"{prefix}out{i}_cols"] = a.row, a.col
            arrays[f"{prefix}out{i}_vals"] = a.data
            entry["out_shape"] = list(a.shape)
        comps.append(entry)
    return {"work64": bool(law["work64"]), "components": comps, "info": law["info"]}


def _load_law(entry, arrays, prefix) -> dict:
    comps = []
    for i, c in enumerate(entry["components"]):
        params = {k: arrays[v] if isinstance(v, str) and v in arrays else v for k, v in c["params"].items()}
        pull = c.get("pull")
        out = None
        if c["out_shape"] is not None:
            out = sp.csr_matrix((arrays[f"{prefix}out{i}_vals"],
                                 (arrays[f"{prefix}out{i}_rows"], arrays[f"{prefix}out{i}_cols"])),
                                shape=tuple(c["out_shape"]))
        spec = {"name": c["name"], "map": c["map"], "params": params, "nout": c["nout"],
                "mean": arrays[f"{prefix}mean{i}"], "out": out, "label": c["map"],
                "support": _load_pair(c.get("support"), arrays),
                "pull": None if pull is None else {"range": _load_pair(pull["range"], arrays),
                                                   "margin": pull["margin"]}}
        if c["map"] in LIVE:
            spec.update(stored=c["stored"], transform=_rebuild_map(c["map"], c["stored"]))
        comps.append(spec)
    return {"factor": arrays[prefix + "factor"], "mix": arrays[prefix + "mix"], "components": comps,
            "work64": entry["work64"], "dofs": arrays.get(prefix + "dofs"),
            "latent_index": arrays.get(prefix + "latent_index"),
            "latent_coordinates": arrays.get(prefix + "latent_coordinates"), "info": entry["info"]}


def _block_seed(seq, block) -> int:
    """The torch seed of one block of device noise: a child of the draw's normal stream (its key and the block)."""
    child = np.random.SeedSequence(seq.entropy, spawn_key=tuple(seq.spawn_key) + (int(block),))
    return int(child.generate_state(1, np.uint64)[0] >> np.uint64(1))


def _floor_pow2(x) -> int:
    """The largest power of two at most x (1 for x < 2)."""
    return 1 << (max(1, int(x)).bit_length() - 1)


def _to_device(t, device):
    """A host tensor on ``device``, without waiting for the device's queue (the driver stages a non-blocking copy from
    pageable memory before it returns)."""
    if device.type == "cpu":
        return t
    return t.to(device, non_blocking=True)


def _stage_key(law):
    """What a law's conversion depends on beyond its latent values: the number of latent values and, per component,
    the map, its parameters, the support, the pull and the output map."""
    def pair(p):
        return None if p is None else tuple(_digest(np.asarray(b)) if isinstance(b, np.ndarray) else b for b in p)

    key = [bool(law["work64"]), int(law["factor"].shape[0])]
    for c in law["components"]:
        live = None
        if c["map"] in LIVE:
            stored = c.get("stored")
            live = json.dumps(stored, sort_keys=True, default=_json_default) if stored else ("object",
                                                                                              id(c.get("transform")))
        key.append((c["map"], tuple(sorted((k, _digest(v)) for k, v in c["params"].items())), pair(c.get("support")),
                    None if c.get("pull") is None else (pair(c["pull"]["range"]), float(c["pull"]["margin"])),
                    int(c["nout"]), None if c["out"] is None else _digest(sp.csr_matrix(c["out"])), live))
    return tuple(key)


class _OutputMap:
    """y = v A^T for an output map A (m x s, SciPy sparse) on the device, in one dtype.

    Dense (one matrix product) when A has at most ``DENSE_OUT`` entries of which more than 1 / ``GATHER_FILL`` are
    stored. Otherwise by gathers of the stored entries, without a sparse library (torch's sparse products synchronise
    the host with the device on some GPUs): the rows are bucketed by their number of stored entries, rounded up to a
    power of two (one bucket of the largest number when that at most doubles the work), each bucket's entries padded
    with zeros to its width w, and y[:, rows] = sum_{t < w} v[:, idx_t] val_t, in a fixed order (deterministic)."""

    def __init__(self, a, device, dtype):
        a = sp.csr_matrix(a)
        m, s = a.shape
        self.m, self.dense, self.buckets = m, None, []
        if m * s <= DENSE_OUT and GATHER_FILL * a.nnz > m * s:
            self.dense = torch.as_tensor(a.toarray()).to(device, dtype)
            return
        lens = np.diff(a.indptr)
        top = int(lens.max()) if m else 0
        if top * m <= 2 * a.nnz:
            widths = np.where(lens > 0, top, 0)
        else:
            widths = np.where(lens > 0, 1 << np.ceil(np.log2(np.maximum(lens, 1))).astype(np.int64), 0)
        for w in np.unique(widths[widths > 0]):
            rows = np.flatnonzero(widths == w)
            t = np.arange(w)[:, None]
            have = t < lens[rows][None, :]
            at = (a.indptr[rows][None, :] + t)[have]
            idx, val = np.zeros((w, rows.size), dtype=np.int64), np.zeros((w, rows.size))
            idx[have], val[have] = a.indices[at], a.data[at]
            self.buckets.append((torch.from_numpy(rows).to(device), torch.from_numpy(idx).to(device),
                                 torch.from_numpy(val).to(device, dtype)))
        self.whole = len(self.buckets) == 1 and self.buckets[0][0].numel() == m  # every row in one bucket, in order
        self.zero_rows = bool((lens == 0).any())

    def __call__(self, v):
        if self.dense is not None:
            return v @ self.dense.T
        if self.whole:
            return self._gather(v, *self.buckets[0][1:])
        y = (torch.zeros if self.zero_rows else torch.empty)((v.shape[0], self.m), dtype=v.dtype, device=v.device)
        for rows, idx, val in self.buckets:
            y.index_copy_(1, rows, self._gather(v, idx, val))
        return y

    @staticmethod
    def _gather(v, idx, val):
        y = v.index_select(1, idx[0]) * val[0]
        for t in range(1, idx.shape[0]):
            y.addcmul_(v.index_select(1, idx[t]), val[t])
        return y


def _clamp_bounds(support, out_dtype, t):
    """(low, high): bounds just inside a finite support in the output precision (as PointwiseField.clamp_open), None
    where the support is unbounded."""
    if support is None:
        return None, None
    dt = torch.empty(0, dtype=out_dtype).numpy().dtype
    bounds = []
    for bound, toward in ((support[0], np.inf), (support[1], -np.inf)):
        b = None if bound is None else np.asarray(bound, dtype=float)
        if b is None or not np.all(np.isfinite(b)):
            bounds.append(None)
            continue
        bounds.append(t(np.nextafter(b.astype(dt), np.asarray(toward, dtype=dt)).astype(np.float64)))
    return tuple(bounds)


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


# ---------------------------------------------------------------------- maps held as objects: storage


def _stored_map(t):
    """(kind, description) of a numerical inversion or a pointwise function, as a file without code can hold it:
    an inversion by its scipy.stats distribution (name and constant parameters), a NumPy or torch function by name;
    ("object", None) otherwise."""
    from .distributions.inversion import Inversion

    if isinstance(t, Inversion):
        dist = t.dist
        gen = getattr(dist, "dist", None)
        name = getattr(gen, "name", None)
        import scipy.stats as st

        if name is None or type(getattr(st, name, None)) is not type(gen):
            return "object", None
        args = [float(a) for a in dist.args]
        kwds = {k: float(v) for k, v in dist.kwds.items()}
        return "inversion", {"scipy": name, "args": args, "kwds": kwds, "loc": float(t.loc), "scale": float(t.scale),
                             "method": t.requested, "tol": float(t.tol), "zmax": float(t.zmax_request)}
    func = getattr(t, "func", None)
    if func is not None and hasattr(t, "torch_func"):  # transforms.PointMap
        module = "numpy" if isinstance(func, np.ufunc) else getattr(func, "__module__", None) or ""
        name = getattr(func, "__name__", None)
        if name and (module == "numpy" or module.startswith("torch")):
            try:
                found = getattr(importlib.import_module(module), name, None)
            except ImportError:
                found = None
            if found is func:
                return "pointmap", {"module": module, "name": name, "torch": bool(t.torch_func)}
    return "object", None


def _rebuild_map(kind, p):
    if kind == "inversion":
        import scipy.stats as st

        from .distributions.inversion import Inversion

        t = Inversion(getattr(st, p["scipy"])(*p["args"], **p["kwds"]), p["loc"], p["scale"], method=p["method"],
                      tol=p["tol"], zmax=p["zmax"])
        t._resolve(10**9, True)  # the table, as the sampler that was saved used
        return t
    if kind == "pointmap":
        from .distributions.transforms import PointMap

        if not (p["module"] == "numpy" or p["module"].startswith("torch")):
            raise ValueError(f"a stored pointwise function must be a NumPy or torch function, not {p['module']}")
        return PointMap(getattr(importlib.import_module(p["module"]), p["name"]), p["torch"])
    raise ValueError(f"unknown stored map {kind!r}")


# ====================================================================== set-up


class DenseOf:
    """Mixin of the Gaussians: ``dense()``, the dense sampler of their coefficients, and ``E @ gaussian``."""

    __array_ufunc__ = None  # E @ gaussian: NumPy (and SciPy, torch) leave the product to __rmatmul__

    def dense(self, dofs=None, observe=None, **kwargs) -> DenseSampler:
        """A :class:`ngsrng.DenseSampler` of this Gaussian's coefficients: at ``dofs`` (default all) or under
        ``observe`` (an m x ndof matrix), as dense products on any device (:mod:`ngsrng.dense`; arguments as
        :func:`ngsrng.dense.dense`). Set up once per arguments and kept with the Gaussian; ``save`` stores it."""
        return dense(self, dofs, observe, **kwargs)

    def __rmatmul__(self, operator):
        """``E @ gaussian``: the observations E f (an :class:`ngsrng.ObservedGaussian`)."""
        from .observed import ObservedGaussian, _as_operator

        return ObservedGaussian(self, operator=_as_operator(operator, self.space.ndof), seed=inherit(None, self),
                                method="pathwise")


class ObservedField(InPlace):
    """``E @ d``: a linear map E of the values of a distribution's draws (``d`` from ``rng.distribution(...)``, a
    pointwise family, or ``ngsrng.correlated(...)``). Gaussian fields observed are an :class:`ngsrng.ObservedGaussian`
    instead. ``sample(size)`` draws ``d`` and applies E (one result per component for correlated fields; ``out=``
    fills buffers in place, :mod:`ngsrng.inplace`); ``dense()`` is ``d.dense(observe=E)``; ``E2 @ this`` is
    ``(E2 E) @ d``."""

    __array_ufunc__ = None

    def __init__(self, source, operator):
        from .observed import _as_operator

        self.source = source
        self.nvalues = _nvalues(source)
        self.operator = _as_operator(operator, self.nvalues)
        self.m = self.operator.shape[0]
        self.components = len(source._parts) if hasattr(source, "_parts") else 1
        self._torch = DeviceCache()

    @property
    def info(self) -> dict:
        return {"law": f"E @ draws (m = {self.m})", "of": self.source.info}

    def __repr__(self):
        return f"ObservedField(m={self.m}, of {self.source!r})"

    def __rmatmul__(self, operator):
        from .observed import _as_operator

        return ObservedField(self.source, _as_operator(operator, self.m) @ self.operator)

    def sample(self, size=None, output=None, out=None):
        """(*size, m) observations of draws of the distribution (``size=None``: one, (m,)); a tuple of them for
        correlated fields. ``out``: a buffer of that shape (one per component for correlated fields) filled in
        place and returned."""
        if out is not None:
            bufs = list(out) if isinstance(out, (list, tuple)) else [out]
            self.sample_into([as_tensor(b).reshape(-1, self.m) for b in bufs])
            return out
        x = self.source.sample(1 if size is None else size, output="torch")
        res = tuple(self._apply(xi, size) for xi in x) if isinstance(x, tuple) else self._apply(x, size)
        if (output or ("numpy" if _first(res).device.type == "cpu" else "torch")) == "numpy":
            return tuple(r.cpu().numpy() for r in res) if isinstance(res, tuple) else res.cpu().numpy()
        return res

    rvs = sample

    def sample_into(self, rows):
        """Fill (n, m) rows in place (one buffer per component for correlated fields), drawing the distribution in
        chunks of ``work_values`` values (one draw: the same values as ``sample``)."""
        bufs = list(rows) if isinstance(rows, (list, tuple)) else [rows]
        bufs = [as_tensor(b) for b in bufs]
        if len(bufs) != self.components or any(b.ndim != 2 or b.shape[1] != self.m for b in bufs):
            raise ValueError(f"out: {self.components} buffer(s) of shape (n, {self.m})")
        n = bufs[0].shape[0]
        step = max(1, int(self.work_values // (self.nvalues + self.m)))
        for k0 in range(0, n, step):
            k1 = min(n, k0 + step)
            x = self.source.sample(k1 - k0, output="torch")
            for b, xi in zip(bufs, x if isinstance(x, tuple) else (x,)):
                b[k0:k1].copy_(self._apply(xi, k1 - k0))
        return rows

    def _apply(self, x, size):
        if not x.is_floating_point():  # discrete families: the observations are real
            x = x.to(torch.float64)
        key = (str(x.device), x.dtype)
        if key not in self._torch:
            self._torch[key] = sparse_to_torch(self.operator, x.device, x.dtype)
        flat = x.reshape(-1, self.nvalues)
        y = (self._torch[key] @ flat.T).T
        return y[0] if size is None else y.reshape(*x.shape[:-1], self.m)

    def dense(self, dofs=None, observe=None, **kwargs) -> DenseSampler:
        """``d.dense(observe=E)``: set up for these observations (``dofs``: some of them; ``observe``: a further
        map)."""
        from .observed import _as_operator

        if dofs is not None and observe is not None:
            raise ValueError("give dofs= or observe=, not both")
        e = self.operator
        if dofs is not None:
            e = e[np.asarray(dofs, dtype=np.int64).ravel()]
        elif observe is not None:
            e = _as_operator(observe, self.m) @ e
        return dense(self.source, None, e, **kwargs)


def _first(res):
    return res[0] if isinstance(res, tuple) else res


def _nvalues(source) -> int:
    from .correlated import CorrelatedFields

    if isinstance(source, CorrelatedFields):
        return source._parts[0].nvalues
    return _Part.of_field(source.field, source.generator, source.name, check=False).nvalues


def observe_distribution(d, operator):
    """``E @ d`` for a distribution: an :class:`ngsrng.ObservedGaussian` for a Gaussian FE field (E composed with the
    field's ``bnd=`` restriction and ``points=`` evaluation), an :class:`ObservedField` otherwise. Either is the law
    of the observations without a set-up: its draws are pathwise (a draw of d, then E) until ``.dense()``."""
    from .generator import _NormalField
    from .observed import ObservedGaussian

    field = d.field
    bnd = getattr(field, "bnd", None)
    if isinstance(field, _NormalField) and (bnd is None or bnd["values"] is None):
        part = _Part.of_field(field, d.generator, d.name)
        rows, _ = part.output_rows(None, operator)
        g = part.latent_rows(rows)
        return ObservedGaussian(field.gauss, operator=g, loc=part.loc, scale=part.scale, seed=d.generator._seed(),
                                dtype=d.generator.dtype, method="pathwise")
    return ObservedField(d, operator)


def dense(source, dofs=None, observe=None, *, rank=None, tol=None, rtol=RTOL, pin=None, zero=ZERO,
          max_points=MAX_POINTS, device=None, dtype=None, noise=None, seed=None, output=None) -> DenseSampler:
    """The dense sampler of ``source`` (a distribution, ``ngsrng.correlated(...)``, an observed law or a Gaussian);
    see :mod:`ngsrng.dense`. The set-up is made once per source and law arguments and kept (on the source; for
    distributions also on their generator, so that ``spawn`` children share it); the sampler once per further
    arguments.

    Parameters
    ----------
    dofs : indices of the outputs among the values a draw returns (FE DoFs for coefficient representations). Default:
        the trace DoFs with ``bnd=``, else all.
    observe : a linear map of those values (m x nvalues; SciPy sparse, NumPy or torch): the outputs are
        ``observe @ values`` (after ``bnd=``), as for ``(observe @ source).dense()``. Not with ``dofs``.
    rtol : the cut: eigenvalues of the latent covariance below rtol times the largest are dropped (default 1e-12);
        None or 0: no cut.
    rank, tol : keep only the leading ``rank`` modes, or the fewest that keep a fraction 1 - ``tol`` of the latent
        variance (Karhunen-Loeve; ``info["exact"]`` is then False).
    pin : latent values whose variance is at most pin times the largest are set to their mean (default None: only
        those with exactly zero variance).
    zero : entries of the output map at most zero times their row's largest are rounding of exact zeros, dropped
        (default 1e-15; None or 0: none).
    max_points : the largest number of latent values (a clear error beyond).
    device, dtype, noise, seed, output : where and how it samples (default: as the source's own draws; ``noise``
        "host" or "device", see the module docstring; ``seed`` default: a child of the source's randomness).
    """
    from .correlated import CorrelatedFields
    from .generator import FieldDistribution
    from .observed import ObservedGaussian

    if isinstance(source, ObservedField):
        return source.dense(dofs, observe, rank=rank, tol=tol, rtol=rtol, pin=pin, zero=zero, max_points=max_points,
                            device=device, dtype=dtype, noise=noise, seed=seed, output=output)
    if dofs is not None and observe is not None:
        raise ValueError("give dofs= or observe=, not both")
    shared = None  # (generator, key) under which a generator keeps the set-up for its spawned children
    if isinstance(source, FieldDistribution):
        owner, mix, parent = source.field, np.eye(1), source.generator
        parts = [_Part.of_field(source.field, source.generator, source.name)]
        keep64 = source.generator._output_kind(source.field, source.name)[1]
        out_dtype, out_device = (torch.float64 if keep64 else source.generator.dtype), source.generator.device
        shared = _shared_key(source.generator, [source.field])
    elif isinstance(source, CorrelatedFields):
        owner, mix, parent = source, source._mix, source._parts[0].gen
        parts = [_Part.of_field(p.field, p.gen, p.name) for p in source._parts]
        out_dtype = torch.float64 if any(p.keep64 for p in source._parts) else parent.dtype
        out_device = parent.device
        if all(p.gen is parent for p in source._parts):
            shared = _shared_key(parent, [p.field for p in source._parts], source._mix)
    elif isinstance(source, ObservedGaussian):
        owner, mix, parent, parts = source, np.eye(1), source, [_Part.of_observed(source)]
        out_dtype, out_device = source.dtype, source.device
    elif hasattr(source, "apply_t") and hasattr(source, "space") and hasattr(source, "mean"):
        owner, parts, mix, parent = source, [_Part.of_gaussian(source)], np.eye(1), source
        out_dtype = getattr(source, "dtype", torch.float64)
        out_device = getattr(source, "torch_device", None) or getattr(source, "device", "cpu")
    else:
        raise TypeError(f"dense samplers are made from distributions, correlated fields, observed laws or "
                        f"Gaussians, not {type(source).__name__}")
    rtol = None if not rtol else float(rtol)
    zero = None if not zero else float(zero)
    law_key = (_digest(dofs), _digest(observe), rank, tol, rtol, pin, zero, int(max_points))
    cache = owned_cache(owner, "_ngsrng_dense")
    laws, samplers = cache.setdefault("laws", {}), cache.setdefault("samplers", {})
    if law_key not in laws:
        law = None
        if shared is not None:
            law = shared[0]._dense_laws.get(shared[1] + (law_key,), (None, None))[1]
        if law is None and isinstance(source, ObservedGaussian) and _reusable(source, dofs, observe, rank, tol, rtol,
                                                                              pin):
            law = _law_from_observed(source)
        if law is None:
            law = _build_law(parts, mix, dofs, observe, rank, tol, rtol, pin, zero, int(max_points))
        if shared is not None:
            shared[0]._dense_laws[shared[1] + (law_key,)] = (shared[2] + (law_key,), law)
        laws[law_key] = law
    law = laws[law_key]
    device = resolve_device(out_device if device is None else device)
    dtype = out_dtype if dtype is None else resolve_dtype(dtype)
    key = law_key + (str(device), str(dtype), noise, _digest(seed), output)
    if key not in samplers:
        src = seed if seed is not None else _child_source(parent)
        samplers[key] = DenseSampler([law], device=device, dtype=dtype, noise=noise, seed=src, output=output)
    return samplers[key]


def _reusable(obs, dofs, observe, rank, tol, rtol, pin) -> bool:
    """Whether an ObservedGaussian's own factor is the dense sampler's: it has one, with the same cut, no selection or
    truncation, its noise folded into it (or none), and the observations as the cheaper latent (noise, or no more
    observations than the DoFs they read)."""
    if not (obs.method == "factor" and dofs is None and observe is None and rank is None and tol is None and
            pin is None and rtol == obs.rtol and (obs._folded or obs._noise_sampler is None)):
        return False
    return obs._noise_factor() is not None or obs.m <= np.unique(sp.csr_matrix(obs.G).indices).size


def _law_from_observed(obs) -> dict:
    """The dense law of an ObservedGaussian from its factor (no new set-up)."""
    f = obs._F64.cpu().numpy()
    comp = {"name": "observed", "map": "affine", "label": "affine", "params": {"shift": np.zeros(obs.m), "scale": 1.0},
            "mean": obs.mean, "out": None, "nout": obs.m, "support": None, "pull": None}
    part = _Part.of_observed(obs)
    info = {"law": "Gaussian observations", "components": [{"name": "observed", "map": "affine",
                                                            "representation": "observations", "bnd": None}],
            "outputs": obs.m, "selection": "all", "latent": "outputs", "rank": int(f.shape[1]), "exact": True,
            "covariance": obs.plan.get("covariance"), "captured_variance": 1.0 - obs.plan.get("dropped_trace", 0.0),
            "dropped_trace": obs.plan.get("dropped_trace", 0.0),
            "truncation": {"rtol": obs.rtol, "rank": None, "tol": None, "pin": None}, "reused": "the sampler's factor",
            "provenance": _provenance(part)}
    law = {"factor": f, "mix": np.eye(1), "components": [comp], "work64": False, "dofs": np.arange(obs.m),
           "latent_index": None, "latent_coordinates": None, "info": info}
    return _compile(law)


def _shared_key(gen, fields, mix=None):
    """(generator, key, how to rebuild the key after unpickling) of a set-up the generator keeps for its spawned
    children: the fields' keys in the generator (None if a field is not one of its own)."""
    infos = []
    for field in fields:
        found = gen._key_of(field)
        if found is None:
            return None
        infos.append(found)
    keys = tuple(k for k, _ in infos)
    extra = _digest(None if mix is None else np.asarray(mix))
    return gen, (keys, extra), ([i for _, i in infos], extra)


def _child_source(parent) -> Source:
    src = getattr(parent, "source", None)
    return src.spawn() if isinstance(src, Source) else Source()


def _digest(x):
    """A hashable stand-in for an argument (arrays and matrices by content)."""
    if x is None or isinstance(x, (numbers.Number, str)):
        return x
    if isinstance(x, (np.random.SeedSequence, np.random.Generator, torch.Generator, Source)):
        return ("id", id(x))
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        x = x.to_dense().numpy() if x.layout != torch.strided else x.numpy()
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

    noise = None  # observation noise (an m x q factor) of an ObservedGaussian

    @classmethod
    def of_field(cls, field, gen, name, check=True):
        from .distributions.transforms import MarginalField
        from .generator import _NormalField

        p = cls()
        p.name, p.field = name, field
        bnd = getattr(field, "bnd", None)
        if check and bnd is not None and bnd["values"] is not None:
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
                f"{name}: dense samplers and E @ d cover pointwise maps of one latent Gaussian (uniform, lognormal, "
                "gamma, beta, ..., normal with a representation=, marginal, indicator, transformed) and Gaussian "
                "fields; this family is built from several latents or is not a function of point values")
        return p

    @classmethod
    def of_gaussian(cls, g):
        p = cls()
        p.name, p.field, p.bnd, p.kind, p.gauss, p.points = type(g).__name__, None, None, "coefficients", g, None
        p.representation = "coefficients"
        p.loc, p.scale, p.evaluation, p.nvalues = np.asarray(g.mean, dtype=float), 1.0, None, g.space.ndof
        return p

    @classmethod
    def of_observed(cls, obs):
        """An ObservedGaussian: its observations y = G (loc + scale (f - mean)) + e are the values. Without noise, of a
        field with a factor, it is the field observed by G: a Gaussian field part (the latent at the DoFs G reads, or
        at the observations when there are fewer of them, as for ``d.dense(observe=G)``)."""
        if obs._noise_factor() is None and hasattr(obs.field, "apply_t"):
            p = cls()
            p.name, p.field, p.bnd, p.kind, p.gauss, p.points = "observed", None, None, "coefficients", obs.field, None
            p.representation = "observations"
            p.loc, p.scale, p.evaluation, p.nvalues = obs._loc, obs.scale, sp.csr_matrix(obs.G), obs.m
            return p
        p = cls()
        p.name, p.field, p.bnd, p.kind, p.gauss, p.points = "observed", obs, None, "observed", obs.field, None
        p.representation = "observations"
        p.nvalues, p.obs = obs.m, obs
        return p

    @property
    def space(self):
        return self.obs.space if self.kind == "observed" else self.gauss.space

    def output_rows(self, dofs, observe):
        """The output map on the values a draw returns (m x nvalues, CSR), with ``bnd=`` (DoFs off the trace are
        zero in a draw) folded in, and the DoF indices of the outputs (None for an observation)."""
        from .observed import _as_operator

        n = self.nvalues
        if observe is not None:
            o, out_dofs = _as_operator(observe, n), None
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
        """``rows`` (m x nvalues) composed with the linear map from the latent vector to the values: the
        coefficients of a Gaussian field (with ``points=``, their evaluation), or the representation's map from the
        latent point values (identity for values at points; the L2 projection, from one mass solve per row). For the
        Bernstein representation the exact pattern only (ones where an entry may be nonzero; the values come from
        :meth:`forward_columns`). CSR, m x (latent vector length)."""
        if self.kind == "coefficients":
            g = (rows @ self.evaluation).tocsr() if self.evaluation is not None else rows
            g.eliminate_zeros()
            return g
        if self.kind == "observed" or self.representation in ("quadrature", "points"):
            return rows
        if self.representation == "bernstein":
            touched = np.unique(rows.indices)
            dep = self.points.bernstein.dependence(touched)
            pattern = (abs(rows[:, touched]) @ dep).tocsr()
            pattern.data[:] = 1.0
            return pattern
        return _projection_rows(self, rows)

    def forward_columns(self, rows, cols):
        """For the Bernstein representation: ``rows`` applied to the library's own conversion of the unit vectors at
        ``cols`` (dense m x |cols|), and the sum of their conversion residuals (the rounding bound of any sample)."""
        bern = self.points.bernstein
        out = np.empty((rows.shape[0], len(cols)))
        resid = 0.0
        step = 256
        for j0 in range(0, len(cols), step):
            blk = cols[j0:j0 + step]
            e = np.zeros((bern.npoints, len(blk)))
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
        """The pointwise map of the latent at ``idx`` as the sampler keeps it: name, parameters, support, pull."""
        if self.kind == "coefficients":  # loc + scale (g - mean_g)
            return {"map": "affine", "params": {"shift": self.loc[idx] - self.scale * mean, "scale": float(self.scale)},
                    "support": None, "pull": None, "label": "affine"}
        t = self.field.transform
        npts = self.points.npoints
        support = getattr(t, "support", None)
        pull_range = support or getattr(t, "closed_range", None)
        spec = {"label": self.field.label, "support": _restrict_pair(support, idx, npts),
                "pull": None if self.representation != "bernstein" or pull_range is None else
                {"range": _restrict_pair(pull_range, idx, npts), "margin": _margin()}}
        from .distributions.transforms import ClosedForm

        if isinstance(t, ClosedForm):
            names = {func: name for name, func in _maps().items()}
            if t.func in names:
                spec.update(map=names[t.func], params={k: _restrict(v, idx, npts) for k, v in t.params.items()})
                return spec
        from .distributions.inversion import Inversion

        if isinstance(t, Inversion):
            if t.varying or np.ndim(t.loc) or np.ndim(t.scale):
                raise NotImplementedError(f"{self.name}: a numerical inversion with parameters that vary in space has "
                                          "no dense sampler (constant parameters are supported)")
            t._resolve(10**9, True)  # the table (as d.sample() on a GPU or for large draws), or ppf
        kind, stored = _stored_map(t)
        spec.update(map=kind, params={}, transform=t, stored=stored)
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


def _projection_rows(part, rows):
    """rows C for the L2 projection C = M_free^-1 B_free (zero at fixed dofs), from the transposes: one mass solve
    per row. Its rows are dense (every point contributes to every coefficient)."""
    from .solvers import NGSolveDirect
    from .space import mass_form

    field, space = part.field, part.space
    if field._mass_solver is None:
        field._mass_solver = NGSolveDirect(space, mass_form)
    free = space.free
    bf = field.points.rule.b.tocsr()[free]
    rt_all = rows.T.tocsc()
    pieces = []
    for i0 in range(0, rows.shape[0], 256):
        rt = rt_all[:, i0:i0 + 256].toarray()
        x = field._mass_solver.apply(torch.from_numpy(np.ascontiguousarray(rt[free]))).numpy()
        pieces.append(sp.csr_matrix(np.asarray((bf.T @ x).T)))
    return sp.vstack(pieces).tocsr()


def _build_law(parts, mix, dofs, observe, rank, tol, rtol, pin, zero, max_points) -> dict:
    from .observed import covariance_factor

    t0 = time.perf_counter()
    k = len(parts)
    p0 = parts[0]
    outs, lat, dropped = [], [], 0.0
    for p in parts:
        o, out_dofs = p.output_rows(dofs, observe)
        o, x = _drop_rounding(o, zero)
        outs.append((o, out_dofs))
        a = p.latent_rows(o)
        if not (p.kind == "points" and p.representation == "bernstein"):  # Bernstein: an exact pattern
            a, y = _drop_rounding(a, zero)
            x = max(x, y)
        lat.append(a)
        dropped = max(dropped, x)
    support = np.unique(np.concatenate([sp.csr_matrix(a).indices for a in lat] + [np.zeros(0, dtype=np.int64)]))
    m0 = outs[0][0].shape[0]
    at_outputs = p0.kind == "observed" or (k == 1 and p0.kind == "coefficients" and m0 < support.size)
    nlat = m0 if at_outputs else support.size
    if nlat > max_points:
        raise ValueError(
            f"the outputs depend on {nlat} latent values, more than max_points={max_points} (a dense factor of "
            f"{8 * nlat**2 / 2**30:.1f} GiB): choose fewer DoFs (dofs=, bnd=) or a representation whose outputs "
            "depend on fewer values (representation='bernstein' rather than 'projection'), or raise max_points")
    comps, scale, noise = [], 1.0, None
    for p, (o, _), a in zip(parts, outs, lat):
        if p.kind == "observed":  # the observations (with their noise) are the latent; loc and scale inside
            obs = p.obs
            g_full = (o @ sp.csr_matrix(obs.G)).tocsr()
            g_op = g_full[:, obs.space.free] if len(obs.space.free) < obs.space.ndof else g_full
            mean = np.asarray(o @ obs._mean).ravel()
            nf = obs._noise_factor()
            scale, noise = obs.scale, None if nf is None else np.asarray(o @ nf)
            spec = {"map": "affine", "label": "affine", "params": {"shift": np.zeros(m0), "scale": 1.0},
                    "support": None, "pull": None}
            out = None
        elif at_outputs:  # a Gaussian field observed by fewer observations than the DoFs they read
            a_full = sp.csr_matrix(a)
            g_op = a_full[:, p.space.free].tocsr()
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
                a_s = sp.csr_matrix(a_s.multiply(fwd))  # the exact pattern, the conversion's values
                if spec["pull"] is not None:
                    spec["pull"] = {**spec["pull"], "margin": max(spec["pull"]["margin"], 1e3 * resid)}
            out = None if _is_identity(a_s) else a_s
        spec.update(name=p.name, mean=mean, out=out, nout=o.shape[0], g=g_op)
        comps.append(spec)
    # the latent covariance (shared by the components) and its factor
    f, finfo = covariance_factor(p0.gauss, comps[0]["g"], p0.space, scale=scale, noise=noise, rtol=rtol, rank=rank,
                                 tol=tol, pin=pin)
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
        **finfo,
        "zero": zero,
        "zero_drop": dropped,
        "provenance": _provenance(p0),
    }
    law = {"factor": f.cpu().numpy(), "mix": np.asarray(mix, dtype=float), "components": comps, "work64": work64,
           "dofs": outs[0][1], "latent_index": None if at_outputs else support,
           "latent_coordinates": None if at_outputs else _coordinates(p0, support), "info": info}
    law = _compile(law)
    law["info"]["setup_s"] = round(time.perf_counter() - t0, 3)
    return law


# ---------------------------------------------------------------------- composition and compilation


def _compose(law, e) -> dict:
    """The law of E @ (the outputs of ``law``), compiled (:func:`_compile`); entries of the composed map at most
    ``zero`` (the law's) times their row's largest are dropped as rounding."""
    comps, zero, dropped = [], law["info"].get("zero", ZERO), law["info"].get("zero_drop", 0.0)
    for c in law["components"]:
        a, x = _drop_rounding(e if c["out"] is None else e @ sp.csr_matrix(c["out"]), zero)
        dropped = max(dropped, x)
        comps.append({**c, "out": a, "nout": e.shape[0]})
    info = {**law["info"], "selection": "observe (composed)", "outputs": int(e.shape[0]), "zero_drop": dropped}
    return _compile({**law, "components": comps, "dofs": None, "info": info})


def _drop_rounding(a, zero):
    """(CSR copy of ``a`` without the entries at most ``zero`` times their row's largest, the largest dropped
    relative entry). ``zero`` None: only exact zeros are dropped."""
    a = sp.csr_matrix(a, dtype=np.float64, copy=True)
    a.eliminate_zeros()
    if not zero or a.nnz == 0:
        return a, 0.0
    size = np.abs(a.data)
    rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    top = np.zeros(a.shape[0])
    np.maximum.at(top, rows, size)
    rel = size / top[rows]
    small = rel <= zero
    if not small.any():
        return a, 0.0
    worst = float(rel[small].max())
    a.data[small] = 0.0
    a.eliminate_zeros()
    return a, worst


def _compile(law) -> dict:
    """Make a law's representation small, without changing it (exact, no tolerance):

    1. latent values no output reads are dropped (a Gaussian's marginal: rows of its factor);
    2. a factor with more columns than rows is recompressed to as many columns (F V from the SVD F = U s V^T: the
       same covariance, z -> V^T z is again standard normal, the same V for every component keeps their coupling);
    3. Gaussian (affine) components with one output map A fold it into the factor, y = A m' + (A F) z (A F recompressed
       the same way to at most m columns), when that costs less per sample (m r' < |S| r + nnz(A))."""
    comps = law["components"]
    f = law["factor"]
    nlat = f.shape[0]
    used = np.arange(nlat)
    if all(c["out"] is not None for c in comps):
        used = np.unique(np.concatenate([sp.csr_matrix(c["out"]).indices for c in comps] + [np.zeros(0, np.int64)]))
    if used.size < nlat:
        law = _restrict_law(law, used)
        comps, f = law["components"], law["factor"]
    if f.shape[1] > f.shape[0]:
        f = _recompress(f)
    law = {**law, "factor": f}
    outs = [c["out"] for c in comps]
    if all(c["map"] == "affine" for c in comps) and outs[0] is not None and \
            all(o is outs[0] or (o.shape == outs[0].shape and (o != outs[0]).nnz == 0) for o in outs):
        a = sp.csr_matrix(outs[0])
        g = np.asarray(a @ f)
        if g.shape[1] > g.shape[0]:
            g = _recompress(g)
        if a.shape[0] * g.shape[1] < f.shape[0] * f.shape[1] + a.nnz:
            new = []
            for c in comps:
                shift = c["params"]["shift"]
                new.append({**c, "out": None, "mean": np.asarray(a @ c["mean"]).ravel(),
                            "params": {**c["params"], "shift": np.asarray(a @ shift).ravel() if np.ndim(shift) else
                                       shift * np.asarray(a.sum(axis=1)).ravel()}})
            law = {**law, "factor": g, "components": new, "latent_index": None, "latent_coordinates": None,
                   "info": {**law["info"], "latent": "outputs", "folded": True}}
    info = law["info"]
    info.update(latent_points=int(law["factor"].shape[0]), rank=int(law["factor"].shape[1]),
                memory_bytes=int(law["factor"].nbytes + sum(_nbytes(c) for c in law["components"])))
    return law


def _recompress(f):
    """F V_r (n x n) for F (n x r), r > n: the same F F^T with n columns, rows that are zero stay exactly zero."""
    _, s, vt = np.linalg.svd(f, full_matrices=False)
    return f @ vt.T


def _restrict_law(law, used) -> dict:
    """The law at the latent values ``used`` only."""
    n = law["factor"].shape[0]

    def cut(x):
        return x[used] if isinstance(x, np.ndarray) and x.ndim and x.shape[0] == n else x

    comps = []
    for c in law["components"]:
        new = {**c, "mean": c["mean"][used], "params": {k: cut(v) for k, v in c["params"].items()},
               "out": None if c["out"] is None else sp.csr_matrix(c["out"])[:, used].tocsr()}
        if c.get("support") is not None:
            new["support"] = tuple(cut(b) for b in c["support"])
        if c.get("pull") is not None:
            new["pull"] = {**c["pull"], "range": tuple(cut(b) for b in c["pull"]["range"])}
        comps.append(new)
    return {**law, "factor": law["factor"][used], "components": comps,
            "latent_index": None if law.get("latent_index") is None else law["latent_index"][used],
            "latent_coordinates": None if law.get("latent_coordinates") is None else law["latent_coordinates"][used]}


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


def _law_label(parts, k):
    p = parts[0]
    what = {"coefficients": "Gaussian field", "observed": "Gaussian observations"}.get(p.kind) or \
        f"{p.field.label} ({p.representation})"
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

