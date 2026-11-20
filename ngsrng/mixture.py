"""Finite mixtures of laws: each draw follows one of several laws, chosen at random with given weights.

    laws = [rng.distribution("uniform", 0, 1, representation="bernstein",
                             correlation=ngsrng.matern_correlation(length, nu=5, boundary="torus"))
            for length in np.geomspace(0.1, 0.4, 16)]
    mix = ngsrng.mixture(laws)                     # equal weights; weights=[...] otherwise
    x, labels = mix.sample(1000, labels=True)      # labels[i]: the index of the law draw i follows
    A = mix.dense(tol=1e-6, device="cuda")         # one dense factor per law: A.sample(n), A.save(...)

The law
-------
A draw is a label l ~ Categorical(weights), then a draw of law l: the mixture's law is sum_l w_l P_l. Each law keeps
everything it has on its own: family and parameters, representation, conditioning, ``bnd=``, the Bernstein range
guarantee, the exact pointwise standardisation of its latent. A hyperparameter with finitely many values (a
correlation length, a smoothness, a variance) gives such a mixture, one law per value with the value's probability as
its weight; a continuous law of a hyperparameter is approximated by a quadrature of it (log-spaced lengths with equal
weights for a log-uniform length).

Laws
----
Laws with the same kind of draws and the same outputs: distributions (``rng.distribution(...)``), correlated fields
(``ngsrng.correlated(...)``), observed laws (``E @ d``), dense samplers and mixtures, whose ``sample(n)`` is (n, ...);
or Gaussians (``Gaussian``, ``StationaryMatern``, posteriors, low-rank and observed Gaussians), whose ``sample(n)`` is
(..., n), and so is the mixture's then. ``sample`` draws each law's group with the law's own sampler; ``dense()`` sets
up every law's dense sampler (:mod:`ngsrng.dense`; ``rank=`` and ``tol=`` per law, relative to its own variance) and
draws them together, on any device: a batch's labels, then one dense product per law for the draws that follow it.
``E @ mixture`` is the mixture of the observed laws.

Randomness
----------
The labels come from the mixture's own randomness (``seed``, :mod:`ngsrng.seeding`; default: a child of the first
law's), on the host and sample by sample, so a batch's labels do not depend on how it is split. Each law draws its
group with its own randomness, as it does alone.
"""

import math
import numbers

import numpy as np
import torch

from .backend import owned_cache
from .dense import MAX_POINTS, RTOL, ZERO
from .inplace import InPlace, as_tensor, default_work_values
from .seeding import Source, as_source

__all__ = ["mixture", "Mixture"]


def mixture(laws, weights=None, seed=None) -> "Mixture":
    """The finite mixture of ``laws`` with probabilities ``weights`` (default: equal; normalised): each draw follows
    law l with probability weights[l] (see :mod:`ngsrng.mixture`). ``seed``: the randomness of the labels."""
    return Mixture(laws, weights, seed)


class Mixture(InPlace):
    """A finite mixture of laws (see :func:`mixture`): ``sample(size, labels=False)``, ``dense()``, ``E @ mixture``,
    ``info``; ``laws``, ``weights``."""

    __array_ufunc__ = None  # E @ mixture: NumPy (and SciPy, torch) leave the product to __rmatmul__

    def __init__(self, laws, weights=None, seed=None):
        from .dense import _mixture_weights

        self.laws = tuple(laws)
        self.weights = _mixture_weights(weights, len(self.laws))
        kinds = {_kind(law) for law in self.laws}
        if len(kinds) > 1:
            raise ValueError("the laws of a mixture draw the same kind of samples: distributions, correlated fields, "
                             "observed laws and dense samplers ((n, ...) draws), or Gaussians ((..., n) draws)")
        self.kind = kinds.pop()
        self.source = as_source(seed) if seed is not None else _randomness(self.laws[0]).spawn()

    def __len__(self):
        return len(self.laws)

    @property
    def info(self) -> dict:
        return {"law": f"mixture of {len(self.laws)} laws", "weights": self.weights.tolist(),
                "laws": [law.info for law in self.laws], "labels": self.source.describe()}

    def __repr__(self):
        return f"Mixture({len(self.laws)} laws, weights={np.round(self.weights, 4).tolist()})"

    # ------------------------------------------------------------------ draws

    def sample(self, size=None, output=None, out=None, labels=False):
        """Draws as the laws' own ``sample(size)`` returns them ((*size, ...) for distributions; (..., n) for
        Gaussians; one per component for correlated fields), each draw from the law of its label. ``out``: buffers of
        that shape filled in place and returned. ``labels``: also return the labels (int64, of shape ``size``), as
        ``(samples, labels)``."""
        if out is not None:
            res, lab = out, self._sample_out(size, out)
        elif size is None:
            lab = self._labels(1)
            res = _draw(self.laws[int(lab[0])], None, output)
            lab = lab[0]
        else:
            shape = (int(size),) if isinstance(size, numbers.Integral) else tuple(int(s) for s in size)
            if self.kind == "columns" and len(shape) != 1:
                raise ValueError("Gaussians draw n samples: size must be an integer")
            res, lab = self._sample(math.prod(shape), output)
            if self.kind == "rows":
                res = _map_arrays(res, lambda a: a.reshape(*shape, *a.shape[1:]))
            lab = lab.reshape(shape)
        if not labels:
            return res
        first = res[0] if isinstance(res, (tuple, list)) else res
        if isinstance(first, torch.Tensor):
            lab = torch.as_tensor(lab, device=first.device)
        return res, lab

    rvs = sample

    def sample_into(self, rows):
        """Fill rows in place: (n, ...) buffers (one per component for correlated fields); for Gaussians, (n, ndof)."""
        bufs = [as_tensor(b) for b in rows] if isinstance(rows, (list, tuple)) else [as_tensor(rows)]
        self._fill(bufs)
        return rows

    def _labels(self, n) -> np.ndarray:
        if len(self.laws) == 1:
            return np.zeros(n, dtype=np.int64)
        u = self.source.streams().labels().random(n)
        return np.searchsorted(np.cumsum(self.weights)[:-1], u, side="right").astype(np.int64)

    def _sample(self, n, output):
        """n draws (allocated as the laws' own draws are), and their labels."""
        lab = self._labels(n)
        if n == 0:
            return _draw(self.laws[0], 0, output), lab
        groups = []
        for l, law in enumerate(self.laws):
            sel = np.flatnonzero(lab == l)
            if sel.size:
                groups.append((sel, _draw(law, sel.size, output)))
        parts = [g[1] if isinstance(g[1], (tuple, list)) else (g[1],) for g in groups]
        out = []
        for j in range(len(parts[0])):
            arrays = [p[j] for p in parts]
            out.append(_assemble(arrays, [g[0] for g in groups], n, self.kind == "columns"))
        return (tuple(out) if isinstance(groups[0][1], (tuple, list)) else out[0]), lab

    def _sample_out(self, size, out):
        bufs = [as_tensor(b) for b in out] if isinstance(out, (list, tuple)) else [as_tensor(out)]
        if self.kind == "columns":  # (..., n): the samples along the last axis
            n = bufs[0].shape[-1] if size is None else int(size)
            if any(b.shape[-1] != n for b in bufs):
                raise ValueError(f"out holds samples along its last axis; size is {n}")
            return self._fill([b.movedim(-1, 0) for b in bufs])
        shape = None if size is None else ((int(size),) if isinstance(size, numbers.Integral)
                                           else tuple(int(s) for s in size))
        k = 1 if shape is None else len(shape)
        if shape is not None and any(tuple(b.shape[:k]) != shape for b in bufs):
            raise ValueError(f"out has shape {tuple(bufs[0].shape)}, size is {shape}")
        n = bufs[0].shape[0] if shape is None else math.prod(shape)
        lab = self._fill([b.reshape(n, *b.shape[k:]) for b in bufs])
        return lab if shape is None else lab.reshape(shape)

    def _fill(self, bufs):
        """Fill (n, ...) buffers (views; NumPy buffers are wrapped) group by group, in chunks of ``work_values``
        values; the labels."""
        n = bufs[0].shape[0]
        labels = np.empty(n, dtype=np.int64)
        per = max(1, sum(b[0].numel() for b in bufs)) if n else 1
        step = max(1, int(default_work_values(bufs[0].device) // per))
        for k0 in range(0, n, step):
            k1 = min(n, k0 + step)
            lab = labels[k0:k1] = self._labels(k1 - k0)
            for l, law in enumerate(self.laws):
                sel = np.flatnonzero(lab == l)
                if not sel.size:
                    continue
                tmp = [torch.empty((sel.size, *b.shape[1:]), dtype=b.dtype, device=b.device) for b in bufs]
                _fill_law(law, tmp, self.kind)
                for b, t in zip(bufs, tmp):
                    b.index_copy_(0, torch.from_numpy(sel + k0).to(b.device), t)
        return labels

    # ------------------------------------------------------------------ dense sampler, observations

    def dense(self, dofs=None, observe=None, *, rank=None, tol=None, rtol=RTOL, pin=None, zero=ZERO,
              max_points=MAX_POINTS, device=None, dtype=None, noise=None, seed=None, output=None):
        """A :class:`ngsrng.DenseSampler` of the mixture: every law's dense law (``d.dense(dofs, observe, rank=,
        tol=, ...)``, each truncated relative to its own variance), drawn together (see :mod:`ngsrng.dense`). Set up
        once per arguments and kept. ``device``, ``dtype``: default the first law's; ``seed``: default a child of
        the mixture's randomness."""
        from .dense import DenseSampler, _digest

        law_kwargs = {"rank": rank, "tol": tol, "rtol": rtol, "pin": pin, "zero": zero, "max_points": max_points}
        key = (_digest(dofs), _digest(observe), tuple(sorted(law_kwargs.items())), str(device), str(dtype), noise,
               _digest(seed), output)
        cache = owned_cache(self, "_ngsrng_dense")
        if key not in cache:
            laws, weights = [], []
            first = None
            for law, w in zip(self.laws, self.weights):
                s = _dense_of(law, dofs, observe, law_kwargs)
                if first is None:
                    first = s
                laws += s.laws
                weights += list(w * s.weights)
            cache[key] = DenseSampler(laws, weights, device=first.device if device is None else device,
                                      dtype=first.dtype if dtype is None else dtype, noise=noise,
                                      seed=self.source.spawn() if seed is None else seed, output=output)
        return cache[key]

    def __rmatmul__(self, operator):
        """``E @ mixture``: the mixture of the observed laws ``E @ law``, with the same weights and labels'
        randomness."""
        return Mixture([operator @ law for law in self.laws], self.weights, seed=self.source)


# ---------------------------------------------------------------------- the laws' interfaces


def _kind(law) -> str:
    """"rows" for laws whose ``sample(n)`` is (n, ...), "columns" for the Gaussians' (..., n)."""
    from .correlated import CorrelatedFields
    from .dense import DenseSampler, ObservedField
    from .generator import FieldDistribution

    if isinstance(law, Mixture):
        return law.kind
    if isinstance(law, (FieldDistribution, CorrelatedFields, ObservedField, DenseSampler)):
        return "rows"
    if hasattr(law, "sample_into") and hasattr(law, "mean") and (hasattr(law, "apply_t") or hasattr(law, "G")):
        return "columns"
    raise TypeError(f"a mixture's laws are distributions, correlated fields, observed laws, dense samplers, mixtures "
                    f"or Gaussians, not {type(law).__name__}")


def _randomness(law) -> Source:
    """The randomness a law draws with (a child of it gives the mixture's labels)."""
    from .correlated import CorrelatedFields
    from .dense import ObservedField
    from .generator import FieldDistribution

    if isinstance(law, FieldDistribution):
        return law.generator.source
    if isinstance(law, CorrelatedFields):
        return law._parts[0].gen.source
    if isinstance(law, ObservedField):
        return _randomness(law.source)
    src = getattr(law, "source", None)
    return src if isinstance(src, Source) else Source()


def _draw(law, n, output):
    """n draws of a law as its own ``sample`` returns them (n=None: one)."""
    if _kind(law) == "columns":
        return law.sample(n)
    return law.sample(n, output=output)


def _fill_law(law, rows, kind):
    """Fill (count, ...) rows (one per component) with draws of ``law``."""
    from .generator import FieldDistribution

    if isinstance(law, FieldDistribution):
        law.sample(out=rows[0])
    elif kind == "columns":
        law.sample_into(rows[0])
    else:
        law.sample_into(rows if len(rows) > 1 else rows[0])


def _assemble(arrays, groups, n, columns):
    """One array of n draws from the groups' draws (NumPy or torch, as they are; the sample axis first, or last for
    Gaussians), each group's at its draws' positions."""
    torch_mode = any(isinstance(a, torch.Tensor) for a in arrays)
    if torch_mode:
        arrays = [torch.as_tensor(a) for a in arrays]
        dt = arrays[0].dtype
        for a in arrays[1:]:
            dt = torch.promote_types(dt, a.dtype)
        dev = arrays[0].device
        first = arrays[0].movedim(-1, 0) if columns else arrays[0]
        out = torch.empty((n, *first.shape[1:]), dtype=dt, device=dev)
        for a, sel in zip(arrays, groups):
            out.index_copy_(0, torch.from_numpy(sel).to(dev), (a.movedim(-1, 0) if columns else a).to(dev, dt))
        return out.movedim(0, -1) if columns else out
    arrays = [np.asarray(a) for a in arrays]
    first = np.moveaxis(arrays[0], -1, 0) if columns else arrays[0]
    out = np.empty((n, *first.shape[1:]), dtype=np.result_type(*arrays))
    for a, sel in zip(arrays, groups):
        out[sel] = np.moveaxis(a, -1, 0) if columns else a
    return np.moveaxis(out, 0, -1) if columns else out


def _map_arrays(res, fn):
    return tuple(fn(r) for r in res) if isinstance(res, (tuple, list)) else fn(res)


def _dense_of(law, dofs, observe, law_kwargs):
    """The dense sampler of one law with the mixture's arguments (a dense sampler: selected or observed; its set-up
    arguments are its own)."""
    from .dense import DenseSampler

    if isinstance(law, DenseSampler):
        if dofs is not None and observe is not None:
            raise ValueError("give dofs= or observe=, not both")
        if dofs is not None:
            pick = np.zeros((len(np.ravel(dofs)), law.nout))
            pick[np.arange(pick.shape[0]), np.ravel(dofs)] = 1.0
            return pick @ law
        return law if observe is None else observe @ law
    return law.dense(dofs, observe, **law_kwargs)
