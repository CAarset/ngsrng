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
Laws from one generator share its set-up (the points, the Bernstein form of the space), so a dense mixture of one
family maps and converts the draws of all its laws at once. ``E @ mixture`` is the mixture of the observed laws.

Mixtures over a parameter
-------------------------
``ngsrng.mixture(factory, over=law)`` is the mixture of ``factory(value)`` over a parameter's law (a ``scipy.stats``
distribution; several parameters: ``over=(law1, law2)`` and a factory of as many arguments). A law with finitely many
values gives the finite mixture of the factory's laws at the values, weighted by their probabilities (any factory). A
continuous law is passed to the factory as itself (a :class:`ngsrng.parameters.RandomParameter`), so the factory must
hand it to a parameter that may be random, ``ngsrng.matern_correlation(length=..., nu=...)``: the result is the field
with random parameters, :class:`ParameterMixture`, the same as writing the law there directly::

    f = lambda length: rng.distribution("uniform", representation="bernstein",
                                        correlation=ngsrng.matern_correlation(length, nu=5, boundary="torus"))
    ngsrng.mixture(f, over=scipy.stats.loguniform(0.1, 0.4))   # the same law as f(scipy.stats.loguniform(0.1, 0.4))

Every mixture over parameters reports each draw's parameter values (``sample(n, parameters=True)``).

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


def mixture(laws, weights=None, seed=None, *, over=None):
    """The finite mixture of ``laws`` with probabilities ``weights`` (default: equal; normalised): each draw follows
    law l with probability weights[l] (see :mod:`ngsrng.mixture`). ``seed``: the randomness of the labels.

    ``over``: ``laws`` is then a factory, value(s) -> law, and the result the mixture of its laws over the
    parameters' law(s) (see "Mixtures over a parameter")."""
    if over is None:
        return Mixture(laws, weights, seed)
    import inspect
    import itertools

    from .parameters import RandomParameter

    if not callable(laws) or weights is not None:
        raise TypeError("mixture(factory, over=law): a factory of the parameter values, and no weights (the law "
                        "gives them)")
    over = tuple(over) if isinstance(over, (tuple, list)) else (over,)
    params = [p if isinstance(p, RandomParameter) else RandomParameter(p) for p in over]
    try:
        names = list(inspect.signature(laws).parameters)[:len(params)]
    except (TypeError, ValueError):
        names = []
    names = names if len(names) == len(params) else [f"parameter{i}" for i in range(len(params))]
    finite = [j for j, p in enumerate(params) if p.finite]
    if len(finite) < len(params):  # a continuous law: handed to the factory as itself, finite ones per value
        made, weights, values = [], [], []
        for combo in itertools.product(*[zip(params[j].values, params[j].weights) for j in finite]):
            args = list(params)
            for j, (v, _) in zip(finite, combo):
                args[j] = float(v)
            law = laws(*args)
            if not isinstance(law, ParameterMixture):
                raise TypeError("a continuous parameter law must reach a parameter that may be random "
                                "(ngsrng.matern_correlation(length=..., nu=...)); the factory gave "
                                f"{type(law).__name__}")
            made.append(law)
            weights.append(math.prod(w for _, w in combo))
            values.append({names[j]: float(v) for j, (v, _) in zip(finite, combo)})
        return made[0] if len(made) == 1 and seed is None else Mixture(made, weights, seed, values=values)
    made, weights, values = [], [], []
    for combo in itertools.product(*[zip(p.values, p.weights) for p in params]):
        made.append(laws(*[float(v) for v, _ in combo]))
        weights.append(math.prod(w for _, w in combo))
        values.append({n: float(v) for n, (v, _) in zip(names, combo)})
    return Mixture(made, weights, seed, values=values)


class Mixture(InPlace):
    """A finite mixture of laws (see :func:`mixture`): ``sample(size, labels=False)``, ``dense()``, ``E @ mixture``,
    ``info``; ``laws``, ``weights``."""

    __array_ufunc__ = None  # E @ mixture: NumPy (and SciPy, torch) leave the product to __rmatmul__

    def __init__(self, laws, weights=None, seed=None, values=None):
        from .dense import _mixture_weights

        self.laws = tuple(laws)
        self.weights = _mixture_weights(weights, len(self.laws))
        # each law's parameter values, for a mixture over parameters ({name: value} per law)
        self.values = None if values is None else [dict(v) for v in values]
        if self.values is not None and len(self.values) != len(self.laws):
            raise ValueError("values: one dict of parameter values per law")
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

    def sample(self, size=None, output=None, out=None, labels=False, parameters=False):
        """Draws as the laws' own ``sample(size)`` returns them ((*size, ...) for distributions; (..., n) for
        Gaussians; one per component for correlated fields), each draw from the law of its label. ``out``: buffers of
        that shape filled in place and returned. ``labels``: also return the labels (int64, of shape ``size``);
        ``parameters``: also the draws' parameter values ({name: array of shape ``size``}; a mixture over
        parameters). Extras follow the samples: ``(samples, labels, parameters)``."""
        if parameters and any(isinstance(law, ParameterMixture) for law in self.laws):
            raise NotImplementedError("parameters of draws of continuous mixtures among the laws: from the dense "
                                      "sampler (mixture.dense().sample(n, parameters=True))")
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
        if not (labels or parameters):
            return res
        first = res[0] if isinstance(res, (tuple, list)) else res
        wrap = (lambda a: torch.as_tensor(a, device=first.device)) if isinstance(first, torch.Tensor) else np.asarray
        extra = [wrap(lab)] if labels else []
        if parameters:
            names = sorted({k for v in (self.values or []) for k in v})
            table = {k: np.array([v.get(k, np.nan) for v in self.values]) for k in names}
            extra.append({k: wrap(t[np.asarray(lab)]) for k, t in table.items()})
        return (res, *extra)

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
            for j, (law, w) in enumerate(zip(self.laws, self.weights)):
                s = _dense_of(law, dofs, observe, law_kwargs)
                if first is None:
                    first = s
                given = {} if self.values is None else self.values[j]
                laws += [{**d, "values": {**d.get("values", {}), **given}} for d in s.laws]
                weights += list(w * s.weights)
            cache[key] = DenseSampler(laws, weights, device=first.device if device is None else device,
                                      dtype=first.dtype if dtype is None else dtype, noise=noise,
                                      seed=self.source.spawn() if seed is None else seed, output=output)
        return cache[key]

    def __rmatmul__(self, operator):
        """``E @ mixture``: the mixture of the observed laws ``E @ law``, with the same weights and labels'
        randomness."""
        return Mixture([operator @ law for law in self.laws], self.weights, seed=self.source, values=self.values)


# ---------------------------------------------------------------------- the laws' interfaces


def _kind(law) -> str:
    """"rows" for laws whose ``sample(n)`` is (n, ...), "columns" for the Gaussians' (..., n)."""
    from .correlated import CorrelatedFields
    from .dense import DenseSampler, ObservedField
    from .generator import FieldDistribution

    if isinstance(law, Mixture):
        return law.kind
    if isinstance(law, (FieldDistribution, CorrelatedFields, ObservedField, DenseSampler, ParameterMixture)):
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
    if isinstance(law, ParameterMixture):
        return law.source
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


class ParameterMixture(InPlace):
    """A field whose correlation parameters have a continuous law: the continuous mixture over them, drawn exactly
    through one spectral basis (:mod:`ngsrng.spectral`). Made by ``rng.distribution(name, ...,
    correlation=ngsrng.matern_correlation(length=law, ...))`` or ``ngsrng.mixture(factory, over=law)``.

    ``sample(size)`` returns what the field's ``sample(size)`` returns (each draw at its own parameters);
    ``dense(dofs, observe, ...)`` its dense sampler (``rank=`` / ``tol=`` truncate each draw relative to its own
    variance); ``E @ mixture`` observes it; ``random``: the parameters' laws."""

    __array_ufunc__ = None

    def __init__(self, generator, name, args, kwargs, spec, observe=None):
        from .stationary import padding_for

        self.generator, self.name, self.args, self.kwargs, self.spec = generator, name, tuple(args), dict(kwargs), spec
        self.random = spec.random
        self.observe = observe
        # the field at the corner of the parameters' range with the largest torus box (any corner otherwise)
        corners = [{}]
        for k, p in self.random.items():
            corners = [{**c, k: v} for c in corners for v in p.support]
        tol = spec.options.get("tol", 1e-3)
        values = {"length": spec.length, "nu": spec.nu}
        ref = max(corners, key=lambda c: padding_for(**{**values, **c}, tol=tol))
        self.template = generator.distribution(name, *args, correlation=spec.at(**ref), **kwargs)
        self.fixed = {k: float(v) for k, v in values.items() if k not in self.random}
        self.source = generator._seed()
        self._kind = "rows"

    @property
    def kind(self):
        return self._kind

    @property
    def info(self) -> dict:
        return {"law": f"{self.name} with random {', '.join(self.random)}: the continuous mixture over them "
                       "(spectral)", "parameters": {k: p.describe() for k, p in self.random.items()},
                "fixed": self.fixed, "field": self.template.info,
                "observed": None if self.observe is None else list(self.observe.shape)}

    def __repr__(self):
        return f"ParameterMixture({self.name}, {', '.join(f'{k} ~ {p.describe()}' for k, p in self.random.items())})"

    def _part(self):
        from .dense import _Part

        return _Part.of_field(self.template.field, self.generator, self.template.name)

    def dense(self, dofs=None, observe=None, *, rank=None, tol=None, rtol=RTOL, pin=None, zero=ZERO,
              max_points=MAX_POINTS, device=None, dtype=None, noise=None, seed=None, output=None):
        """The :class:`ngsrng.DenseSampler` of the mixture (one spectral law; :mod:`ngsrng.dense`): at ``dofs`` or
        under ``observe`` (after this object's own observation). Set up once per arguments and kept. Values with
        zero variance are exact without ``pin`` (their rows of B are zero)."""
        from .dense import DenseSampler, _build_law, _digest
        from .observed import _as_operator

        if pin is not None:
            raise ValueError("pin=: a spectral law's variances depend on the parameters; values with zero variance "
                             "are exact without it")

        if self.observe is not None:
            if dofs is not None and observe is not None:
                raise ValueError("give dofs= or observe=, not both")
            e = self.observe if dofs is None else self.observe[np.asarray(dofs, dtype=np.int64).ravel()]
            observe = e if observe is None else _as_operator(observe, e.shape[0]) @ e
            dofs = None
        cache = owned_cache(self, "_ngsrng_dense")
        law_key = (_digest(dofs), _digest(observe), rank, tol, rtol, zero, int(max_points))
        laws = cache.setdefault("laws", {})
        if law_key not in laws:
            part = self._part()
            laws[law_key] = _build_law([part], np.eye(1), dofs, observe, rank, tol, rtol, None, zero,
                                       int(max_points), spectral={"modes": part.gauss.spectral_modes(),
                                                                  "random": self.random, "values": self.fixed})
        gen = self.generator
        keep64 = gen._output_kind(self.template.field, self.template.name)[1]
        key = law_key + (str(device), str(dtype), noise, _digest(seed), output)
        samplers = cache.setdefault("samplers", {})
        if key not in samplers:
            samplers[key] = DenseSampler([laws[law_key]], device=gen.device if device is None else device,
                                         dtype=(torch.float64 if keep64 else gen.dtype) if dtype is None else dtype,
                                         noise=noise, seed=self.source.spawn() if seed is None else seed,
                                         output=output)
        return samplers[key]

    def sample(self, size=None, output=None, out=None, parameters=False):
        """Draws as the field's ``sample(size)`` returns them ((*size, nvalues); one: a GridFunction for FE
        representations), each at its own parameters; observed: (*size, m). ``parameters``: also return the draws'
        parameter values ({name: array of shape ``size``}), as ``(samples, parameters)``."""
        from .correlated import _Part as Deliver

        part = self._part()
        nv = part.nvalues if self.observe is None else self.observe.shape[0]
        d = self.dense(dofs=np.arange(nv) if self.observe is None else None)
        if out is not None:
            t = as_tensor(out)
            shape = tuple(t.shape[:1]) if size is None else ((int(size),) if isinstance(size, numbers.Integral)
                                                            else tuple(int(s) for s in size))
            _, theta = d.sample(out=t.reshape(math.prod(shape), nv), parameters=True)
            return (out, {k: v.reshape(shape) for k, v in theta.items()}) if parameters else out
        shape = () if size is None else ((int(size),) if isinstance(size, numbers.Integral)
                                         else tuple(int(s) for s in size))
        rows, theta = d.sample(math.prod(shape) if shape else 1, output="torch", parameters=True)
        if self.observe is not None:
            res = rows.reshape(*shape, nv) if shape else rows[0]
            res = res.cpu().numpy() if (output or ("numpy" if rows.device.type == "cpu" else "torch")) == "numpy" \
                else res
        else:
            res = Deliver(self.template).deliver(rows if shape else rows, size, shape, output)
        if not parameters:
            return res
        theta = {k: (v.reshape(shape) if shape else v[0]) for k, v in theta.items()}
        if isinstance(res, np.ndarray) or not shape:
            theta = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in theta.items()}
        return res, theta

    rvs = sample

    def sample_into(self, rows):
        """Fill (n, nvalues) rows in place (observed: (n, m))."""
        self.sample(out=rows)
        return rows

    def __rmatmul__(self, operator):
        """``E @ mixture``: the draws observed by E (m x nvalues), a mixture again (the same parameters' laws)."""
        from .observed import _as_operator

        part = self._part()
        e = _as_operator(operator, part.nvalues if self.observe is None else self.observe.shape[0])
        new = ParameterMixture.__new__(ParameterMixture)
        new.__dict__.update({k: v for k, v in self.__dict__.items() if k != "_ngsrng_dense"})
        new.observe = e if self.observe is None else e @ self.observe
        return new
