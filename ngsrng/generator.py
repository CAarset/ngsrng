"""numpy-style random functions on FE spaces.

    rng = ngsrng.default_rng(fes, seed=1, correlation=ngsrng.matern_correlation(0.2, nu=1))
    u = rng.normal()                       # GridFunction
    k = rng.lognormal(0.0, 0.5, size=100)  # (100, npoints) array of point values
    w = rng.sample("gamma", 2.0, size=10)  # same as rng.gamma(2.0, size=10)

Every distribution is a random *function* whose spatial dependence comes from a Gaussian random function (the
``correlation``: a correlation length, a :class:`MaternCorrelation`, a precision specification, or any Gaussian such
as a posterior). Method names and parameters follow ``numpy.random.Generator``; ``sample(name, ...)`` also accepts
the usual aliases ("gaussian", "chi2", "t", "student_t", torch.distributions-style names such as "StudentT", and,
as a fallback, any scipy.stats distribution name with scipy's parameters).

Output follows numpy: ``size=None`` gives one sample (a GridFunction for scalar fields), otherwise an array of shape
(*size, nvalues, *event_shape). ``out=`` fills a preallocated NumPy array or torch tensor of that shape in place (any
float dtype, any device; ``size`` may then be omitted), batch by batch without full-size intermediates: the way to
draw large batches repeatedly. nvalues is the number of FE dofs for fields that are FE functions (normal,
Student-t process, type-G fields, white noise), and the number of quadrature points for pointwise-defined families in
the default "quadrature" representation (``representation="projection"`` gives FE coefficients of the L2
projection instead). Arrays are NumPy on the CPU and torch tensors on a GPU device (``output=`` overrides).

On a GPU the draws never leave it: point evaluation, the transforms, the projection and Bernstein representations,
``bnd=``, KL expansions, posteriors and their set-ups all run there. ``compute=`` (per call) chooses where a pointwise
family or a KL expansion does this work: "auto" (the GPU, if the Gaussian has one and the family a torch path),
"host" (NumPy and NGSolve on the CPU, the samples copied there) or "device" (the Gaussian's torch device, required).
Families without a torch path always compute on the host: scipy quantiles (shape parameters that vary in space,
``method="ppf"``) and your own NumPy functions in ``transformed`` (NumPy ufuncs and torch functions are recognised and
run on the device; ``torch=True`` declares your own function written for torch tensors).

Algorithms are chosen automatically for every family (and never change its law); ``info(name, ...)`` or
``distribution(name, ...).info`` report the choice.
"""

import copy
import math
import numbers
import warnings

import ngsolve as ngs
import numpy as np
import scipy.stats as st
import torch

from .backend import (
    COMPUTE,
    Rebuildable,
    compute_device,
    device_copy,
    mean_array,
    ngs_sparse,
    resolve_device,
    resolve_dtype,
    same_device,
    sparse_to_torch,
)
from .boundary import attach_boundary  # [bnd] boundary (trace) sampling: ngsrng/boundary.py
from .correlation import MaternCorrelation, matern
from .distributions.bernstein import BernsteinForm, default_ok
from .distributions.expansion import KLField
from .distributions.mixtures import StudentTField, TypeGField
from .distributions.pointwise import GaussianAtPoints
from .distributions.quadratic import ChiSquareField, MatrixLogNormalField, WishartField
from .distributions.transforms import (
    CLOSED,
    SCIPY,
    DirichletField,
    indicator,
    marginal,
    named_marginal,
    transformed,
)
from .gaussian import Gaussian, check_method
from .inplace import InPlace, as_tensor, chunks, default_work_values
from .lowrank import LowRankGaussian, karhunen_loeve
from .observations import observation_operator
from .observed import RTOL, ObservedGaussian
from .posterior import MatheronPosterior
from .seeding import Source, as_source, draw
from .space import SpaceInfo, mass_form
from .specs import AssembledPrecision, OperatorCovariance, OperatorPrecision, PDEPrecision
from .stationary import StationaryMatern

GAUSSIAN_TYPES = (Gaussian, LowRankGaussian, MatheronPosterior, StationaryMatern)  # used as they are

DISCRETE = {"poisson", "binomial", "bernoulli", "negative_binomial", "geometric", "logseries", "zipf",
            "hypergeometric"}
SPECIAL = {"normal", "standard_normal", "white_noise", "chisquare", "standard_t", "dirichlet", "wishart",
           "inverse_wishart", "matrix_lognormal", "normal_inverse_gaussian", "variance_gamma", "indicator",
           "transformed", "marginal", "standard_exponential", "standard_gamma", "karhunen_loeve", "observed"}

ALIASES = {
    "gaussian": "normal", "norm": "normal", "standard_gaussian": "standard_normal", "randn": "standard_normal",
    "rand": "uniform", "chi2": "chisquare", "chi_square": "chisquare", "chisq": "chisquare",
    "t": "standard_t", "student_t": "standard_t", "studentt": "standard_t",
    "lognorm": "lognormal", "log_normal": "lognormal", "expon": "exponential", "exp": "exponential",
    "weibull_min": "weibull", "invgauss": "wald", "inverse_gaussian": "wald",
    "nbinom": "negative_binomial", "geom": "geometric", "binom": "binomial", "logser": "logseries",
    "hypergeom": "hypergeometric", "cauchy": "standard_cauchy", "halfnorm": "halfnormal",
    "lomax": "pareto", "powerlaw": "power", "triang": "triangular", "gumbel_r": "gumbel",
    "fisher_snedecor": "f", "fishersnedecor": "f", "ncx2": "noncentral_chisquare", "ncf": "noncentral_f",
    "invwishart": "inverse_wishart", "inv_wishart": "inverse_wishart",
    "nig": "normal_inverse_gaussian", "vg": "variance_gamma", "matrix_log_normal": "matrix_lognormal",
    "excursion": "indicator", "kl": "karhunen_loeve",
}

CORRELATION_KEYS = ("correlation", "precision", "covariance")
FIELD_KEYS = ("representation", "method", "tol", "std", "order", "construction", "points", "standardize_by",
              "bnd", "bnd_values", "compute")  # [bnd] boundary (trace) sampling: see ngsrng/boundary.py


def _canonical(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    known = set(CLOSED) | set(SCIPY) | SPECIAL
    if key in known:
        return key
    if key in ALIASES:
        return ALIASES[key]
    squashed = {k.replace("_", ""): k for k in known} | {k.replace("_", ""): v for k, v in ALIASES.items()}
    if key.replace("_", "") in squashed:
        return squashed[key.replace("_", "")]
    if isinstance(getattr(st, name, None), (st.rv_continuous, st.rv_discrete)):
        return "scipy:" + name
    raise ValueError(f"unknown distribution {name!r}")


def _nparams(name) -> int:
    if name in CLOSED:
        return len(CLOSED[name][1])
    if name in SCIPY:
        return len(SCIPY[name][1])
    return {"normal": 2, "standard_normal": 0, "white_noise": 0, "chisquare": 1, "standard_t": 1, "dirichlet": 1,
            "wishart": 2, "inverse_wishart": 2, "matrix_lognormal": 2, "normal_inverse_gaussian": 1,
            "variance_gamma": 1, "indicator": 1, "transformed": 1, "marginal": 1, "standard_exponential": 0,
            "standard_gamma": 1, "karhunen_loeve": 1, "observed": 2}.get(name, 0)


def _field_key(name, args, kwargs, corr, opts):
    return name, _freeze(args), _freeze(kwargs), tuple(sorted((k, id(v)) for k, v in corr.items())), _freeze(opts)


def _rekey_gaussian(key, value):
    if key[0] == "id":
        return ("id", id(value[0]))
    if key[0] == "kl":
        return ("kl", id(value[0]), *key[2:])
    return key


def _freeze(v):
    """A hashable stand-in for a parameter (arrays by content, other unhashables by identity)."""
    if isinstance(v, np.ndarray):
        return ("ndarray", v.shape, v.dtype.str, v.tobytes())
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    try:
        hash(v)
        return v
    except TypeError:
        return ("id", id(v))


REQUIRED = object()


def _as_spec(key, value):
    """``covariance=`` also takes an NGSolve BaseMatrix, BilinearForm or callable: wrapped in an OperatorCovariance
    with its default convention (pass an OperatorCovariance for another one)."""
    known = (*GAUSSIAN_TYPES, OperatorCovariance, MaternCorrelation, PDEPrecision, AssembledPrecision,
             OperatorPrecision, numbers.Real, dict)
    if key == "covariance" and not isinstance(value, known) and (
            isinstance(value, (ngs.BaseMatrix, ngs.BilinearForm)) or callable(value)):
        return OperatorCovariance(value)
    return value


class _NormalField(InPlace, Rebuildable):  # pickles; the NGSolve evaluation matrix is made on first use
    """loc + scale (g - mean_g): an exact FE-space Gaussian (scale must be a number), optionally evaluated at
    ``points``."""

    def __init__(self, gauss, loc, scale, points=None):
        if not isinstance(scale, numbers.Real):
            raise ValueError("a spatially varying scale is not an FE-space operation: use representation='quadrature' "
                             "(pointwise) or a precision with variable coefficients")
        self.gauss, self.scale = gauss, float(scale)
        self.space = gauss.space
        if loc is None:
            self.loc = gauss.mean
        else:
            gf = ngs.GridFunction(gauss.space.fes)
            if isinstance(loc, ngs.GridFunction):
                gf.vec.data = loc.vec
            else:
                gf.Set(loc if isinstance(loc, ngs.CoefficientFunction) else ngs.CoefficientFunction(float(loc)))
            self.loc = gf.vec.FV().NumPy().copy()
        self.trivial = loc is None and self.scale == 1.0
        self.evaluation = None if points is None else observation_operator(gauss.space.fes, points=points)
        self.info = {"law": "Gaussian (FE function)" + ("" if points is None else ", values at points"),
                     "gaussian": gauss.plan}

    def sample(self, n=None):
        if self.evaluation is None:
            return self._full(n)
        ns = 1 if n is None else int(n)
        if getattr(self.gauss, "engine", None) == "ngsolve":  # float64 coefficients evaluated in the worker threads
            vals = self.gauss._ngs_rows(ns, np.empty((ns, self.evaluation.shape[0])), evaluate=self._ngs_evaluation(),
                                        add_mean=self.trivial, transform=None if self.trivial else self._affine).T
        else:  # on the Gaussian's device (torch engine): G as a torch sparse matrix there
            full = self._full(ns)
            key = (str(full.device), full.dtype)
            cache = self.__dict__.setdefault("_eval_torch", {})
            if key not in cache:
                cache[key] = sparse_to_torch(self.evaluation, full.device, full.dtype)
            vals = cache[key] @ full
        return vals[:, 0] if n is None else vals

    def sample_into(self, rows):
        """Fill rows (k x ndof, or k x npoints with ``points``) in place (:mod:`ngsrng.inplace`). The CPU engine's
        worker threads write every sample straight into its row (loc, scale and the point evaluation in float64,
        one cast at the end); the torch engine writes on its device; other Gaussians work in chunks."""
        g = self.gauss
        rows = as_tensor(rows)
        width = g.space.ndof if self.evaluation is None else self.evaluation.shape[0]
        if rows[0].numel() != width:
            raise ValueError(f"out holds {rows[0].numel()} values per sample, normal gives {width}")
        rows = rows if rows.ndim == 2 else rows.view(rows.shape[0], width)
        engine = getattr(g, "engine", None)
        if engine == "ngsolve" and rows.device.type == "cpu" and rows.dtype in (torch.float32, torch.float64):
            g._ngs_rows(rows.shape[0], rows.numpy(), evaluate=self._ngs_evaluation(), add_mean=self.trivial,
                        transform=None if self.trivial else self._affine)
            return rows
        if engine == "torch" and self.evaluation is None and same_device(rows.device, g.device) \
                and rows.dtype == g.dtype:
            loc = None if self.trivial else device_copy(self, self.loc, rows.device, rows.dtype)
            for k0, k1 in chunks(rows.shape[0], self._per_sample(), self.work_values):
                g._torch_rows(rows[k0:k1], loc=loc, scale=self.scale)
            return rows
        return super().sample_into(rows)

    def _affine(self, r):
        """loc + scale c for a zero-mean coefficient vector c, in place (float64, in the worker thread)."""
        r *= self.scale
        r += self.loc

    def _ngs_evaluation(self):
        if self.evaluation is None:
            return None
        if getattr(self, "_ngs_eval", None) is None:
            self._ngs_eval = ngs_sparse(self.evaluation)
        return self._ngs_eval

    def _full(self, n=None):
        g = self.gauss
        if n is None:
            gf = g.sample()
            if not self.trivial:
                v = gf.vec.FV().NumPy()
                v[:] = self.loc + self.scale * (v - g.mean)
            return gf
        if not self.trivial and getattr(g, "engine", None) == "ngsolve" and hasattr(g, "_ngs_rows"):
            return torch.from_numpy(self._rows64(int(n))).T.to(g.dtype)  # loc and scale applied in float64
        c = g.sample(int(n))
        if self.trivial:
            return c
        mean = device_copy(self, mean_array(g), c.device, c.dtype)[:, None]
        return device_copy(self, self.loc, c.device, c.dtype)[:, None] + self.scale * (c - mean)

    def _rows64(self, n):
        g = self.gauss
        return g._ngs_rows(n, np.empty((n, g.space.ndof)), add_mean=False, transform=self._affine)

    # A draw in two halves, the Gaussian part and the affine map (ngsrng.correlated mixes between them)
    @property
    def latent_device(self):
        """Where :meth:`_latent` delivers its draws: the Gaussian's GPU (torch tensors), or None (NumPy)."""
        return compute_device("auto", self.gauss)

    def _latent(self, n):
        """n draws of the Gaussian's coefficients (ndof, n) in float64, mean included: on the Gaussian's GPU, if it
        has one (a torch tensor), else NumPy."""
        g, n = self.gauss, int(n)
        if getattr(g, "engine", None) == "ngsolve" and hasattr(g, "_ngs_rows"):
            return g._ngs_rows(n, np.empty((n, g.space.ndof)), add_mean=True).T
        if self.latent_device is not None:
            return g.sample(n).to(torch.float64)
        return g.sample(n).detach().double().cpu().numpy()

    def _latent_mean(self) -> np.ndarray:
        return np.asarray(self.gauss.mean, dtype=np.float64)

    def _from_latent(self, z):
        if self.trivial:
            return z
        if isinstance(z, torch.Tensor):
            loc = device_copy(self, self.loc, z.device, z.dtype)[:, None]
            return loc + self.scale * (z - device_copy(self, mean_array(self.gauss), z.device, z.dtype)[:, None])
        return self.loc[:, None] + self.scale * (z - self.gauss.mean[:, None])



class FieldDistribution:
    """A frozen random-field distribution: ``sample(size=None)`` (alias ``rvs``) and ``info``."""

    def __init__(self, generator, name, field):
        self.generator, self.name, self.field = generator, name, field

    @property
    def info(self) -> dict:
        out = self.field.info
        if getattr(self.field, "bnd", None) is not None:  # [bnd] boundary (trace) sampling: ngsrng/boundary.py
            out = {**out, "bnd": self.field.bnd["info"]}
        return out

    def sample(self, size=None, output=None, out=None):
        """``out``: a buffer (*size, *value shape) filled in place and returned (:mod:`ngsrng.inplace`)."""
        return self.generator._draw(self.field, size, output, self.name, out=out)

    rvs = sample

    def __repr__(self):
        return f"FieldDistribution({self.name}, {self.info})"


class Generator:
    """numpy-style generator of random functions on ``fes``; see the module docstring.

    Parameters
    ----------
    fes : FE space (H1 or L2)
    seed : None (default), a torch.Generator or numpy Generator, or an int / SeedSequence (:mod:`ngsrng.seeding`).
        None follows torch's global random state at every draw, as ``torch.randn`` does: ``torch.manual_seed`` (or
        Lightning's ``seed_everything``) reproduces the draws, and copies in DataLoader workers differ. A generator
        object is read at every draw the same way. An int or SeedSequence gives an own stream, numpy style; every
        Gaussian and mixing stream the generator creates then gets its own child stream.
    correlation (or precision, covariance) : default spatial structure: a correlation length (Matern, nu = 1), a dict of
        :func:`ngsrng.matern_correlation` arguments, a :class:`MaternCorrelation`, a precision spec (PDEPrecision,
        AssembledPrecision, OperatorPrecision), a Gaussian, LowRankGaussian, MatheronPosterior or StationaryMatern
        object (e.g. a posterior, a sample covariance, a Matérn field with constant variance up to the boundary), or a
        covariance given as an operator: an :class:`OperatorCovariance`, or
        (with ``covariance=``) an NGSolve BaseMatrix, BilinearForm or callable, wrapped in one with its default
        convention.
        Can be overridden per call with ``correlation=``, ``precision=`` or ``covariance=``.
    device, dtype, method, tol : passed to the Gaussians (``method`` selects Gaussian algorithms, e.g. {"noise": ...})
    representation : for pointwise-defined families: "quadrature" (default; values at the quadrature points, exact
        marginals there), "bernstein" (an FE function whose values stay inside the family's range at every point:
        Bernstein-Bezier convex hull property; float64 coefficients) or "projection" (L2 projection; no range
        guarantee). With ``bnd=`` the default is "bernstein" ("projection" where it is unsupported: mixed element
        types or variable order).
        Per call, ``points=`` (m x dim coordinates) gives values at any points instead
    order : quadrature order of the "quadrature" representation (use the order your forms integrate with)
    std : pointwise standardisation of copula families ("auto" | "exact" | "lowrank" | "mc" | array), see
        GaussianAtPoints and :func:`ngsrng.local_covariance`
    standardize_by : "prior" (default) or "own": for a conditioned Gaussian (posterior), pointwise families
        standardise with the prior's mean and std, so that e.g. ``uniform(correlation=post)`` is the posterior of the
        prior's uniform field; "own" uses the posterior's copula only
    output : None (NumPy on the CPU, torch on a GPU) | "numpy" | "torch"
    work_values : values per internal chunk when filling ``out=`` buffers in place (bounds the transient memory;
        default 2^20 on the CPU, 2^24 on a GPU, where larger chunks batch better), see :mod:`ngsrng.inplace`
    """

    def __init__(self, fes, seed=None, *, correlation=None, precision=None, covariance=None, device="cpu", dtype=None,
                 method=None, tol=1e-8, representation="quadrature", std="auto", order=None, output=None,
                 standardize_by="prior", work_values=None):
        self.fes = fes
        self.source = as_source(seed)
        given = [(k, v) for k, v in zip(CORRELATION_KEYS, (correlation, precision, covariance)) if v is not None]
        if len(given) > 1:
            raise ValueError("give at most one of correlation, precision, covariance")
        self.correlation = _as_spec(*given[0]) if given else None
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype)
        self.method, self.tol = (None if method is None else check_method(method)), tol
        self.representation, self.std, self.order, self.output = representation, std, order, output
        self.standardize_by = standardize_by
        self.work_values = default_work_values(self.device) if work_values is None else int(work_values)
        self._gaussians, self._points, self._fields, self._bernstein = {}, {}, {}, {}
        self._boundary = {}  # (id(Gaussian), boundary name) -> (Gaussian, BoundarySampler)
        self._factors = {}  # (id(Gaussian), observation key) -> (Gaussian, factor of an ObservedGaussian)

    # ------------------------------------------------------------------ plumbing

    def __setstate__(self, state):
        """After unpickling: caches keyed by object identity get the new objects' identities."""
        self.__dict__.update(state)
        self._gaussians = {_rekey_gaussian(k, v): v for k, v in self._gaussians.items()}
        self._points = {(id(v[0]), k[1], k[2] if isinstance(v[1], str) or v[1] is None else id(v[1]), *k[3:]): v
                        for k, v in self._points.items()}
        self._bernstein = {id(v[0]): v for v in self._bernstein.values()}
        self._fields = {_field_key(k[0], *v[0]): v for k, v in self._fields.items()}
        self._boundary = {(id(v[0]), k[1]): v for k, v in self.__dict__.get("_boundary", {}).items()}
        self._factors = {(id(v[0]), *k[1:]): v for k, v in self.__dict__.get("_factors", {}).items()}

    @property
    def seed_sequence(self):
        """The SeedSequence of an explicitly seeded generator (None for draw-time randomness)."""
        return self.source.seq

    def _seed(self) -> Source:
        """The randomness of a new Gaussian or stream: a child stream, or the same draw-time generator."""
        return self.source.spawn()

    def spawn(self, n):
        """n child generators with the same settings (numpy's ``Generator.spawn``): independent streams for an
        explicitly seeded generator; for draw-time randomness (seed None or a generator) children read the same
        generator, and their draws are independent because every draw takes a new key.

        Children share the parent's deterministic set-up built so far -- Gaussian pipelines (factorisations, noise
        samplers), pointwise standard deviations, Bernstein forms, boundary samplers -- and only get new randomness.
        So the set-up is paid once: draw once from every field in the parent, then spawn one child per worker
        process (children pickle; see the README, "Worker processes")."""
        children = []
        for _ in range(n):
            child = Generator(self.fes, self._seed(), correlation=self.correlation, device=self.device,
                              dtype=self.dtype, method=self.method, tol=self.tol, representation=self.representation,
                              std=self.std, order=self.order, output=self.output, standardize_by=self.standardize_by,
                              work_values=self.work_values)
            child._adopt(self)
            children.append(child)
        return children

    def _adopt(self, parent):
        """Take over ``parent``'s deterministic set-up: its Gaussians with new randomness (sharing their pipelines),
        their point evaluations and standard deviations, Bernstein forms and boundary samplers. Fields themselves
        are rebuilt on first use (cheap given these), with this generator's randomness."""
        own = {}  # id(parent's Gaussian) -> this generator's
        for key, (spec, g) in parent._gaussians.items():
            if key[0] == "kl":
                continue
            if isinstance(g, Gaussian) and g._prior is None:
                own[id(g)] = g._reseeded(self._seed())
            elif isinstance(g, (LowRankGaussian, StationaryMatern)):  # e.g. an operator covariance's KL: share
                own[id(g)] = copy.copy(g)
                own[id(g)].source = as_source(self._seed())
            else:
                continue
            self._gaussians[key] = (spec, own[id(g)])
        for key, (g, kl) in parent._gaussians.items():
            if key[0] == "kl" and id(g) in own:
                self._gaussians[("kl", id(own[id(g)]), *key[2:])] = (own[id(g)], kl)  # modes: deterministic
        for key, (g, std, pts) in parent._points.items():
            if id(g) in own:
                child = copy.copy(pts)
                child.gauss = own[id(g)]
                if pts.reference is g:
                    child.reference = own[id(g)]
                self._points[(id(own[id(g)]), *key[1:])] = (own[id(g)], std, child)
        self._bernstein.update(parent._bernstein)
        for (gid, name), (g, sampler) in parent._boundary.items():
            if gid in own:
                self._boundary[(id(own[gid]), name)] = (own[gid], sampler)
        for key, (g, factor) in parent._factors.items():
            if key[0] in own:
                self._factors[(id(own[key[0]]), *key[1:])] = (own[key[0]], factor)

    def gaussian(self, correlation=None, precision=None, covariance=None):
        """The (cached) Gaussian random function behind a correlation specification."""
        given = [(k, v) for k, v in zip(CORRELATION_KEYS, (correlation, precision, covariance)) if v is not None]
        if len(given) > 1:
            raise ValueError("give at most one of correlation, precision, covariance")
        spec = _as_spec(*given[0]) if given else self.correlation
        if spec is None:
            raise ValueError("no spatial structure: pass correlation= (e.g. a correlation length or "
                             "ngsrng.matern_correlation(...)) to default_rng or to the call")
        if isinstance(spec, GAUSSIAN_TYPES):
            return spec
        if isinstance(spec, numbers.Real):
            spec = matern(float(spec))
        elif isinstance(spec, dict):
            spec = matern(**spec)
        key = (("matern", spec.length, spec.nu, spec.rtol, spec.boundary, spec.beta, repr(sorted(spec.options.items())))
               if isinstance(spec, MaternCorrelation) else ("id", id(spec)))
        if key not in self._gaussians:
            if isinstance(spec, OperatorCovariance):  # a KL expansion (computed with NumPy), sampling on the device
                gauss = spec.bind(self.fes, seed=self._seed(), device=self.device)
            elif isinstance(spec, MaternCorrelation) and spec.stationary:  # a sampler of its own, on the device
                gauss = spec.bind(self.fes, seed=self._seed(), device=self.device)
            else:
                bound = spec.bind(self.fes) if hasattr(spec, "bind") else spec
                if not isinstance(bound, (PDEPrecision, AssembledPrecision, OperatorPrecision)):
                    raise TypeError(f"unsupported correlation specification {type(spec).__name__}")
                gauss = Gaussian(self.fes, bound, method=self.method, device=self.device, dtype=self.dtype,
                                 tol=self.tol, seed=self._seed())
            self._gaussians[key] = (spec, gauss)  # keep spec alive: the key may be its id
        return self._gaussians[key][1]

    def _gaussian_at_points(self, gauss, std=None, order=None, points=None, standardize_by=None,
                            bernstein=False) -> GaussianAtPoints:
        std = self.std if std is None else std
        order = self.order if order is None else order
        standardize_by = self.standardize_by if standardize_by is None else standardize_by
        key = (id(gauss), order, std if isinstance(std, str) or std is None else id(std), _freeze(points),
               standardize_by, bernstein)
        if key not in self._points:
            form = self.bernstein_form(gauss.space) if bernstein else None
            pts = GaussianAtPoints(gauss, order=order, std=std, points=points, standardize_by=standardize_by,
                                   bernstein=form)
            self._points[key] = (gauss, std, pts)
        return self._points[key][2]

    def bernstein_form(self, space) -> BernsteinForm:
        """The (cached) Bernstein-Bezier description of an FE space."""
        if id(space) not in self._bernstein:
            self._bernstein[id(space)] = (space, BernsteinForm(space))
        return self._bernstein[id(space)][1]

    def points(self, order=None) -> np.ndarray:
        """Coordinates (npoints x dim) of the quadrature points that index pointwise-family values."""
        from .quadrature import QuadratureRule

        return QuadratureRule(SpaceInfo(self.fes), self.order if order is None else order).points

    def dx(self, order=None):
        """The integration measure to use with quadrature-point fields in forms: ``a * grad(u) * grad(v) *
        rng.dx()``. Quadrature-point fields can only be evaluated at their own points; with any other rule NGSolve
        6.2 crashes (segfault). The FE representations (``representation="bernstein"``) work with any ``dx``."""
        from .quadrature import QuadratureRule

        rule = QuadratureRule(SpaceInfo(self.fes), self.order if order is None else order)
        if len(rule.rule) > 256:
            warnings.warn(
                f"this rule has {len(rule.rule)} points: NGSolve 6.2 crashes in SIMD evaluation of quadrature-point "
                "fields with such rules (3D, order >= 3). Use representation='bernstein', or a SymbolicBFI with "
                "intrule=... and simd_evaluate=False",
                stacklevel=2,
            )
        et = next(iter(rule.space.mesh.Elements(ngs.VOL))).type
        return ngs.dx(intrules={et: rule.rule})

    # ------------------------------------------------------------------ construction

    def _field(self, name, args, kwargs):
        corr = {k: kwargs.pop(k) for k in CORRELATION_KEYS if k in kwargs}
        opts = {k: kwargs.pop(k) for k in FIELD_KEYS if k in kwargs}
        key = _field_key(name, args, kwargs, corr, opts)
        if key not in self._fields:  # values are kept alive with the entry, so identity keys stay valid
            field = self._build(name, args, dict(kwargs), corr, dict(opts))
            _set_compute(field, name, opts.get("compute", "auto"))
            field.out_dtype = torch.empty(0, dtype=self.dtype).numpy().dtype  # bounded families clamp in it
            if opts.get("bnd") is not None:  # [bnd] boundary (trace) sampling: ngsrng/boundary.py
                attach_boundary(field, opts["bnd"], opts.get("bnd_values"), cache=self._boundary)
            field.work_values = self.work_values  # chunk size of in-place fills (out=)
            self._fields[key] = ((args, kwargs, corr, opts), field)
        return self._fields[key][1]

    def _build(self, name, args, kw, corr, opts):
        explicit_rep = "representation" in opts
        rep = opts.pop("representation", self.representation)
        if opts.get("bnd") is not None and not explicit_rep:  # [bnd] boundary DoFs only exist for a representation
            # whose samples are finite element coefficients; Bernstein keeps the boundary values in the family's range
            rep = "bernstein" if default_ok(SpaceInfo(self.fes)) else "projection"
        method, tol = opts.pop("method", "auto"), opts.pop("tol", 1e-10)
        construction, compute = opts.pop("construction", None), opts.pop("compute", "auto")
        std, order = opts.pop("std", None), opts.pop("order", None)
        points, standardize_by = opts.pop("points", None), opts.pop("standardize_by", None)
        if points is not None:
            points = np.atleast_2d(np.asarray(points, dtype=float))

        def pts(**extra):
            return self._gaussian_at_points(self.gaussian(**corr, **extra), std, order, points, standardize_by,
                                            bernstein=rep == "bernstein" and points is None)

        def params(defaults):
            names = list(defaults)
            out = dict(zip(names, args))
            out.update(kw)
            unknown = [k for k in out if k not in defaults]
            if unknown:
                raise TypeError(f"{name}() got an unexpected parameter {unknown[0]!r}")
            for k, v in defaults.items():
                if k not in out:
                    if v is REQUIRED:
                        raise TypeError(f"{name}() missing parameter {k!r}")
                    out[k] = v
            return out

        if name in ("normal", "standard_normal"):
            p = params({"loc": None, "scale": 1.0}) if name == "normal" else {**params({}), "loc": 0.0, "scale": 1.0}
            if explicit_rep or not isinstance(p["scale"], numbers.Real):  # pointwise-standardised normal marginals
                loc = 0.0 if p["loc"] is None else p["loc"]
                return named_marginal(pts(), "normal", loc, p["scale"], representation=rep)
            return _NormalField(self.gaussian(**corr), p["loc"], p["scale"], points=points)
        if name == "observed":
            p = params({"loc": None, "scale": 1.0, "operator": None, "functionals": None, "averages": None,
                        "adjoint": None, "m": None, "noise": None, "rtol": RTOL})
            gauss = self.gaussian(**corr)
            obs_args = {k: p[k] for k in ("operator", "functionals", "averages", "adjoint", "m")}
            key = (id(gauss), _freeze(obs_args), _freeze(points), _freeze(p["noise"]), p["scale"], p["rtol"], method)
            cached = self._factors.get(key)
            field = ObservedGaussian(gauss, **obs_args, points=points, noise=p["noise"], loc=p["loc"],
                                     scale=p["scale"], method=method, rtol=p["rtol"], seed=self._seed(),
                                     dtype=self.dtype, factor=None if cached is None else cached[1])
            if field.method == "factor":
                self._factors[key] = (gauss, field.factor_data())
            return field
        if name == "white_noise":
            params({})
            return _NormalField(self.gaussian(precision=self._mass_precision()), None, 1.0)
        if name == "standard_exponential":
            params({})
            return named_marginal(pts(), "exponential", 1.0, representation=rep)
        if name == "standard_gamma":
            return named_marginal(pts(), "gamma", params({"shape": REQUIRED})["shape"], 1.0, method=method, tol=tol,
                                  representation=rep)
        if name == "standard_t":
            df = params({"df": REQUIRED})["df"]
            construction = construction or "process"
            if construction == "process":
                return StudentTField(self.gaussian(**corr), df, seed=self._seed())
            if construction == "copula":
                return named_marginal(pts(), "standard_t", df, method=method, tol=tol, representation=rep)
            raise ValueError("standard_t construction must be 'process' or 'copula'")
        if name == "chisquare":
            p = params({"df": REQUIRED, "noncentral": False})
            df = p["df"]
            construction = construction or ("squares" if float(df).is_integer() else "copula")
            if construction == "squares":
                return ChiSquareField(pts(), df, noncentral=p["noncentral"], representation=rep)
            if construction == "copula":
                return named_marginal(pts(), "chisquare", df, method=method, tol=tol, representation=rep)
            raise ValueError("chisquare construction must be 'squares' or 'copula'")
        if name in ("normal_inverse_gaussian", "variance_gamma"):
            nu = params({"nu": 1.0})["nu"]
            mixing = "nig" if name == "normal_inverse_gaussian" else "gamma"
            return TypeGField(self.gaussian(**corr), mixing=mixing, nu=nu, seed=self._seed())
        if name == "karhunen_loeve":
            p = params({"coefficients": "normal", "rank": None})
            gauss = self.gaussian(**corr)
            key = ("kl", id(gauss), p["rank"], compute)
            if key not in self._gaussians:
                self._gaussians[key] = (gauss, karhunen_loeve(gauss, p["rank"], seed=self._seed(), compute=compute))
            return KLField(gauss, self._gaussians[key][1], p["coefficients"], seed=self._seed(),
                           truncated=not (isinstance(gauss, LowRankGaussian) and p["rank"] is None))
        if name == "dirichlet":
            alpha = params({"alpha": REQUIRED})["alpha"]
            return DirichletField(pts(), alpha, method=method, tol=tol, representation=rep)
        if name in ("wishart", "inverse_wishart"):
            p = params({"df": REQUIRED, "scale": REQUIRED})
            return WishartField(pts(), p["df"], p["scale"], inverse=name == "inverse_wishart",
                                construction=construction or "process", representation=rep,
                                seed=self._seed(), method=method, tol=tol)
        if name == "matrix_lognormal":
            p = params({"p": REQUIRED, "sigma": 1.0, "mean": None})
            return MatrixLogNormalField(pts(), p["p"], p["sigma"], p["mean"], representation=rep)
        if name == "indicator":
            p = params({"level": 0.0, "standardize": True})
            return indicator(pts(), p["level"], p["standardize"], representation=rep)
        if name == "transformed":
            p = params({"func": REQUIRED, "standardize": False, "torch": None})
            return transformed(pts(), p["func"], p["standardize"], representation=rep, torch=p["torch"])
        if name == "marginal":
            return marginal(pts(), params({"dist": REQUIRED})["dist"], method=method, tol=tol, representation=rep)
        if name.startswith("scipy:"):
            dist = getattr(st, name[6:])(*args, **kw)
            return marginal(pts(), dist, method=method, tol=tol, representation=rep)
        return named_marginal(pts(), name, *args, method=method, tol=tol, representation=rep, **kw)

    def _mass_precision(self):
        if not hasattr(self, "_mass_spec"):
            self._mass_spec = AssembledPrecision(mass_form)
        return self._mass_spec

    # ------------------------------------------------------------------ output

    @draw
    def _draw_into(self, field, size, name, out):
        """Fill ``out`` (NumPy array or torch tensor, shape (*size, *value shape)) in place: ``field.sample_into``,
        which works in chunks of ``work_values`` values (:mod:`ngsrng.inplace`)."""
        target = as_tensor(out)
        if not target.is_contiguous():
            raise ValueError("out must be contiguous")
        if getattr(field, "representation", None) == "bernstein" and target.dtype != torch.float64:
            raise ValueError("the Bernstein representation's range guarantee is about float64 coefficients: pass a "
                             "float64 out")
        if size is None:
            shape = tuple(target.shape[:1])
        else:
            shape = (int(size),) if isinstance(size, numbers.Integral) else tuple(int(s) for s in size)
            if tuple(target.shape[: len(shape)]) != shape:
                raise ValueError(f"out has shape {tuple(target.shape)}, size is {shape}")
        n = math.prod(shape)
        flat = target.view(n, -1)
        field.sample_into(flat)
        if getattr(field, "bnd", None) is not None:  # [bnd] in place; the boundary step bounds its own temporaries
            self._apply_bnd(field, flat)
        return out

    @staticmethod
    def _output_kind(field, name):
        """(discrete, keep64) of a field's output. FE coefficients (projection, bernstein) are not values: no integer
        cast; bernstein stays in float64, since its range guarantee is about these exact coefficients."""
        rep = getattr(field, "representation", None)
        coefficients = rep in ("projection", "bernstein")
        return (name in DISCRETE or field.info.get("discrete", False)) and not coefficients, rep == "bernstein"

    @draw
    def _draw(self, field, size, output, name, out=None):
        if out is not None:
            return self._draw_into(field, size, name, out)
        discrete, keep64 = self._output_kind(field, name)
        if size is None:
            out = field.sample(None)
            if isinstance(out, ngs.GridFunction):
                if getattr(field, "bnd", None) is not None:  # [bnd]
                    self._apply_bnd(field, out.vec.FV().NumPy())
                return out
            return self._convert(out, output, discrete, keep64)
        shape = (int(size),) if isinstance(size, numbers.Integral) else tuple(int(s) for s in size)
        n = math.prod(shape)
        vals = field.sample(n)
        if getattr(field, "bnd", None) is not None:  # [bnd] in place on the (ndof, n) sample array
            self._apply_bnd(field, vals.T)
        if isinstance(vals, torch.Tensor):
            vals = vals.movedim(-1, 0).reshape(*shape, *vals.shape[:-1])
        else:
            vals = np.moveaxis(vals, -1, 0).reshape(*shape, *vals.shape[:-1])
        return self._convert(vals, output, discrete, keep64)

    def _apply_bnd(self, field, rows):
        """Boundary (trace) step of a ``bnd=`` draw, on ``rows`` (samples by DoFs, in place, NumPy or torch).

        See :mod:`ngsrng.boundary`: without ``bnd_values`` the draw is restricted to the boundary DoFs (their exact
        marginal); with them it is conditioned on that trace.
        """
        b = field.bnd
        sampler = b["sampler"]
        if rows.shape[-1] != sampler.fes.ndof:
            raise ValueError(f'bnd="{sampler.name}" restricts the DoFs of the samples: expected {sampler.fes.ndof} '
                             f'values per sample, got {rows.shape[-1]}. This distribution does not deliver finite '
                             "element coefficients (use representation='projection' or 'bernstein').")
        if b["values"] is not None:
            sampler.condition(rows, b["values"])
        else:
            sampler.restrict(rows)

    def _convert(self, vals, output, discrete, keep64=False):
        output = output or self.output or ("torch" if self.device.type == "cuda" else "numpy")
        dtype = torch.float64 if keep64 else self.dtype
        if output == "numpy":
            if isinstance(vals, torch.Tensor):
                vals = vals.detach().cpu().numpy()
            # C order: samples are contiguous rows (internal arrays are often transposed views)
            target = np.int64 if discrete else torch.empty(0, dtype=dtype).numpy().dtype
            try:
                with np.errstate(over="raise"):
                    return vals.astype(target, order="C", copy=False)
            except FloatingPointError:
                warnings.warn(f"values beyond the range of {np.dtype(target).name} became +-inf in the output; "
                              "pass dtype='float64' for heavy-tailed fields", RuntimeWarning, stacklevel=4)
                with np.errstate(over="ignore"):
                    return vals.astype(target, order="C", copy=False)
        if output == "torch":
            t = torch.as_tensor(vals)
            return t.to(self.device, torch.int64 if discrete else dtype).contiguous()
        raise ValueError("output must be 'numpy' or 'torch'")

    # ------------------------------------------------------------------ public API

    def _split(self, name, args, kwargs):
        size = kwargs.pop("size", None)
        output = kwargs.pop("output", None)
        out = kwargs.pop("out", None)
        k = _nparams(name)
        # numpy allows size as the positional argument after the parameters (scipy fallbacks: size by keyword only)
        if len(args) > k and not name.startswith("scipy:"):
            if size is not None or len(args) > k + 1:
                raise TypeError(f"too many positional arguments for {name}()")
            args, size = args[:k], args[k]
        return args, size, output, out

    def sample(self, name, *args, **kwargs):
        """Draw from the named distribution: ``rng.sample("gamma", 2.0, size=10)`` == ``rng.gamma(2.0, size=10)``.

        ``bnd="name"`` restricts the draw to the DoFs of the named boundary: the sample is zero at every DoF whose
        basis function vanishes there, and its trace is an exact draw of the boundary marginal of the field, for
        every family and representation (ngsrng/boundary.py). With ``bnd_values=g`` it instead draws the whole field
        conditioned on the prescribed trace g (affine Gaussian fields), exactly.
        """
        name = _canonical(name)
        args, size, output, out = self._split(name, args, kwargs)
        return self._draw(self._field(name, args, kwargs), size, output, name, out=out)

    def distribution(self, name, *args, **kwargs) -> FieldDistribution:
        """A frozen distribution (set-up done once): ``d = rng.distribution("beta", 2, 5); d.sample(100)``."""
        name = _canonical(name)
        if "size" in kwargs or "output" in kwargs or "out" in kwargs:
            raise TypeError("pass size/output/out to FieldDistribution.sample")
        return FieldDistribution(self, name, self._field(name, args, kwargs))

    def info(self, name, *args, **kwargs) -> dict:
        """The law and the automatically chosen algorithms for a distribution."""
        return self.distribution(name, *args, **kwargs).info

    def normal(self, loc=None, scale=1.0, size=None, **kwargs):
        """Gaussian random function with mean ``loc`` (number, CoefficientFunction or GridFunction, interpolated;
        default: the mean of the correlation's Gaussian, 0 for specifications) and covariance scale^2 C.

        An exact FE-space Gaussian with all the Gaussian machinery (priors, posteriors, fractional and apply-only
        precisions). With ``representation="quadrature"``/``"projection"``, a pointwise-standardised Gaussian
        (exactly N(loc, scale^2) at every point, loc and scale may vary in space) instead. ``bnd=`` and its two
        companions restrict or prescribe the trace; see :meth:`sample`.
        """
        return self.sample("normal", loc, scale, size=size, **kwargs)

    def standard_normal(self, size=None, **kwargs):
        return self.sample("standard_normal", size=size, **kwargs)

    def white_noise(self, size=None, **kwargs):
        """L2 projection of Gaussian white noise into the FE space: coefficients ~ N(0, M^-1). Not a function-space
        random variable in the limit (its norm grows under refinement); the building block of SPDE samplers."""
        return self.sample("white_noise", size=size, **kwargs)

    def chisquare(self, df, size=None, **kwargs):
        """Chi-square field. ``construction``: "squares" (default for integer df: the chi-square process, sum of df
        squared independent fields) or "copula" (any df > 0: Gaussian copula with chi^2_df marginals, one field)."""
        return self.sample("chisquare", df, size=size, **kwargs)

    def standard_t(self, df, size=None, **kwargs):
        """Student-t field. ``construction``: "process" (default: elliptical t-process, an FE function with
        multivariate-t finite-dimensional laws) or "copula" (pointwise t_df marginals, Gaussian copula)."""
        return self.sample("standard_t", df, size=size, **kwargs)

    def dirichlet(self, alpha, size=None, **kwargs):
        """Simplex-valued field with Dirichlet(alpha) marginals: (*size, npoints, K)."""
        return self.sample("dirichlet", alpha, size=size, **kwargs)

    def wishart(self, df, scale, size=None, **kwargs):
        """SPD-matrix field with Wishart_p(scale, df) marginals: (*size, npoints, p, p). ``construction``:
        "process" (default, Wishart process) or "bartlett" (Bartlett factor with Haar rotation; any real df > p-1)."""
        return self.sample("wishart", df, scale, size=size, **kwargs)

    def inverse_wishart(self, df, scale, size=None, **kwargs):
        """SPD-matrix field with inverse-Wishart_p(df, scale) marginals (scipy convention)."""
        return self.sample("inverse_wishart", df, scale, size=size, **kwargs)

    def matrix_lognormal(self, p, sigma=1.0, size=None, **kwargs):
        """SPD-matrix field expm(mean + sigma G) with a symmetric Gaussian matrix field G: (*size, npoints, p, p)."""
        return self.sample("matrix_lognormal", p, sigma, size=size, **kwargs)

    def normal_inverse_gaussian(self, nu=1.0, size=None, **kwargs):
        """Type-G field driven by normal-inverse-Gaussian noise (an FE function, Gaussian covariance, heavy tails)."""
        return self.sample("normal_inverse_gaussian", nu, size=size, **kwargs)

    def variance_gamma(self, nu=1.0, size=None, **kwargs):
        """Type-G field driven by variance-gamma (Laplace-type) noise."""
        return self.sample("variance_gamma", nu, size=size, **kwargs)

    def observed(self, loc=None, scale=1.0, size=None, **kwargs):
        """Finite observations y = G f (+ e) of the Gaussian random function f ~ N(loc, scale^2 C): values (*size, m).

        The observation map G is any finite set of bounded linear functionals (``points=``, ``averages=``,
        ``functionals=``, ``operator=`` a matrix or LinearOperator, ``adjoint=`` with ``m=``; see
        :func:`ngsrng.observation_operator`); ``noise=`` adds e ~ N(0, Gamma): variance(s), a covariance matrix, or an
        independent :class:`ngsrng.ObservedGaussian` (coloured noise from another random field). Exact:
        ``method="factor"`` (an m x m factor, set up once per generator and shared with spawned children; dense
        draws, the fast route for many draws), ``"pathwise"`` (a field draw and G per sample) or ``"auto"``; see
        :mod:`ngsrng.observed`. ``out=`` fills a preallocated (n, m) buffer, also on the GPU. Unlike
        ``normal(points=...)``, also for observations other than point values and with noise.
        """
        return self.sample("observed", loc, scale, size=size, **kwargs)

    def karhunen_loeve(self, coefficients="normal", rank=None, size=None, **kwargs):
        """f = m + sum_k sqrt(lambda_k) xi_k phi_k: the Karhunen-Loeve expansion of the Gaussian with iid
        standardised ``coefficients`` xi_k ("normal", "uniform", "rademacher", "laplace", a frozen scipy.stats
        distribution, or a callable (rng, shape) -> values with mean 0 and variance 1).

        Its covariance is exactly the Gaussian's (finite-rank Gaussians: sample covariances, operator covariances,
        KL expansions) or exactly its rank-``rank`` truncation (other Gaussians; ``rank`` is then required), whatever
        the coefficient law: unlike pointwise families, the covariance belongs to the field itself. Bounded
        coefficients give bounded fields. An FE function exactly.
        """
        return self.sample("karhunen_loeve", coefficients, rank=rank, size=size, **kwargs)

    def standard_exponential(self, size=None, **kwargs):
        return self.sample("standard_exponential", size=size, **kwargs)

    def standard_gamma(self, shape, size=None, **kwargs):
        return self.sample("standard_gamma", shape, size=size, **kwargs)

    def indicator(self, level=0.0, size=None, **kwargs):
        """Excursion-set indicator 1{z > level} of the standardised field."""
        return self.sample("indicator", level, size=size, **kwargs)

    def transformed(self, func, size=None, **kwargs):
        """Pointwise map func(g) of the Gaussian point values (``standardize=True``: of the standardised values).

        func gets the array type it is written for, on any device: NumPy ufuncs (``np.tanh``) run as their torch twins
        on a GPU, torch functions (``torch.tanh``) get torch tensors (CPU tensors on the CPU engine); pass
        ``torch=True`` for your own function written for torch tensors. Any other function gets NumPy arrays, so on a
        GPU the values are copied to the host for it."""
        return self.sample("transformed", func, size=size, **kwargs)

    def marginal(self, dist, size=None, **kwargs):
        """Gaussian-copula field with any frozen scipy.stats marginal ``dist``."""
        return self.sample("marginal", dist, size=size, **kwargs)

    def as_gridfunction(self, values, order=None, kind="auto") -> ngs.GridFunction:
        """A GridFunction from one sample's values.

        ``kind="coefficients"``: FE coefficients (length ndof; Gaussian-type fields, ``representation="bernstein"`` or
        ``"projection"``), giving a GridFunction on the generator's space. ``kind="points"``: values at the quadrature
        points of ``order`` (the default representation), giving a GridFunction on an **IntegrationRuleSpace**: a
        coefficient for forms integrated with ``rng.dx(order)``, but not an FE function. It can only be evaluated at
        its own points; drawing it or evaluating it elsewhere crashes NGSolve 6.2. Use
        ``representation="bernstein"`` for an FE function. ``kind="auto"`` decides by length and refuses a length that
        fits both.
        """
        from .quadrature import QuadratureRule

        values = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
        order = self.order if order is None else order
        if kind == "auto":
            npoints = QuadratureRule.count(SpaceInfo(self.fes), order)
            if values.shape[0] == self.fes.ndof == npoints:
                raise ValueError(f"{npoints} values fit both the coefficients and the quadrature points: pass "
                                 "kind='coefficients' or kind='points'")
            kind = "coefficients" if values.shape[0] == self.fes.ndof else "points"
        if kind == "coefficients":
            if values.shape[0] != self.fes.ndof:
                raise ValueError(f"{values.shape[0]} values, the space has {self.fes.ndof} dofs")
            gf = ngs.GridFunction(self.fes)
            gf.vec.FV().NumPy()[:] = values
            return gf
        if kind != "points":
            raise ValueError("kind must be 'auto', 'coefficients' or 'points'")
        rule = QuadratureRule(SpaceInfo(self.fes), order)
        if values.shape[0] != rule.npoints:
            raise ValueError(f"{values.shape[0]} values, the space has {self.fes.ndof} dofs and {rule.npoints} "
                             f"quadrature points (order {rule.order})")
        return rule.to_gridfunction(values)

    def __repr__(self):
        return f"Generator({self.fes.type}, ndof={self.fes.ndof}, device={self.device}, dtype={self.dtype})"


def _set_compute(field, name, compute):
    """Where a pointwise family processes its draws (``compute=``, see PointwiseField); KL expansions take it at
    construction (where the modes are computed and the expansion samples); other families follow their Gaussian."""
    from .distributions.pointwise import PointwiseField

    if compute not in COMPUTE:
        raise ValueError(f"compute must be one of {', '.join(COMPUTE)}, not {compute!r}")
    if isinstance(field, PointwiseField):
        field.compute = compute
        field.check_compute(name)
    elif compute != "auto" and not isinstance(field, KLField):
        raise ValueError(f"compute= applies to pointwise families and karhunen_loeve; {name} computes where its "
                         "Gaussian does")


def _make_method(name):
    defaults = (CLOSED.get(name) or SCIPY.get(name))[1]

    def method(self, *args, **kwargs):
        return self.sample(name, *args, **kwargs)

    sig = ", ".join(k if v is None else f"{k}={v}" for k, v in defaults.items())
    method.__name__ = name
    method.__doc__ = (f"{name}({sig}, size=None, **options): Gaussian-copula field with numpy's {name} marginals "
                      "(parameters may be CoefficientFunctions for spatially varying marginals).")
    return method


for _name in list(CLOSED) + list(SCIPY):
    if not hasattr(Generator, _name) and _name not in ("normal",):
        setattr(Generator, _name, _make_method(_name))


def default_rng(fes, seed=None, **kwargs) -> Generator:
    """A :class:`Generator` of random functions on ``fes`` (numpy's ``default_rng`` with a space)."""
    return Generator(fes, seed, **kwargs)
