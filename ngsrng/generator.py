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

Algorithms are chosen automatically for every family (and never change its law); ``info(name, ...)`` or
``distribution(name, ...).info`` report the choice.
"""

import math
import numbers
import warnings

import ngsolve as ngs
import numpy as np
import scipy.stats as st
import torch

from .backend import resolve_device, resolve_dtype
from .correlation import MaternCorrelation, matern
from .distributions.bernstein import BernsteinForm
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
from .gaussian import Gaussian
from .lowrank import LowRankGaussian, karhunen_loeve
from .observations import observation_operator
from .posterior import MatheronPosterior
from .space import mass_form
from .specs import AssembledPrecision, OperatorCovariance, OperatorPrecision, PDEPrecision

DISCRETE = {"poisson", "binomial", "bernoulli", "negative_binomial", "geometric", "logseries", "zipf",
            "hypergeometric"}
SPECIAL = {"normal", "standard_normal", "white_noise", "chisquare", "standard_t", "dirichlet", "wishart",
           "inverse_wishart", "matrix_lognormal", "normal_inverse_gaussian", "variance_gamma", "indicator",
           "transformed", "marginal", "standard_exponential", "standard_gamma", "karhunen_loeve"}

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
FIELD_KEYS = ("representation", "method", "tol", "std", "order", "construction", "points", "standardize_by")


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
            "standard_gamma": 1, "karhunen_loeve": 1}.get(name, 0)


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
    known = (Gaussian, LowRankGaussian, MatheronPosterior, OperatorCovariance, MaternCorrelation, PDEPrecision,
             AssembledPrecision, OperatorPrecision, numbers.Real, dict)
    if key == "covariance" and not isinstance(value, known) and (
            isinstance(value, (ngs.BaseMatrix, ngs.BilinearForm)) or callable(value)):
        return OperatorCovariance(value)
    return value


class _NormalField:
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
        if self.evaluation is not None:
            full = self._full(1 if n is None else n)
            vals = self.evaluation @ full.double().cpu().numpy()
            return vals[:, 0] if n is None else vals
        return self._full(n)

    def sample_into(self, rows) -> bool:
        """Fill rows (k x ndof) with samples directly where possible: the CPU engine writes them from its threads
        (float32 rows by casting the float64 coefficients), the torch engine on its device. Returns False otherwise
        (the caller then copies from ``sample``)."""
        g = self.gauss
        if (self.trivial and self.evaluation is None and getattr(g, "engine", None) == "torch"
                and rows.device == g.device and rows.dtype == g.dtype and hasattr(g, "_torch_rows")):
            g._torch_rows(rows)  # GPU: written on the device, no transposed intermediate
            return True
        if (self.evaluation is not None or getattr(g, "engine", None) != "ngsolve" or not hasattr(g, "_ngs_rows")
                or rows.device.type != "cpu" or rows.dtype not in (torch.float32, torch.float64)):
            return False
        if rows.shape[1] != g.space.ndof:
            raise ValueError(f"out holds {rows.shape[1]} values per sample, normal gives {g.space.ndof}")
        if self.trivial:
            g._ngs_rows(rows.shape[0], rows.numpy())
        else:
            rows.copy_(torch.from_numpy(self._rows64(rows.shape[0])))  # float64: loc and scale exactly
        return True

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
        mean = torch.from_numpy(g.mean).to(c.device, c.dtype)[:, None]
        return torch.from_numpy(self.loc).to(c.device, c.dtype)[:, None] + self.scale * (c - mean)

    def _rows64(self, n):
        g = self.gauss
        full = g._ngs_rows(n, np.empty((n, g.space.ndof)))
        full -= g.mean
        full *= self.scale
        full += self.loc
        return full


class FieldDistribution:
    """A frozen random-field distribution: ``sample(size=None)`` (alias ``rvs``) and ``info``."""

    def __init__(self, generator, name, field):
        self.generator, self.name, self.field = generator, name, field

    @property
    def info(self) -> dict:
        return self.field.info

    def sample(self, size=None, output=None, out=None):
        return self.generator._draw(self.field, size, output, self.name, out=out)

    rvs = sample

    def __repr__(self):
        return f"FieldDistribution({self.name}, {self.info})"


class Generator:
    """numpy-style generator of random functions on ``fes``; see the module docstring.

    Parameters
    ----------
    fes : FE space (H1 or L2)
    seed : int, SeedSequence or None. Every Gaussian and mixing stream the generator creates gets its own child seed.
    correlation (or precision, covariance) : default spatial structure: a correlation length (Matern, nu = 1), a dict of
        :func:`ngsrng.matern_correlation` arguments, a :class:`MaternCorrelation`, a precision spec (PDEPrecision,
        AssembledPrecision, OperatorPrecision), a Gaussian, LowRankGaussian or MatheronPosterior object (e.g. a
        posterior, a sample covariance), or a covariance given as an operator: an :class:`OperatorCovariance`, or
        (with ``covariance=``) an NGSolve BaseMatrix, BilinearForm or callable, wrapped in one with its default
        convention.
        Can be overridden per call with ``correlation=``, ``precision=`` or ``covariance=``.
    device, dtype, method, tol : passed to the Gaussians (``method`` selects Gaussian algorithms, e.g. {"noise": ...})
    representation : for pointwise-defined families: "quadrature" (default; values at the quadrature points, exact
        marginals there), "bernstein" (an FE function whose values stay inside the family's range at every point:
        Bernstein-Bezier convex hull property; float64 coefficients) or "projection" (L2 projection; no range
        guarantee). Per call, ``points=`` (m x dim coordinates) gives values at any points instead
    order : quadrature order of the "quadrature" representation (use the order your forms integrate with)
    std : pointwise standardisation of copula families ("auto" | "exact" | "lowrank" | "mc" | array), see
        GaussianAtPoints and :func:`ngsrng.local_covariance`
    standardize_by : "prior" (default) or "own": for a conditioned Gaussian (posterior), pointwise families
        standardise with the prior's mean and std, so that e.g. ``uniform(correlation=post)`` is the posterior of the
        prior's uniform field; "own" uses the posterior's copula only
    output : None (NumPy on the CPU, torch on a GPU) | "numpy" | "torch"
    """

    def __init__(self, fes, seed=None, *, correlation=None, precision=None, covariance=None, device="cpu", dtype=None,
                 method=None, tol=1e-8, representation="quadrature", std="auto", order=None, output=None,
                 standardize_by="prior"):
        self.fes = fes
        self.seed_sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        given = [(k, v) for k, v in zip(CORRELATION_KEYS, (correlation, precision, covariance)) if v is not None]
        if len(given) > 1:
            raise ValueError("give at most one of correlation, precision, covariance")
        self.correlation = _as_spec(*given[0]) if given else None
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype)
        self.method, self.tol = method, tol
        self.representation, self.std, self.order, self.output = representation, std, order, output
        self.standardize_by = standardize_by
        self._np = np.random.default_rng(self.seed_sequence.spawn(1)[0])
        self._gaussians, self._points, self._fields, self._bernstein = {}, {}, {}, {}

    # ------------------------------------------------------------------ plumbing

    def _seed(self) -> int:
        return int(self.seed_sequence.spawn(1)[0].generate_state(1, np.uint64)[0] >> np.uint64(2))

    def spawn(self, n):
        """n independent child generators with the same settings (numpy's ``Generator.spawn``)."""
        return [Generator(self.fes, s, correlation=self.correlation, device=self.device, dtype=self.dtype,
                          method=self.method, tol=self.tol, representation=self.representation, std=self.std,
                          order=self.order, output=self.output, standardize_by=self.standardize_by)
                for s in self.seed_sequence.spawn(n)]

    def gaussian(self, correlation=None, precision=None, covariance=None):
        """The (cached) Gaussian random function behind a correlation specification."""
        given = [(k, v) for k, v in zip(CORRELATION_KEYS, (correlation, precision, covariance)) if v is not None]
        if len(given) > 1:
            raise ValueError("give at most one of correlation, precision, covariance")
        spec = _as_spec(*given[0]) if given else self.correlation
        if spec is None:
            raise ValueError("no spatial structure: pass correlation= (e.g. a correlation length or "
                             "ngsrng.matern_correlation(...)) to default_rng or to the call")
        if isinstance(spec, (Gaussian, LowRankGaussian, MatheronPosterior)):
            return spec
        if isinstance(spec, numbers.Real):
            spec = matern(float(spec))
        elif isinstance(spec, dict):
            spec = matern(**spec)
        key = (("matern", spec.length, spec.nu, spec.rtol, spec.boundary, spec.robin)
               if isinstance(spec, MaternCorrelation) else ("id", id(spec)))
        if key not in self._gaussians:
            if isinstance(spec, OperatorCovariance):
                gauss = spec.bind(self.fes, seed=self._seed())
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

        return QuadratureRule(self.gaussian().space, self.order if order is None else order).points

    def dx(self, order=None):
        """The integration measure to use with quadrature-point fields in forms: ``a * grad(u) * grad(v) *
        rng.dx()``. Quadrature-point fields can only be evaluated at their own points; with any other rule NGSolve
        6.2 crashes (segfault). The FE representations (``representation="bernstein"``) work with any ``dx``."""
        from .quadrature import QuadratureRule

        rule = QuadratureRule(self.gaussian().space, self.order if order is None else order)
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
        key = (name, _freeze(args), _freeze(kwargs), tuple(sorted((k, id(v)) for k, v in corr.items())), _freeze(opts))
        if key not in self._fields:  # values are kept alive with the entry, so identity keys stay valid
            field = self._build(name, args, dict(kwargs), corr, dict(opts))
            field.out_dtype = torch.empty(0, dtype=self.dtype).numpy().dtype  # bounded families clamp in it
            self._fields[key] = ((args, kwargs, corr, opts), field)
        return self._fields[key][1]

    def _build(self, name, args, kw, corr, opts):
        explicit_rep = "representation" in opts
        rep = opts.pop("representation", self.representation)
        method, tol = opts.pop("method", "auto"), opts.pop("tol", 1e-10)
        construction = opts.pop("construction", None)
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
            for k, v in defaults.items():
                if k not in out:
                    if v is REQUIRED:
                        raise TypeError(f"{name}() missing parameter {k!r}")
                    out[k] = v
            return out

        if name in ("normal", "standard_normal"):
            p = params({"loc": None, "scale": 1.0}) if name == "normal" else {"loc": 0.0, "scale": 1.0}
            if explicit_rep or not isinstance(p["scale"], numbers.Real):  # pointwise-standardised normal marginals
                loc = 0.0 if p["loc"] is None else p["loc"]
                return named_marginal(pts(), "normal", loc, p["scale"], representation=rep)
            return _NormalField(self.gaussian(**corr), p["loc"], p["scale"], points=points)
        if name == "white_noise":
            return _NormalField(self.gaussian(precision=self._mass_precision()), None, 1.0)
        if name == "standard_exponential":
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
            key = ("kl", id(gauss), p["rank"])
            if key not in self._gaussians:
                self._gaussians[key] = (gauss, karhunen_loeve(gauss, p["rank"], seed=self._seed()))
            return KLField(gauss, self._gaussians[key][1], p["coefficients"], seed=self._seed(),
                           truncated=not (isinstance(gauss, LowRankGaussian) and p["rank"] is None))
        if name == "dirichlet":
            alpha = params({"alpha": REQUIRED})["alpha"]
            return DirichletField(pts(), alpha, method=method, tol=tol, representation=rep)
        if name in ("wishart", "inverse_wishart"):
            p = params({"df": REQUIRED, "scale": REQUIRED})
            return WishartField(pts(), p["df"], p["scale"], inverse=name == "inverse_wishart",
                                construction=construction or "process", representation=rep,
                                rng=np.random.default_rng(self._seed()), method=method, tol=tol)
        if name == "matrix_lognormal":
            p = params({"p": REQUIRED, "sigma": 1.0, "mean": None})
            return MatrixLogNormalField(pts(), p["p"], p["sigma"], p["mean"], representation=rep)
        if name == "indicator":
            p = params({"level": 0.0, "standardize": True})
            return indicator(pts(), p["level"], p["standardize"], representation=rep)
        if name == "transformed":
            p = params({"func": REQUIRED, "standardize": False})
            return transformed(pts(), p["func"], p["standardize"], representation=rep)
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

    BATCH_VALUES = 2e7  # values per internal batch of ``out=`` draws (bounds the intermediates)

    def _draw_into(self, field, size, name, out):
        """Fill ``out`` (NumPy array or torch tensor, shape (*size, *value shape)) in place, batch by batch."""
        target = torch.from_numpy(out) if isinstance(out, np.ndarray) else out
        if not isinstance(target, torch.Tensor):
            raise TypeError("out must be a NumPy array or a torch tensor")
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
        per = flat.shape[1]
        # bound the output values and the Gaussian coefficients behind them (ndof per sample; values at a few points
        # would otherwise put the whole batch's coefficients, and the solvers' work blocks, in memory at once)
        ndof = getattr(getattr(field, "space", None), "ndof", 0)
        batch = max(1, int(self.BATCH_VALUES // max(per, ndof, 1)))
        for k0 in range(0, n, batch):
            rows = flat[k0 : k0 + batch]
            if hasattr(field, "sample_into") and field.sample_into(rows):
                continue
            vals = field.sample(rows.shape[0])
            vals = vals if isinstance(vals, torch.Tensor) else torch.from_numpy(vals)
            vals = vals.movedim(-1, 0).reshape(rows.shape[0], -1)
            if vals.shape[1] != per:
                raise ValueError(f"out holds {per} values per sample, {name} gives {vals.shape[1]}")
            rows.copy_(vals)  # cast (and device transfer) in one pass
        return out

    def _draw(self, field, size, output, name, out=None):
        if out is not None:
            return self._draw_into(field, size, name, out)
        rep = getattr(field, "representation", None)
        # FE coefficients (projection, bernstein) are not values: no integer cast; bernstein stays in float64, since
        # its range guarantee is about these exact coefficients
        coefficients = rep in ("projection", "bernstein")
        discrete = (name in DISCRETE or field.info.get("discrete", False)) and not coefficients
        keep64 = rep == "bernstein"
        if size is None:
            out = field.sample(None)
            if isinstance(out, ngs.GridFunction):
                return out
            return self._convert(out, output, discrete, keep64)
        shape = (int(size),) if isinstance(size, numbers.Integral) else tuple(int(s) for s in size)
        n = math.prod(shape)
        vals = field.sample(n)
        if isinstance(vals, torch.Tensor):
            vals = vals.movedim(-1, 0).reshape(*shape, *vals.shape[:-1])
        else:
            vals = np.moveaxis(vals, -1, 0).reshape(*shape, *vals.shape[:-1])
        return self._convert(vals, output, discrete, keep64)

    def _convert(self, vals, output, discrete, keep64=False):
        output = output or self.output or ("torch" if self.device.type == "cuda" else "numpy")
        dtype = torch.float64 if keep64 else self.dtype
        if output == "numpy":
            if isinstance(vals, torch.Tensor):
                vals = vals.detach().cpu().numpy()
            # C order: samples are contiguous rows (internal arrays are often transposed views)
            return vals.astype(np.int64 if discrete else torch.empty(0, dtype=dtype).numpy().dtype, order="C",
                               copy=False)
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
        """Draw from the named distribution: ``rng.sample("gamma", 2.0, size=10)`` == ``rng.gamma(2.0, size=10)``."""
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
        (exactly N(loc, scale^2) at every point, loc and scale may vary in space) instead.
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
        """Pointwise map func(g) of the Gaussian point values (NumPy)."""
        return self.sample("transformed", func, size=size, **kwargs)

    def marginal(self, dist, size=None, **kwargs):
        """Gaussian-copula field with any frozen scipy.stats marginal ``dist``."""
        return self.sample("marginal", dist, size=size, **kwargs)

    def as_gridfunction(self, values, order=None) -> ngs.GridFunction:
        """A GridFunction from one sample's values: FE coefficients (length ndof) or point values (length npoints)."""
        values = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
        if values.shape[0] == self.fes.ndof:
            gf = ngs.GridFunction(self.fes)
            gf.vec.FV().NumPy()[:] = values
            return gf
        from .quadrature import QuadratureRule

        rule = QuadratureRule(self.gaussian().space, self.order if order is None else order)
        return rule.to_gridfunction(values)

    def __repr__(self):
        return f"Generator({self.fes.type}, ndof={self.fes.ndof}, device={self.device}, dtype={self.dtype})"


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
