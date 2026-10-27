"""Fields with prescribed pointwise marginals: Gaussian-copula ("translation") transforms of a standardised field.

f(x) = F^-1(Phi(z(x))), z = (g - mean) / std, so the marginal at every point is exactly F (with exact
standardisation; mean and std are the prior's when g is a posterior). Values of bounded families stay strictly inside
the support in the output precision. The spatial dependence is the Gaussian copula of g. Quantile maps are
closed-form where possible (``CLOSED``, any device, any spatially varying parameters) and otherwise numerical
inversion of a scipy distribution (``SCIPY``; method chosen automatically, see :mod:`.inversion`). Parameter names
and conventions follow ``numpy.random.Generator``.
"""

import numpy as np
import scipy.stats as st

from .inversion import Inversion
from .pointwise import XP, GaussianAtPoints, PointwiseField, is_conditioned, warn_copies_of_posterior


def _cauchy(z, xp, loc=0.0, scale=1.0):
    # tan(pi (u - 1/2)) written as -cot(pi Phi(z)) / cot(pi Phi(-z)) to keep both tails accurate
    lower = -1.0 / xp.m.tan(np.pi * xp.ndtr(z))
    upper = 1.0 / xp.m.tan(np.pi * xp.ndtr(-z))
    return loc + scale * xp.where(z < 0, lower, upper)


def _triangular(z, xp, left, mode, right):
    u, s = xp.ndtr(z), xp.ndtr(-z)
    width = right - left
    c = (mode - left) / width
    return xp.where(u < c, left + xp.m.sqrt(u * width * (mode - left)), right - xp.m.sqrt(s * width * (right - mode)))


INF = np.inf

# name -> (quantile map of z, parameters with defaults (None = required), support(**parameters) -> (lo, hi)).
# Upper tails use Phi(-z) (log_ndtr(-z)) so that they keep full relative accuracy.
CLOSED = {
    "normal": (lambda z, xp, loc, scale: loc + scale * z, {"loc": 0.0, "scale": 1.0}, lambda **p: (-INF, INF)),
    "uniform": (
        lambda z, xp, low, high: low + (high - low) * xp.ndtr(z),
        {"low": 0.0, "high": 1.0},
        lambda low, high: (low, high),
    ),
    "lognormal": (
        lambda z, xp, mean, sigma: xp.m.exp(mean + sigma * z),
        {"mean": 0.0, "sigma": 1.0},
        lambda **p: (0.0, INF),
    ),
    "exponential": (lambda z, xp, scale: -scale * xp.log_ndtr(-z), {"scale": 1.0}, lambda **p: (0.0, INF)),
    "weibull": (lambda z, xp, a: (-xp.log_ndtr(-z)) ** (1.0 / a), {"a": None}, lambda **p: (0.0, INF)),
    "laplace": (
        lambda z, xp, loc, scale: xp.where(
            z < 0, loc + scale * (np.log(2.0) + xp.log_ndtr(z)), loc - scale * (np.log(2.0) + xp.log_ndtr(-z))
        ),
        {"loc": 0.0, "scale": 1.0},
        lambda **p: (-INF, INF),
    ),
    "gumbel": (
        lambda z, xp, loc, scale: loc - scale * xp.m.log(-xp.log_ndtr(z)),
        {"loc": 0.0, "scale": 1.0},
        lambda **p: (-INF, INF),
    ),
    "logistic": (
        lambda z, xp, loc, scale: loc + scale * (xp.log_ndtr(z) - xp.log_ndtr(-z)),
        {"loc": 0.0, "scale": 1.0},
        lambda **p: (-INF, INF),
    ),
    "rayleigh": (
        lambda z, xp, scale: scale * xp.m.sqrt(-2.0 * xp.log_ndtr(-z)),
        {"scale": 1.0},
        lambda **p: (0.0, INF),
    ),
    "pareto": (  # numpy's pareto is the Lomax distribution
        lambda z, xp, a: xp.m.expm1(-xp.log_ndtr(-z) / a),
        {"a": None},
        lambda **p: (0.0, INF),
    ),
    "power": (lambda z, xp, a: xp.m.exp(xp.log_ndtr(z) / a), {"a": None}, lambda **p: (0.0, 1.0)),
    "standard_cauchy": (_cauchy, {}, lambda **p: (-INF, INF)),
    "triangular": (
        _triangular,
        {"left": None, "mode": None, "right": None},
        lambda left, mode, right: (left, right),
    ),
    "halfnormal": (
        lambda z, xp, scale: -scale * xp.ndtri(0.5 * xp.ndtr(-z)),
        {"scale": 1.0},
        lambda **p: (0.0, INF),
    ),
}


def _hypergeom(ngood, nbad, nsample):
    return st.hypergeom(ngood + nbad, ngood, nsample), 0.0, 1.0


# name -> (builder(**params) -> (frozen scipy distribution, loc, scale), parameters with defaults)
SCIPY = {
    "gamma": (lambda shape, scale: (st.gamma(shape), 0.0, scale), {"shape": None, "scale": 1.0}),
    "beta": (lambda a, b: (st.beta(a, b), 0.0, 1.0), {"a": None, "b": None}),
    "chisquare": (lambda df: (st.chi2(df), 0.0, 1.0), {"df": None}),
    "noncentral_chisquare": (lambda df, nonc: (st.ncx2(df, nonc), 0.0, 1.0), {"df": None, "nonc": None}),
    "standard_t": (lambda df: (st.t(df), 0.0, 1.0), {"df": None}),
    "f": (lambda dfnum, dfden: (st.f(dfnum, dfden), 0.0, 1.0), {"dfnum": None, "dfden": None}),
    "noncentral_f": (
        lambda dfnum, dfden, nonc: (st.ncf(dfnum, dfden, nonc), 0.0, 1.0),
        {"dfnum": None, "dfden": None, "nonc": None},
    ),
    "wald": (lambda mean, scale: (st.invgauss(mean / scale), 0.0, scale), {"mean": None, "scale": None}),
    "vonmises": (lambda mu, kappa: (st.vonmises(kappa), mu, 1.0), {"mu": None, "kappa": None}),
    "poisson": (lambda lam: (st.poisson(lam), 0.0, 1.0), {"lam": 1.0}),
    "binomial": (lambda n, p: (st.binom(n, p), 0.0, 1.0), {"n": None, "p": None}),
    "bernoulli": (lambda p: (st.bernoulli(p), 0.0, 1.0), {"p": None}),
    "negative_binomial": (lambda n, p: (st.nbinom(n, p), 0.0, 1.0), {"n": None, "p": None}),
    "geometric": (lambda p: (st.geom(p), 0.0, 1.0), {"p": None}),
    "logseries": (lambda p: (st.logser(p), 0.0, 1.0), {"p": None}),
    "zipf": (lambda a: (st.zipf(a), 0.0, 1.0), {"a": None}),
    "hypergeometric": (_hypergeom, {"ngood": None, "nbad": None, "nsample": None}),
}


def _bind(defaults, args, kwargs, name):
    names = list(defaults)
    if len(args) > len(names):
        raise TypeError(f"{name}() takes at most {len(names)} distribution parameters")
    params = dict(zip(names, args))
    for k, v in kwargs.items():
        if k not in defaults:
            raise TypeError(f"{name}() got an unexpected parameter {k!r}")
        if k in params:
            raise TypeError(f"{name}() got multiple values for {k!r}")
        params[k] = v
    for k, v in defaults.items():
        if k not in params:
            if v is None:
                raise TypeError(f"{name}() missing parameter {k!r}")
            params[k] = v
    return params


class ClosedForm:
    """Closed-form pointwise map func(z, xp, **params) with (possibly spatially varying) parameters; ``support`` is
    the (lo, hi) range of the values (None: not a continuous family, no clamping)."""

    torch_capable = True

    def __init__(self, func, params, info=None, support=None, closed_range=None):
        self.func, self.params, self.info = func, params, {"inversion": "closed"} if info is None else info
        self.support, self.closed_range = support, closed_range

    def __call__(self, z, xp):
        return self.func(z, xp, **{k: xp.const(v, z) for k, v in self.params.items()})


class MarginalField(PointwiseField):
    """f = transform(z, xp) for standardised point values z (``standardize=True``) or raw values g."""

    def __init__(self, points: GaussianAtPoints, transform, *, standardize=True, representation="quadrature",
                 label="marginal"):
        super().__init__(points, representation)
        self.transform, self.standardize, self.label = transform, standardize, label
        self.torch_capable = getattr(transform, "torch_capable", False)

    @property
    def info(self) -> dict:
        """Law and algorithms (the inversion method of "auto" is resolved on the first draw)."""
        out = {"law": f"Gaussian copula, {self.label} marginals", **getattr(self.transform, "info", {})}
        if getattr(self.transform, "discrete", False):
            out["discrete"] = True
        return {**out, **self._common_info()}

    def _values(self, n):
        dev = self.on_device
        z = self.points.standardized(n, dev) if self.standardize else self.points.draw(n, dev)
        xp = XP(dev)
        return self.clamp_open(self.transform(z, xp), getattr(self.transform, "support", None), xp)

    def _pull_inside(self, values, margin):
        # open supports and attained (closed) ranges alike: rounding in the conversion must not cross the bounds
        support = getattr(self.transform, "support", None) or getattr(self.transform, "closed_range", None)
        if support is None:  # e.g. transformed(func): range unknown, no guarantee
            return values
        lo, hi = (np.asarray(b, dtype=float) for b in support)
        out = np.where(np.isfinite(lo), np.maximum(values, lo + margin), values)  # bounds may vary per point
        return np.where(np.isfinite(hi), np.minimum(out, hi - margin), out)


def named_marginal(points, name, *args, method="auto", tol=1e-10, representation="quadrature", **kwargs):
    """Copula field with the named marginal (numpy names and parameters). Parameters may vary in space
    (CoefficientFunctions, GridFunctions or point-value arrays)."""
    if name in CLOSED and method in ("auto", "closed"):
        func, defaults, support = CLOSED[name]
        params = {k: points.parameter(v) for k, v in _bind(defaults, args, kwargs, name).items()}
        transform = ClosedForm(func, params, support=support(**params))
    elif name in SCIPY:
        params = _bind(SCIPY[name][1], args, kwargs, name)
        params = {k: points.parameter(v) for k, v in params.items()}
        dist, loc, scale = SCIPY[name][0](**params)
        transform = Inversion(dist, loc, scale, method=method, tol=tol)
    elif name in CLOSED:
        raise ValueError(f"{name} has a closed-form quantile; method must be 'auto' or 'closed'")
    else:
        raise ValueError(f"unknown marginal {name!r}")
    return MarginalField(points, transform, representation=representation, label=name)


def marginal(points, dist, method="auto", tol=1e-10, representation="quadrature"):
    """Any frozen scipy.stats distribution (continuous or discrete) as pointwise marginal: F^-1(Phi(z))."""
    label = getattr(getattr(dist, "dist", None), "name", type(dist).__name__)
    return MarginalField(points, Inversion(dist, method=method, tol=tol), representation=representation, label=label)


def indicator(points, level=0.0, standardize=True, representation="quadrature"):
    """Excursion indicator 1{g > level} (``standardize=True``: level in units of the pointwise standard deviation)."""
    transform = ClosedForm(lambda g, xp, level: (g > level) * 1.0, {"level": points.parameter(level)}, {},
                           closed_range=(0.0, 1.0))
    return MarginalField(points, transform, standardize=standardize, representation=representation,
                         label="indicator")


def transformed(points, func, standardize=False, representation="quadrature"):
    """General pointwise map f = func(g) of NumPy point values (``standardize=True``: of z)."""
    return MarginalField(points, lambda x, xp: func(x), standardize=standardize, representation=representation,
                         label="transformed")


class DirichletField(PointwiseField):
    """Simplex-valued field: x_k = G_k / sum_j G_j with independent Gamma(alpha_k) copula fields G_k, so that
    x(point) ~ Dirichlet(alpha) exactly. Samples have shape (npoints, K, n).

    Gamma quantiles via :class:`.inversion.Inversion` (tables, so it runs on the GPU too). Very small alpha_k
    (below ~0.02) make lower-tail gamma quantiles underflow to zero, which is not handled.
    """

    def __init__(self, points, alpha, method="auto", tol=1e-10, representation="quadrature"):
        super().__init__(points, representation)
        self.alpha = np.asarray(alpha, dtype=float)
        if self.alpha.ndim != 1 or (self.alpha <= 0).any():
            raise ValueError("alpha must be a vector of positive numbers")
        if is_conditioned(points.gauss):
            warn_copies_of_posterior()
        cache = {}
        self.inv = [cache.setdefault(a, Inversion(st.gamma(a), method=method, tol=tol)) for a in self.alpha]
        self.torch_capable = all(i.torch_capable for i in self.inv)

    @property
    def info(self) -> dict:
        return {"law": "Dirichlet (normalised Gamma copula fields)", **self.inv[0].info, **self._common_info()}

    def _pull_inside(self, values, margin):
        k = len(self.alpha)
        m = np.minimum(margin, 0.5 / k)
        return (1 - k * m) * values + m  # every component >= m, components still sum to one

    def _values(self, n):
        dev = self.on_device
        k = len(self.alpha)
        z = self.points.standardized(n * k, dev).reshape(self.points.npoints, k, n)
        xp = XP(dev)
        g = xp.m.stack([self.inv[j](z[:, j, :], xp) for j in range(k)], 1)
        return g / g.sum(1, keepdims=True) if not xp.torch else g / g.sum(1, keepdim=True)
