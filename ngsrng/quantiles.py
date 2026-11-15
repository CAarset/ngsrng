"""Closed-form quantile maps z -> F^-1(Phi(z)) of standard normal values, written once for NumPy and torch.

The maps of ``CLOSED`` are the pointwise transforms of the copula families with a closed-form quantile (``uniform``,
``lognormal``, ``exponential``, ...; parameter names and conventions follow ``numpy.random.Generator``). This module
needs NumPy, SciPy and torch only, not NGSolve: stored dense samplers (:mod:`ngsrng.dense`) evaluate them without it.
"""

import numpy as np
import scipy.special as sps
import torch

_SQRT1_2 = float(np.sqrt(0.5))


class XP:
    """Minimal array-module switch so each transform is written once for NumPy (CPU) and torch (GPU)."""

    def __init__(self, torch_mode: bool):
        self.torch = torch_mode
        self.m = torch if torch_mode else np

    def ndtr(self, x):
        if not self.torch:
            return sps.ndtr(x)
        # torch.special.ndtr computes (1 + erf) / 2, which cancels in the lower tail (4% relative error at x = -8,
        # zero below -10); the erfc form keeps full relative accuracy in both tails
        tail = 0.5 * torch.special.erfc(x.abs() * _SQRT1_2)
        return torch.where(x < 0, tail, 1.0 - tail)

    def log_ndtr(self, x):
        return torch.special.log_ndtr(x) if self.torch else sps.log_ndtr(x)

    def ndtri(self, x):
        return torch.special.ndtri(x) if self.torch else sps.ndtri(x)

    def where(self, c, a, b):
        return torch.where(c, a, b) if self.torch else np.where(c, a, b)


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


def _normal(z, xp, loc, scale):
    return loc + scale * z


def _uniform(z, xp, low, high):
    return low + (high - low) * xp.ndtr(z)


def _lognormal(z, xp, mean, sigma):
    return xp.m.exp(mean + sigma * z)


def _exponential(z, xp, scale):
    return -scale * xp.log_ndtr(-z)


def _weibull(z, xp, a):
    return (-xp.log_ndtr(-z)) ** (1.0 / a)


def _laplace(z, xp, loc, scale):
    return xp.where(z < 0, loc + scale * (np.log(2.0) + xp.log_ndtr(z)), loc - scale * (np.log(2.0) + xp.log_ndtr(-z)))


def _gumbel(z, xp, loc, scale):
    return loc - scale * xp.m.log(-xp.log_ndtr(z))


def _logistic(z, xp, loc, scale):
    return loc + scale * (xp.log_ndtr(z) - xp.log_ndtr(-z))


def _rayleigh(z, xp, scale):
    return scale * xp.m.sqrt(-2.0 * xp.log_ndtr(-z))


def _lomax(z, xp, a):  # numpy's pareto is the Lomax distribution
    return xp.m.expm1(-xp.log_ndtr(-z) / a)


def _power(z, xp, a):
    return xp.m.exp(xp.log_ndtr(z) / a)


def _halfnormal(z, xp, scale):
    return -scale * xp.ndtri(0.5 * xp.ndtr(-z))


def _exceeds(g, xp, level):
    """The excursion indicator 1{g > level} (``indicator``: a step, not a quantile map)."""
    return (g > level) * 1.0


def _between_low_high(low, high):
    return low, high


def _between_left_right(left, mode, right):
    return left, right


REAL, POSITIVE, UNIT = (-INF, INF), (0.0, INF), (0.0, 1.0)

# name -> (quantile map of z, parameters with defaults (None = required), support: (lo, hi), or a function of the
# parameters returning it). Upper tails use Phi(-z) (log_ndtr(-z)) so that they keep full relative accuracy. Named
# functions, not lambdas: fields store the map, and stored functions must pickle.
CLOSED = {
    "normal": (_normal, {"loc": 0.0, "scale": 1.0}, REAL),
    "uniform": (_uniform, {"low": 0.0, "high": 1.0}, _between_low_high),
    "lognormal": (_lognormal, {"mean": 0.0, "sigma": 1.0}, POSITIVE),
    "exponential": (_exponential, {"scale": 1.0}, POSITIVE),
    "weibull": (_weibull, {"a": None}, POSITIVE),
    "laplace": (_laplace, {"loc": 0.0, "scale": 1.0}, REAL),
    "gumbel": (_gumbel, {"loc": 0.0, "scale": 1.0}, REAL),
    "logistic": (_logistic, {"loc": 0.0, "scale": 1.0}, REAL),
    "rayleigh": (_rayleigh, {"scale": 1.0}, POSITIVE),
    "pareto": (_lomax, {"a": None}, POSITIVE),
    "power": (_power, {"a": None}, UNIT),
    "standard_cauchy": (_cauchy, {}, REAL),
    "triangular": (_triangular, {"left": None, "mode": None, "right": None}, _between_left_right),
    "halfnormal": (_halfnormal, {"scale": 1.0}, POSITIVE),
}

# every pointwise map stored by name (dense samplers save the name, the parameters and the range)
MAPS = {name: entry[0] for name, entry in CLOSED.items()} | {"indicator": _exceeds}
NAMES = {func: name for name, func in MAPS.items()}
