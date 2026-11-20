"""Random parameters of a law: a correlation length or smoothness drawn anew for every draw of a field.

``ngsrng.matern_correlation(length=law, nu=law)`` takes, for either parameter, a number or a law of it: a
``scipy.stats`` distribution (frozen, or ``scipy.stats.rv_discrete(values=(x, p))``). The field's law is then the
mixture over the parameter's law (:mod:`ngsrng.mixture`):

- a law with finitely many values (``rv_discrete(values=...)``, a frozen discrete law with finite support): the finite
  mixture of the field at each value, with the values' probabilities as weights; every law of a field is available
  (conditioning, ``bnd=``, any construction);
- a continuous law on a bounded interval [a, b] with 0 < a < b < inf (``scipy.stats.loguniform(a, b)``,
  ``uniform(a, b - a)``, a truncated law, ...): the continuous mixture, drawn exactly through one spectral basis
  (:mod:`ngsrng.spectral`) for the constructions whose covariance is a function of one fixed operator.

The same law results from ``ngsrng.mixture(lambda value: <the field at value>, over=law)``.
"""

import math
import numbers

import numpy as np
import scipy.stats as st

__all__ = ["RandomParameter", "as_parameter", "is_random"]


class RandomParameter:
    """The law of a parameter: ``values`` and ``weights`` (finite), or a continuous ``scipy.stats`` law on a bounded
    interval. ``ppf(u)`` maps uniforms to values; ``support`` is the interval (or the smallest and largest value)."""

    def __init__(self, law=None, values=None, weights=None):
        self.law, self.values, self.weights = None, None, None
        if values is not None:
            v = np.asarray(values, dtype=float).reshape(-1)
            w = np.full(v.size, 1.0 / v.size) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
            if v.size == 0 or w.shape != v.shape or (w < 0).any() or w.sum() <= 0 or not np.isfinite(v).all():
                raise ValueError("a finite parameter law needs finite values and as many nonnegative weights")
            keep = w > 0
            self.values, self.weights = v[keep], w[keep] / w[keep].sum()
            return
        finite = _finite_support(law)
        if finite is not None:
            self.values, self.weights = finite
            return
        if not _is_scipy_law(law):
            raise TypeError(f"a random parameter is a scipy.stats distribution, not {type(law).__name__}")
        a, b = (float(x) for x in law.support())
        if not (math.isfinite(a) and math.isfinite(b) and a < b):
            raise ValueError(f"a continuous parameter law needs a bounded support [a, b]; this one has [{a}, {b}] "
                             "(use e.g. scipy.stats.loguniform(a, b), uniform(a, b - a) or a truncated law)")
        self.law = law

    @property
    def finite(self) -> bool:
        return self.values is not None

    @property
    def support(self) -> tuple:
        if self.finite:
            return float(self.values.min()), float(self.values.max())
        return tuple(float(x) for x in self.law.support())

    def ppf(self, u) -> np.ndarray:
        """Values from uniforms u in [0, 1) (inverse CDF); finite laws: the value whose cumulative weight bracket
        holds u."""
        u = np.asarray(u, dtype=float)
        if self.finite:
            return self.values[np.searchsorted(np.cumsum(self.weights)[:-1], u, side="right")]
        a, b = self.support
        return np.clip(self.law.ppf(u), a, b)

    def describe(self) -> str:
        if self.finite:
            return f"{self.values.size} values in [{self.values.min():.4g}, {self.values.max():.4g}]"
        args = ", ".join(f"{float(a):.6g}" for a in self.law.args)
        return f"{_name(self.law)}({args}) on [{self.support[0]:.4g}, {self.support[1]:.4g}]"

    def store(self) -> dict:
        """A description without code (JSON): the scipy.stats distribution's name and parameters, or the values and
        weights."""
        if self.finite:
            return {"values": self.values.tolist(), "weights": self.weights.tolist()}
        name = _name(self.law)
        if type(getattr(st, name, None)) is not type(self.law.dist):
            raise NotImplementedError(f"the parameter law {name} is not a scipy.stats distribution by name")
        return {"scipy": name, "args": [float(a) for a in self.law.args],
                "kwds": {k: float(v) for k, v in self.law.kwds.items()}}

    @classmethod
    def load(cls, d) -> "RandomParameter":
        if "values" in d:
            return cls(values=d["values"], weights=d["weights"])
        return cls(getattr(st, d["scipy"])(*d["args"], **d["kwds"]))

    def __repr__(self):
        return f"RandomParameter({self.describe()})"

    def __eq__(self, other):
        return isinstance(other, RandomParameter) and self.store() == other.store()

    def __hash__(self):
        return hash(repr(sorted(self.store().items())))


def is_random(x) -> bool:
    """Whether a parameter is given as a law (rather than a number)."""
    return isinstance(x, RandomParameter) or _is_scipy_law(x) or _finite_support(x) is not None


def as_parameter(x):
    """A number as a float, a law as a :class:`RandomParameter`."""
    if isinstance(x, RandomParameter):
        return x
    if isinstance(x, numbers.Real):
        return float(x)
    return RandomParameter(x)


def _is_scipy_law(x) -> bool:
    return isinstance(x, st._distn_infrastructure.rv_frozen)


def _name(law) -> str:
    return getattr(getattr(law, "dist", None), "name", type(law).__name__)


def _finite_support(law):
    """(values, probabilities) of a law with finitely many values, or None."""
    if isinstance(law, st.rv_discrete) and hasattr(law, "xk"):  # rv_discrete(values=...)
        return np.asarray(law.xk, dtype=float), np.asarray(law.pk, dtype=float)
    if _is_scipy_law(law) and isinstance(law.dist, st.rv_discrete):
        if hasattr(law.dist, "xk"):
            return np.asarray(law.dist.xk, dtype=float), np.asarray(law.dist.pk, dtype=float)
        a, b = law.support()
        if math.isfinite(a) and math.isfinite(b) and b - a <= 10**6:
            x = np.arange(a, b + 1)
            p = law.pmf(x)
            return x[p > 0], p[p > 0] / p[p > 0].sum()
    return None
