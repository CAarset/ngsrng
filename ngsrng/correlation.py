"""Unit-variance spatial correlation structures, bound to an FE space on use.

``matern(length, nu)`` is the Whittle-Matern field of the SPDE (kappa^2 - Laplace)^(alpha/2) u = W with
alpha = nu + d/2 and kappa = sqrt(8 nu) / length (``length`` is the practical correlation range: correlation ~0.13 at
that distance). It is scaled so that the marginal variance on R^d is one (Lindgren, Rue, Lindstrom 2011):

    sigma^2 = Gamma(nu) / (Gamma(nu + d/2) (4 pi)^(d/2) kappa^(2 nu)).

On a bounded domain the operator needs a boundary condition. The natural (Neumann) condition inflates the variance
within about one correlation length of the boundary (about 2x at edges, more in corners); families that need exact
unit variance standardise pointwise. ``boundary="robin"`` uses the Robin condition du/dn + beta u = 0, which keeps the
variance close to one up to the boundary for a suitable beta. The coefficient has no default: the best value depends
on nu and on the domain (the classical beta = kappa / 1.42 of Roininen, Huttunen & Lasanen 2014, used by Daon &
Stadler 2018 and Villa & O'Leary-Roseberry 2024, suits nu = 1 in 2D; nu = 5 needs about kappa / 3.6). Pass it, or
``beta=None`` to estimate the beta whose pointwise variance stays closest to constant (:func:`ngsrng.tune_robin`).
``ngsrng.pointwise_variance`` shows the result exactly. With this normalisation, ``normal(loc, scale,
correlation=matern(...))`` has pointwise standard deviation ~scale, as in numpy.
"""

from dataclasses import dataclass, field
from math import gamma, pi, sqrt

from .specs import PDEPrecision, matern_dg
from .specs import matern as _matern_h1


class _Required:
    """Marks a boundary-condition parameter that has no default (one object, also after pickling)."""

    def __repr__(self):
        return "<required>"

    def __reduce__(self):
        return "REQUIRED"


REQUIRED = _Required()

ROBIN_REQUIRED = (
    "boundary='robin' needs the Robin coefficient beta (the condition du/dn + beta u = 0; it has no default, since the "
    "best value depends on nu and on the domain). Pass beta=<number>, or beta=None to estimate the beta that keeps the "
    "pointwise variance closest to constant (ngsrng.tune_robin: seconds per space, cached; "
    ".robin_tuning(fes) reports it)."
)


@dataclass
class MaternCorrelation:
    length: float
    nu: float = 1.0
    rtol: float = 1e-7
    boundary: str = "neumann"
    beta: float | None = REQUIRED  # Robin coefficient (boundary="robin" only): a number, or None to estimate it
    # tuned coefficients per space, kept with the correlation so that pickled copies (worker processes) reuse them
    _tuned: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.boundary not in ("neumann", "robin"):
            raise ValueError("boundary must be 'neumann' or 'robin'")
        if self.boundary == "robin":
            if self.beta is REQUIRED:
                raise ValueError(ROBIN_REQUIRED)
            if self.beta is not None and not float(self.beta) >= 0:
                raise ValueError("the Robin coefficient beta must be nonnegative")
        elif self.beta is not REQUIRED:
            raise ValueError("beta is the coefficient of the Robin condition: pass boundary='robin' as well (the "
                             "natural condition, 'neumann', has no parameter)")

    def kappa(self) -> float:
        return sqrt(8.0 * self.nu) / self.length

    def bind(self, fes) -> PDEPrecision:
        d = fes.mesh.dim
        kappa = self.kappa()
        power = self.nu + d / 2
        var = gamma(self.nu) / (gamma(self.nu + d / 2) * (4 * pi) ** (d / 2) * kappa ** (2 * self.nu))
        scale = 1.0 / sqrt(var)
        beta = self.robin_beta(fes) if self.boundary == "robin" else None
        if fes.type == "l2ho":
            spec = matern_dg(fes, kappa, power, scale, robin=beta)
        else:
            spec = _matern_h1(kappa, power, scale, robin=beta)
        spec.rtol = self.rtol
        return spec

    def robin_beta(self, fes) -> float:
        """The Robin coefficient on ``fes``: ``beta`` itself, or the estimated one for ``beta=None``."""
        if self.boundary != "robin":
            raise ValueError("the natural (Neumann) condition has no Robin coefficient")
        return self.robin_tuning(fes).beta if self.beta is None else float(self.beta)

    def robin_tuning(self, fes):
        """The :class:`ngsrng.RobinTuning` of this field's length and nu on ``fes`` (the estimate behind
        ``beta=None``; computed on first use, for any boundary choice, e.g. to compare a given beta with it)."""
        from .robin import _fingerprint, tune_robin

        key = (_fingerprint(fes), self.length, self.nu, self.rtol)
        if key not in self._tuned:
            self._tuned[key] = tune_robin(fes, self.length, self.nu, rtol=self.rtol)
        return self._tuned[key]


def matern(length: float, nu: float = 1.0, rtol: float = 1e-7, boundary: str = "neumann", beta=REQUIRED, *,
           robin=REQUIRED) -> MaternCorrelation:
    """Unit-variance Matern correlation with range ``length`` and smoothness ``nu`` (samples are C^(nu-epsilon)).

    ``boundary``: "neumann" (the natural condition; variance inflated near the boundary) or "robin", with the Robin
    coefficient ``beta`` (du/dn + beta u = 0), which has no default: a number, or ``None`` to estimate the beta that
    keeps the pointwise variance closest to constant (:func:`ngsrng.tune_robin`: seconds per space, cached;
    ``.robin_tuning(fes)`` reports the estimate and the remaining deviation).
    """
    if robin is not REQUIRED:
        raise TypeError("robin= (a factor of kappa) was replaced by beta=, the Robin coefficient itself (beta = factor "
                        "* kappa with kappa = sqrt(8 nu) / length); beta=None estimates it for the space")
    if length <= 0 or nu <= 0:
        raise ValueError("length and nu must be positive")
    return MaternCorrelation(length, nu, rtol, boundary, beta)


def bind(spec, fes):
    """Resolve a correlation/covariance spec against a space (specs without ``bind`` are returned unchanged)."""
    return spec.bind(fes) if hasattr(spec, "bind") else spec
