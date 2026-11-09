"""Unit-variance spatial correlation structures, bound to an FE space on use.

``matern(length, nu)`` is the Whittle-Matern field of the SPDE (kappa^2 - Laplace)^(alpha/2) u = W with
alpha = nu + d/2 and kappa = sqrt(8 nu) / length (``length`` is the practical correlation range: correlation ~0.13 at
that distance). It is scaled so that the marginal variance on R^d is one (Lindgren, Rue, Lindstrom 2011):

    sigma^2 = Gamma(nu) / (Gamma(nu + d/2) (4 pi)^(d/2) kappa^(2 nu)).

On bounded domains the natural (Neumann) boundary condition inflates the variance within about one correlation
length of the boundary (about 2x at edges, more in corners); families that need exact unit variance standardise
pointwise. ``boundary="robin"`` uses the Robin condition du/dn + beta u = 0 with beta = robin * kappa (default
robin = 1/1.42: Roininen, Huttunen & Lasanen 2014; Daon & Stadler 2018; Villa & O'Leary-Roseberry 2024), which keeps
the variance close to one up to the boundary for nu = 1 in 2D; ``ngsrng.pointwise_variance`` shows the result exactly.
The best factor depends on nu (about 1/4 for nu = 5) and on the domain: ``robin="tune"`` chooses it for the space
(:func:`ngsrng.tune_robin`).
With this normalisation, ``normal(loc, scale, correlation=matern(...))`` has pointwise standard deviation ~scale, as in
numpy.
"""

from dataclasses import dataclass, field
from math import gamma, pi, sqrt

from .specs import ROBIN_FACTOR, PDEPrecision, matern_dg
from .specs import matern as _matern_h1


@dataclass
class MaternCorrelation:
    length: float
    nu: float = 1.0
    rtol: float = 1e-7
    boundary: str = "neumann"
    robin: float | str = ROBIN_FACTOR
    # tuned factors per space ("tune"), kept with the correlation so that pickled copies (worker processes) reuse them
    _tuned: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.boundary not in ("neumann", "robin"):
            raise ValueError("boundary must be 'neumann' or 'robin'")
        if isinstance(self.robin, str):
            if self.robin != "tune":
                raise ValueError("robin must be a nonnegative factor or 'tune'")
            if self.boundary != "robin":
                raise ValueError("robin='tune' tunes the Robin condition: pass boundary='robin' as well")
        elif self.robin < 0:
            raise ValueError("the Robin factor must be nonnegative")

    def kappa(self) -> float:
        return sqrt(8.0 * self.nu) / self.length

    def bind(self, fes) -> PDEPrecision:
        d = fes.mesh.dim
        kappa = self.kappa()
        power = self.nu + d / 2
        var = gamma(self.nu) / (gamma(self.nu + d / 2) * (4 * pi) ** (d / 2) * kappa ** (2 * self.nu))
        scale = 1.0 / sqrt(var)
        beta = self.robin_factor(fes) * kappa if self.boundary == "robin" else None
        if fes.type == "l2ho":
            spec = matern_dg(fes, kappa, power, scale, robin=beta)
        else:
            spec = _matern_h1(kappa, power, scale, robin=beta)
        spec.rtol = self.rtol
        return spec

    def robin_factor(self, fes) -> float:
        """The factor c of beta = c kappa on ``fes``: ``robin`` itself, or the tuned one for ``robin="tune"``."""
        if not isinstance(self.robin, str):
            return float(self.robin)
        return self.robin_tuning(fes).factor

    def robin_tuning(self, fes):
        """The :class:`ngsrng.robin.RobinTuning` behind ``robin="tune"`` on ``fes`` (tuned on first use)."""
        from .robin import _fingerprint, tune_robin

        key = (_fingerprint(fes), self.length, self.nu, self.rtol)
        if key not in self._tuned:
            self._tuned[key] = tune_robin(fes, self.length, self.nu, rtol=self.rtol)
        return self._tuned[key]


def matern(length: float, nu: float = 1.0, rtol: float = 1e-7, boundary: str = "neumann",
           robin: float | str = ROBIN_FACTOR) -> MaternCorrelation:
    """Unit-variance Matern correlation with range ``length`` and smoothness ``nu`` (samples are C^(nu-epsilon)).

    ``boundary``: "neumann" (natural condition; variance inflated near the boundary) or "robin" (Robin coefficient
    ``robin`` * kappa; the default factor 1/1.42 keeps the variance nearly constant for nu = 1 in 2D, but pulls it
    down near the boundary for larger nu). ``robin="tune"`` chooses the factor for each space it is bound to
    (:func:`ngsrng.tune_robin`: seconds per domain, cached; ``.robin_tuning(fes)`` reports the result).
    """
    if length <= 0 or nu <= 0:
        raise ValueError("length and nu must be positive")
    return MaternCorrelation(length, nu, rtol, boundary, robin)


def bind(spec, fes):
    """Resolve a correlation/covariance spec against a space (specs without ``bind`` are returned unchanged)."""
    return spec.bind(fes) if hasattr(spec, "bind") else spec
