"""Per-domain tuning of the Robin coefficient of Matern priors: ``matern_correlation(..., robin="tune")``.

The Robin condition du/dn + beta u = 0 with beta = c kappa keeps the pointwise variance of a Matern field near its
interior value up to the boundary, where the natural (Neumann) condition doubles it (more in corners). The best
constant c depends on the smoothness: 1/1.42 is the classical value for nu = 1 in 2D, while for nu = 5 it pulls the
variance down to 0.1 in corners and c ~ 1/4 does better. It also depends on the domain (corners, curvature, thin
parts, the length relative to the domain). :func:`tune_robin` chooses c for a given FE space: it minimises the
root-mean-square of log(variance) over the boundary layer (the points within one correlation length of the
boundary, area-weighted), with the variances of the discretised sampler computed exactly (sigma^2(x) = |L^T g_x|^2).

A constant c cannot remove every boundary effect: for large nu the variance still dips in corners and overshoots
slightly further in. The result records the remaining deviation, and that of the classical factor for comparison.
"""

import math
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import blake2b

import ngsolve as ngs
import numpy as np

from .specs import ROBIN_FACTOR

PROBES = 300  # points of the boundary layer at which the variance is evaluated (one application of L^T each)
BOUNDS = (0.01, 10.0)  # search interval for c
XTOL = 0.02  # tolerance on log c (c to about 2%)
_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 32


@dataclass(frozen=True)
class RobinTuning:
    """Result of :func:`tune_robin`: the Robin coefficient is beta = ``factor`` * kappa.

    ``rms`` is the area-weighted root-mean-square of log(variance) over the boundary layer with this factor (0 for a
    variance of exactly 1 throughout; 0.1 is about a 10% typical deviation), ``variance_range`` the smallest and
    largest variance at the probe points. ``default_rms`` and ``default_range`` are the same for the classical factor
    1/1.42.
    """

    factor: float
    kappa: float
    rms: float
    variance_range: tuple
    default_rms: float
    default_range: tuple
    probes: int
    evaluations: int
    seconds: float

    @property
    def beta(self) -> float:
        return self.factor * self.kappa

    def __str__(self):
        lo, hi = self.variance_range
        dlo, dhi = self.default_range
        return (f"Robin factor {self.factor:.4g} (beta = {self.beta:.4g}): boundary-layer variance in [{lo:.3f}, "
                f"{hi:.3f}], rms log-deviation {self.rms:.3f} (1/1.42: [{dlo:.3f}, {dhi:.3f}], "
                f"{self.default_rms:.3f}); {self.probes} probes, {self.evaluations} set-ups, {self.seconds:.1f} s")


def tune_robin(fes, length: float, nu: float = 1.0, *, rtol: float = 1e-7, probes: int = PROBES,
               bounds: tuple = BOUNDS, xtol: float = XTOL) -> RobinTuning:
    """The constant c (beta = c kappa) for which ``matern_correlation(length, nu, boundary="robin", robin=c)`` on
    ``fes`` has its pointwise variance closest to one near the boundary.

    Criterion: the area-weighted mean of log(variance)^2 over the mesh vertices within ``length`` of the boundary
    (a deterministic subsample of ``probes`` of them on fine meshes: half on the boundary itself, where the deviation
    is largest, half inside the layer), over boundaries without a Dirichlet condition (points within ``length`` of a
    Dirichlet boundary are left out: their variance is pinned down whatever c is). Variances are exact for the
    discretised sampler, so the result accounts for the mesh, the element order and the rational approximation of
    fractional powers.

    Cost: about 15 set-ups of the Gaussian and ``probes`` applications of L^T each (seconds for 1e4 dofs in 2D). The
    best c does not depend on the mesh once the mesh resolves ``length``: tune on a coarser mesh of the same geometry
    and pass ``robin=result.factor`` to save time. Results are cached per space and parameters, and
    ``matern_correlation(..., robin="tune")`` uses this function.
    """
    from .correlation import MaternCorrelation

    if length <= 0 or nu <= 0:
        raise ValueError("length and nu must be positive")
    lo, hi = (float(b) for b in bounds)
    if not 0 < lo < hi:
        raise ValueError("bounds must satisfy 0 < low < high")
    key = (_fingerprint(fes), float(length), float(nu), float(rtol), int(probes), lo, hi, float(xtol))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    start = time.perf_counter()
    points, weights = _layer_probes(fes, float(length), int(probes))
    variances = _Variances(fes, points, float(length), float(nu), float(rtol))
    weights = weights / weights.sum()

    def rms(v):
        return float(np.sqrt(weights @ np.log(v) ** 2))

    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda t: rms(variances(math.exp(t))), bounds=(math.log(lo), math.log(hi)),
                          method="bounded", options={"xatol": float(xtol)})
    factor = float(math.exp(res.x))
    if min(res.x - math.log(lo), math.log(hi) - res.x) < 2 * xtol:
        warnings.warn(f"the tuned Robin factor {factor:.3g} lies at the end of the search interval {bounds}",
                      RuntimeWarning, stacklevel=2)
    v, vd = variances(factor), variances(ROBIN_FACTOR)
    out = RobinTuning(factor=factor, kappa=MaternCorrelation(length, nu).kappa(), rms=rms(v),
                      variance_range=(float(v.min()), float(v.max())), default_rms=rms(vd),
                      default_range=(float(vd.min()), float(vd.max())), probes=len(points),
                      evaluations=variances.evaluations, seconds=time.perf_counter() - start)
    _CACHE[key] = out
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return out


class _Variances:
    """c -> exact pointwise variances at the probes of the Matern Gaussian with Robin factor c (memoised)."""

    def __init__(self, fes, points, length, nu, rtol):
        from .observations import observation_operator
        from .space import SpaceInfo

        self.fes, self.length, self.nu, self.rtol = fes, length, nu, rtol
        self.gt = observation_operator(fes, points=points)[:, SpaceInfo(fes).free].T.tocsc()
        self.known = {}
        self.evaluations = 0

    def __call__(self, c):
        import torch

        from .correlation import MaternCorrelation
        from .covariance import block_columns
        from .gaussian import Gaussian

        if c in self.known:
            return self.known[c]
        spec = MaternCorrelation(self.length, self.nu, self.rtol, "robin", c).bind(self.fes)
        gauss = Gaussian(self.fes, spec, dtype="float64", seed=0)
        m = self.gt.shape[1]
        out = np.empty(m)
        step = block_columns(gauss.nrand, gauss.space.nfree)
        for i in range(0, m, step):
            y = gauss.apply_t(torch.from_numpy(self.gt[:, i:i + step].toarray()))
            out[i:i + step] = (y.double() ** 2).sum(0).numpy()
        if not np.all(out > 0):
            raise RuntimeError("a probe point has zero variance (a Dirichlet dof?): cannot tune the Robin factor")
        self.evaluations += 1
        self.known[c] = out
        return out


def _layer_probes(fes, length, probes):
    """Mesh vertices within ``length`` of the non-Dirichlet boundary (and not within ``length`` of a Dirichlet one),
    with their lumped (P1) area weights; above ``probes`` of them, a weighted subsample in two strata (boundary
    vertices, the rest) with weights adjusted so that the weighted mean stays an estimate of the area mean."""
    from scipy.spatial import cKDTree

    mesh = fes.mesh
    xyz = np.asarray(mesh.ngmesh.Coordinates(), dtype=float).reshape(mesh.nv, -1)[:, :mesh.dim]
    free = np.array(fes.FreeDofs(), dtype=bool)
    robin, dirichlet, on = [], [], np.zeros(mesh.nv, bool)
    for el in mesh.Elements(ngs.BND):
        ids = [v.nr for v in el.vertices]
        dofs = [d for d in fes.GetDofNrs(el) if d >= 0]
        vs = xyz[ids]  # sample points: vertices, centroid, edge midpoints (a dense enough boundary for distances)
        pts = np.vstack([vs, vs.mean(0, keepdims=True), 0.5 * (vs + np.roll(vs, 1, axis=0))])
        if dofs and not free[dofs].any():
            dirichlet.append(pts)
        else:
            robin.append(pts)
            on[ids] = True
    if not robin:
        raise ValueError("every boundary of the space carries a Dirichlet condition: there is no Robin coefficient "
                         "to tune")
    near = cKDTree(np.vstack(robin)).query(xyz)[0] < length
    if dirichlet:
        near &= cKDTree(np.vstack(dirichlet)).query(xyz)[0] >= length
    if not near.any():
        raise ValueError("no mesh vertex lies near a boundary without a Dirichlet condition")
    p1 = ngs.H1(mesh, order=1)
    w = ngs.LinearForm(p1.TestFunction() * ngs.dx).Assemble().vec.FV().NumPy()[:mesh.nv].copy()
    rng = np.random.default_rng(0)  # a fixed subsample: tuning is deterministic
    first = _pick(np.flatnonzero(near & on), w, probes // 2, rng)
    second = _pick(np.flatnonzero(near & ~on), w, probes - len(first[0]), rng)
    idx = np.concatenate([first[0], second[0]])
    return xyz[idx], np.concatenate([first[1], second[1]])


def _pick(sel, w, k, rng):
    if len(sel) <= k:
        return sel, w[sel]
    chosen = np.sort(rng.choice(sel, k, replace=False, p=w[sel] / w[sel].sum()))
    return chosen, np.full(k, w[sel].sum() / k)


def _fingerprint(fes) -> str:
    """Identifies a space for the cache: vertex coordinates, free dofs, type, order, sizes, curvature."""
    mesh = fes.mesh
    h = blake2b(digest_size=16)
    h.update(np.ascontiguousarray(mesh.ngmesh.Coordinates(), dtype=float).tobytes())
    h.update(np.array(fes.FreeDofs(), dtype=bool).tobytes())
    h.update(f"{fes.type} {fes.globalorder} {fes.ndof} {mesh.ne} {mesh.nedge} {mesh.GetCurveOrder()}".encode())
    return h.hexdigest()
