"""Estimating the Robin coefficient of Matern priors for a space: ``matern_correlation(..., boundary="robin",
beta=None)``.

The Robin condition du/dn + beta u = 0 keeps the pointwise variance of a Matern field near its interior value up to
the boundary, where the natural (Neumann) condition doubles it (more in corners), if beta suits the field and the
domain. :func:`tune_robin` chooses beta for a given FE space: the beta whose variance is closest to constant,

    J(beta) = ( (1/|L|) int_L (log sigma_beta^2(x) - log s^2)^2 dx )^(1/2),   L = {x : dist(x, boundary) < length},

the root-mean-square deviation of log(variance) over the boundary layer from the interior level s^2: the variance
(geometric mean) at points farther than ``length`` from every boundary, which beta does not reach. That is one for
the continuous field; the discretised one can differ slightly (a rough field's FE variance dips between nodes), and
matching the interior keeps that offset out of beta. Domains without such points use s^2 = 1. The variances are those
of the discretised sampler, computed exactly (sigma^2(x) = |L^T g_x|^2), at the points of a quadrature of L that
does not depend on the mesh's vertices: boundary patches about 1.5/kappa wide crossed with depth bins graded from
0.4/kappa at the boundary (the dip of the variance there is about 1/(2 kappa) wide), two points per cell near the
Gauss depths of its bin, with weights from the cell's exact area that integrate linear variation in depth exactly.
The estimate therefore depends on the mesh only through the accuracy of the variance itself. Unit square, length
0.3, P2 (c = beta / kappa for nu = 1 / 2 / 5): 0.713 / 0.487 / 0.298 at h = 0.1 (h kappa = 0.94 / 1.33 / 2.1),
0.716 / 0.489 / 0.298 on a mesh graded to h = 0.005 at the boundary (a vertex-based quadrature gave 0.665 / 0.404 /
0.239 and 0.716 / 0.490 / 0.301); a disk, an L-shape and a curved domain with five holes move by at most 0.7% from h
= 0.05 to 0.025; the minimiser of J with a fine quadrature of the exact variance is within 0.5%.

A constant beta cannot remove every boundary effect: for large nu the variance still dips in corners and overshoots
slightly further in. The result records the remaining deviation.
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

# The layer quadrature, in units of 1/kappa (scaled by tune_robin's ``spacing``)
PATCH = 1.5  # boundary patches (tangential)
FIRST_DEPTH = 0.4  # the first depth bin; the bins grow by GROWTH up to DEEPEST
DEEPEST = 1.5
GROWTH = 1.3
MAX_PROBES = 4000  # beyond this, the patches and bins are coarsened
MAX_BACKGROUND = 3_000_000  # points of the fine background rule that defines the cells' areas
BACKGROUND = {1: 12, 2: 12, 3: 8}  # its points per 1/kappa near the boundary, by dimension
SAMPLES = {1: 16, 2: 16, 3: 8}  # boundary samples per 1/kappa (the depth is measured to their tangent planes)
EDGE_PROBES = 400  # boundary vertices (sharpest first) at which the variance is reported, not weighted
INNER_PROBES = 64  # points beyond the layer that measure the interior level
BOUNDS = (0.01, 10.0)  # the widest search interval for c = beta / kappa
START = 1.95  # the search starts around beta = START / length (1.9-2.0 for nu = 1/2..5 in 1D and 2D)
WINDOW = 0.4  # ... within a factor exp(WINDOW) of it, widened to BOUNDS if the minimum lies at its end
XTOL = 0.01  # tolerance on log c (c to about 1%)
_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 32


@dataclass(frozen=True)
class RobinTuning:
    """Result of :func:`tune_robin`: the Robin coefficient ``beta`` (= ``factor`` * kappa).

    ``rms`` is the root-mean-square deviation of log(variance) over the boundary layer from the interior level
    ``level`` with this beta (0 for a constant variance; 0.1 is about a 10% typical deviation); ``level`` is the
    variance beyond the layer (1.0, nominal, where the domain has no such points: ``reference``), and
    ``variance_range`` the smallest and largest variance at the probe points and the boundary vertices.
    ``classical_rms`` and ``classical_range`` are the same for the classical beta = kappa / 1.42 (with
    ``compare=True``; None otherwise).
    """

    beta: float
    kappa: float
    rms: float
    variance_range: tuple
    probes: int
    evaluations: int
    seconds: float
    device: str
    level: float = 1.0
    reference: str = "interior"
    classical_rms: float | None = None
    classical_range: tuple | None = None

    @property
    def factor(self) -> float:
        return self.beta / self.kappa

    def __str__(self):
        lo, hi = self.variance_range
        classical = ""
        if self.classical_rms is not None:
            clo, chi = self.classical_range
            classical = f" (kappa / 1.42: [{clo:.3f}, {chi:.3f}], {self.classical_rms:.3f})"
        return (f"Robin beta = {self.beta:.4g} ({self.factor:.4g} kappa): boundary-layer variance in [{lo:.3f}, "
                f"{hi:.3f}], rms log-deviation {self.rms:.3f} from the {self.reference} level {self.level:.4f}"
                f"{classical}; {self.probes} probes, {self.evaluations} set-ups on {self.device}, {self.seconds:.1f} s")


def tune_robin(fes, length: float, nu: float = 1.0, *, rtol: float = 1e-7, spacing: float = 1.0,
               max_probes: int = MAX_PROBES, bounds: tuple = BOUNDS, xtol: float = XTOL, device="auto",
               method=None, compare: bool = False) -> RobinTuning:
    """The Robin coefficient beta for which ``matern_correlation(length, nu, boundary="robin", beta=beta)`` on ``fes``
    has its pointwise variance closest to constant (its interior value) near the boundary.

    Criterion: the mean squared deviation of log(variance) from its interior level (see the module docstring) over
    the points within ``length`` of the boundaries without a Dirichlet condition (points within ``length`` of a
    Dirichlet boundary are left out: their variance is pinned down whatever beta is), integrated with a quadrature
    fitted to the boundary layer (see the module docstring; ``spacing``
    scales its cells, ``max_probes`` caps their number). The variances are exact for the discretised sampler, so the
    estimate depends on the mesh only through their accuracy: it changes under refinement only while the variance
    near the boundary does (see ``ngsrng.pointwise_variance``).

    The search runs over c = beta / kappa (log scale, to ``xtol``), starting around beta = 1.95 / length and
    widening to ``bounds`` if needed. Cost: 5-8 set-ups of the Gaussian and one application of L^T per probe and
    set-up (the probes resolve 1/kappa along the boundary: 220-900 on the unit square at length 0.3 for nu = 1-5,
    1000-2700 on the five-hole domain, 1540 on the unit cube at length 0.4 for nu = 1.5, 200 s for 5.7k dofs on a
    laptop CPU). To estimate on a coarser mesh of the same geometry and pass ``beta=result.beta`` on the fine one,
    the coarse mesh must resolve the variance near the boundary; a discretisation offset common to the whole domain
    is absorbed by the interior level (on the unit square, P2 at h kappa = 0.94 / 1.33 / 2.1 for nu = 1 / 2 / 5 gave
    the fine meshes' estimate within 0.6%). ``device``: where the set-ups run ("auto": the GPU if torch sees one;
    the estimate does not depend on it beyond rounding); ``method``: algorithm choices for the set-ups, as in
    :class:`ngsrng.Gaussian`. ``compare=True`` also reports the classical beta = kappa / 1.42 (one more set-up).
    Results are cached per space and parameters, and
    ``matern_correlation(..., boundary="robin", beta=None)`` uses this function.
    """
    import torch

    if length <= 0 or nu <= 0:
        raise ValueError("length and nu must be positive")
    lo, hi = (float(b) for b in bounds)
    if not 0 < lo < hi:
        raise ValueError("bounds must satisfy 0 < low < high")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(torch.device(device))
    key = (_fingerprint(fes), float(length), float(nu), float(rtol), float(spacing), int(max_probes), lo, hi,
           float(xtol), repr(method), bool(compare))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    start = time.perf_counter()
    kappa = math.sqrt(8.0 * nu) / length
    points, weights, edge, inner, inner_w = _layer_quadrature(fes, float(length), kappa, float(spacing),
                                                               int(max_probes))
    variances = _Variances(fes, dict(layer=points, edge=edge, inner=inner), float(length), float(nu), float(rtol),
                           device, method)
    weights = weights / weights.sum()
    best = {}

    def deviation(gauss):
        """(rms of log(variance / level) over the layer, level)"""
        level = 1.0
        if len(inner):
            level = float(np.exp(inner_w @ np.log(variances.values(gauss, "inner")) / inner_w.sum()))
        return float(np.sqrt(weights @ np.log(variances.values(gauss) / level) ** 2)), level

    def objective(t):
        c = math.exp(t)
        gauss = variances.gaussian(c)
        value, level = deviation(gauss)
        if not best or value < best["value"]:  # keep the best set-up only: the boundary vertices are evaluated with it
            best.update(value=value, level=level, c=c, gauss=gauss)
        return value

    from scipy.optimize import minimize_scalar

    full = (math.log(lo), math.log(hi))
    t0 = min(max(math.log(START / (kappa * length)), full[0]), full[1])
    window = (max(t0 - WINDOW, full[0]), min(t0 + WINDOW, full[1]))
    res = minimize_scalar(objective, bounds=window, method="bounded", options={"xatol": float(xtol)})
    if min(res.x - window[0], window[1] - res.x) < 2 * xtol and window != full:
        minimize_scalar(objective, bounds=full, method="bounded", options={"xatol": float(xtol)})
    factor = best["c"]
    if min(math.log(factor) - full[0], full[1] - math.log(factor)) < 2 * xtol:
        warnings.warn(f"the estimated Robin coefficient ({factor:.3g} kappa) lies at the end of the search interval "
                      f"{bounds} (in units of kappa)", RuntimeWarning, stacklevel=2)
    v = np.concatenate([variances.values(best["gauss"]), variances.values(best["gauss"], "edge")])
    classical = {}
    if compare:
        gauss = variances.gaussian(ROBIN_FACTOR)
        vc = np.concatenate([variances.values(gauss), variances.values(gauss, "edge")])
        classical = dict(classical_rms=deviation(gauss)[0], classical_range=(float(vc.min()), float(vc.max())))
    out = RobinTuning(beta=factor * kappa, kappa=kappa, rms=best["value"],
                      variance_range=(float(v.min()), float(v.max())), probes=len(points),
                      evaluations=variances.evaluations, seconds=time.perf_counter() - start, device=device,
                      level=best["level"], reference="interior" if len(inner) else "nominal", **classical)
    _CACHE[key] = out
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return out


class _Variances:
    """Matern Gaussians with beta = c kappa, and their exact pointwise variances at the probes."""

    def __init__(self, fes, groups, length, nu, rtol, device, method):
        from .correlation import MaternCorrelation
        from .observations import observation_operator
        from .space import SpaceInfo

        self.fes, self.length, self.nu, self.rtol, self.device, self.method = fes, length, nu, rtol, device, method
        self.kappa = MaternCorrelation(length, nu).kappa()
        free = SpaceInfo(fes).free
        self.gt = {name: observation_operator(fes, points=pts)[:, free].T.tocsc() if len(pts) else None
                   for name, pts in groups.items()}
        self.evaluations = 0

    def gaussian(self, c):
        from .correlation import MaternCorrelation
        from .gaussian import Gaussian

        spec = MaternCorrelation(self.length, self.nu, self.rtol, "robin", c * self.kappa).bind(self.fes)
        with warnings.catch_warnings():  # the set-ups are internal: no calibration notices from them
            warnings.simplefilter("ignore")
            gauss = Gaussian(self.fes, spec, method=self.method, dtype="float64", seed=0, device=self.device)
        self.evaluations += 1
        return gauss

    def values(self, gauss, group="layer"):
        """|L^T g_x|^2 at a group of points: the layer's probes, the boundary vertices ("edge") or "inner"."""
        import torch

        from .covariance import block_columns

        gt = self.gt[group]
        if gt is None:
            return np.empty(0)
        m = gt.shape[1]
        out = np.empty(m)
        step = block_columns(gauss.nrand, gauss.space.nfree)
        for i in range(0, m, step):
            y = gauss.apply_t(torch.from_numpy(gt[:, i:i + step].toarray()).to(gauss.device))
            out[i:i + step] = (y.double() ** 2).sum(0).cpu().numpy()
        if not np.all(out > 0):
            raise RuntimeError("a probe point has zero variance (a Dirichlet dof?): cannot tune the Robin factor")
        return out


# ------------------------------------------------------------------ the quadrature of the boundary layer


def _layer_quadrature(fes, length, kappa, spacing, max_probes):
    """Points and weights integrating over the layer {dist(x, Robin boundary) < length} (outside the layer of any
    Dirichlet boundary), with cells resolving the boundary dip whatever the mesh; and boundary vertices for the
    reported range (sharpest corners first).

    The cells: (boundary patch of the nearest boundary point, its normal direction, depth bin). Their areas come from
    a fine background rule mapped to the (possibly curved) elements; each cell's probe is the background point
    nearest to the cell's centroid (so it lies in the mesh and in its cell)."""
    from scipy.spatial import cKDTree

    mesh = fes.mesh
    dim = mesh.dim
    bnd = _boundary_samples(fes, kappa)
    if not bnd["robin"].any():
        raise ValueError("every boundary of the space carries a Dirichlet condition: there is no Robin coefficient "
                         "to tune")
    xb, nb = bnd["x"][bnd["robin"]], bnd["n"][bnd["robin"]]
    tree = cKDTree(xb)
    xv, wv = _background(mesh, kappa, tree)
    dist, foot = tree.query(xv, workers=-1)
    # the distance to the nearest sample's tangent plane: exact on flat pieces whatever the samples' spacing; within
    # the spacing r of the distance to that sample (which exceeds the true distance by at most r) elsewhere, e.g. near
    # re-entrant corners, where the plane passes closer than the boundary
    r = tree.query(xb, k=2, workers=-1)[0][:, 1].max() if len(xb) > 1 else 0.0
    dist = np.clip(np.abs(((xb[foot] - xv) * nb[foot]).sum(1)), dist - r, dist)
    keep = dist < length
    if (~bnd["robin"]).any():
        away = cKDTree(bnd["x"][~bnd["robin"]]).query(xv, workers=-1)[0] >= length
        keep &= away
    else:
        away = np.ones(len(xv), bool)
    inner, inner_w = _inner_probes(xv[(dist >= length) & away], wv[(dist >= length) & away], length)
    if not keep.any():
        raise ValueError("no point of the mesh lies near a boundary without a Dirichlet condition")
    xv, wv, dist, foot = xv[keep], wv[keep], dist[keep], foot[keep]
    if dim == 1:
        sector = foot[:, None]  # the two ends
    elif dim == 2:
        sector = (np.floor((np.arctan2(nb[foot, 1], nb[foot, 0]) + np.pi) / (np.pi / 4)).astype(np.int64) % 8)[:, None]
    else:
        sector = np.rint(nb[foot]).astype(np.int64)  # one of 26 directions
    edges = _depth_edges(length, kappa, spacing)
    depth = np.searchsorted(edges, dist, side="right") - 1
    width = PATCH * spacing / kappa
    while True:  # too many cells: wider patches (the depth bins stay: they resolve the dip)
        cell, ncells = _group(np.column_stack([np.floor(xb[foot] / width).astype(np.int64), sector, depth]))
        if ncells <= max_probes or width > 4 * length:
            break
        width *= max(ncells / max_probes, 1.1) ** (1.0 / max(dim - 1, 1))
    area = np.bincount(cell, wv, ncells)
    centroid = np.stack([np.bincount(cell, wv * xv[:, k], ncells) for k in range(dim)], 1) / area[:, None]
    mean_depth = np.bincount(cell, wv * dist, ncells) / area
    bin_of_cell = np.zeros(ncells, np.int64)
    bin_of_cell[cell] = depth
    a, b = edges[bin_of_cell], np.minimum(edges[bin_of_cell + 1], length)
    near = ((xv - centroid[cell]) ** 2).sum(1) / (0.5 * width) ** 2
    by_cell = np.argsort(cell, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(cell[by_cell])) + 1]
    seg = np.repeat(np.arange(ncells), np.diff(np.r_[starts, len(cell)]))

    def nearest(target):  # per cell: the background point closest to the target depth, then to the centroid
        score = np.nan_to_num((((dist - target[cell]) / (0.1 * (b - a))[cell]) ** 2 + near)[by_cell], nan=np.inf)
        hit = np.flatnonzero(score == np.minimum.reduceat(score, starts)[seg])
        return by_cell[hit[np.r_[0, np.flatnonzero(np.diff(seg[hit])) + 1]]]

    half = (b - a) / (2 * np.sqrt(3.0))  # two-point Gauss depths of each bin
    p, q = nearest((a + b) / 2 - half), nearest((a + b) / 2 + half)
    dp, dq = dist[p], dist[q]
    # weights integrating 1 and the depth exactly over the cell; one point where that is ill-posed
    with np.errstate(divide="ignore", invalid="ignore"):
        wq = area * (mean_depth - dp) / (dq - dp)
    two = (dq - dp > 0.2 * (b - a)) & (wq > 0) & (wq < area)
    single = nearest(mean_depth)
    points = np.concatenate([xv[p[two]], xv[q[two]], xv[single[~two]]])
    weights = np.concatenate([(area - wq)[two], wq[two], area[~two]])
    return points, weights, _edge_probes(fes, bnd, length), inner, inner_w


def _inner_probes(x, w, length):
    """At most INNER_PROBES points beyond the layer (one per cell of a grid, coarsened as needed), with their cells'
    areas: they measure the interior level of the variance."""
    if not len(x):
        return x, w
    size = length / 2
    while True:
        cell, ncells = _group(np.floor(x / size).astype(np.int64))
        if ncells <= INNER_PROBES:
            break
        size *= 1.25
    area = np.bincount(cell, w, ncells)
    centroid = np.stack([np.bincount(cell, w * x[:, k], ncells) for k in range(x.shape[1])], 1) / area[:, None]
    order = np.lexsort((((x - centroid[cell]) ** 2).sum(1), cell))
    first = order[np.r_[0, np.flatnonzero(np.diff(cell[order])) + 1]]
    return x[first], area


def _group(rows):
    """Cell numbers 0..n-1 of the distinct integer rows (one key per row: faster than np.unique(axis=0))."""
    rows = rows - rows.min(0)
    span = rows.max(0) + 1
    if np.prod(span.astype(float)) < 2.0**62:
        key = np.ravel_multi_index(tuple(rows.T.astype(np.intp)), tuple(int(k) for k in span))
        _, cell = np.unique(key, return_inverse=True)
    else:
        _, cell = np.unique(rows, axis=0, return_inverse=True)
    cell = cell.ravel()
    return cell, int(cell.max()) + 1


def _depth_edges(length, kappa, spacing):
    edges, width = [0.0], FIRST_DEPTH * spacing / kappa
    while edges[-1] + width < length:
        edges.append(edges[-1] + width)
        width = min(width * GROWTH, DEEPEST * spacing / kappa)
    return np.array(edges + [np.inf])


def _boundary_samples(fes, kappa):
    """Points on the boundary (curved geometry), SAMPLES per 1/kappa, with outward normals, whether their element
    carries a Dirichlet condition, and the boundary elements' vertices and normals (for the sharpest vertices)."""
    mesh = fes.mesh
    dim = mesh.dim
    free = np.array(fes.FreeDofs(), dtype=bool)
    els = list(mesh.Elements(ngs.BND))
    verts = [[v.nr for v in el.vertices] for el in els]
    dirichlet = np.zeros(len(els), bool)
    for i, el in enumerate(els):
        dofs = [d for d in fes.GetDofNrs(el) if d >= 0]
        dirichlet[i] = bool(dofs) and not free[dofs].any()
    coords = _vertex_coords(mesh)
    size = _sizes(coords, verts).max(initial=0.0)
    n = max(2, math.ceil(SAMPLES[dim] * kappa * size))
    if dim == 3:
        n = min(n, max(2, int(math.sqrt(2_000_000 / max(len(els), 1)))))
    rules = {et: ngs.IntegrationRule(et, 2 * n - 1) for et in {el.type for el in els}}
    counts = np.array([len(rules[el.type].points) for el in els])
    mp = mesh.MapToAllElements(rules, ngs.BND)
    x = _coords(mesh, mp)
    nrm = np.asarray(ngs.specialcf.normal(dim)(mp), dtype=float).reshape(-1, dim)
    owner = np.repeat(np.arange(len(els)), counts)
    el_normal = np.stack([np.bincount(owner, nrm[:, k], len(els)) for k in range(dim)], 1)
    el_normal /= np.maximum(np.linalg.norm(el_normal, axis=1, keepdims=True), 1e-300)
    return dict(x=x, n=nrm, robin=~dirichlet[owner], verts=verts, dirichlet=dirichlet, el_normal=el_normal,
                coords=coords)


def _background(mesh, kappa, tree):
    """A fine rule on all volume elements (points on the curved elements, weights = reference weights x |det J|),
    with BACKGROUND points per 1/kappa in elements near the boundary and 3 elsewhere (capped at MAX_BACKGROUND)."""
    dim = mesh.dim
    els = list(mesh.Elements(ngs.VOL))
    coords = _vertex_coords(mesh)
    verts = [[v.nr for v in el.vertices] for el in els]
    size, centre = _sizes(coords, verts, centres=True)
    near = tree.query(centre)[0] < 2.0 / kappa + size
    n = max(math.ceil(BACKGROUND[dim] * kappa * size[near].max()) if near.any() else 2,
            math.ceil(3 * kappa * size.max()), 2)
    n = min(n, max(2, int((MAX_BACKGROUND / max(len(els), 1)) ** (1.0 / dim))))
    rules = {et: ngs.IntegrationRule(et, 2 * n - 1) for et in {el.type for el in els}}
    mp = mesh.MapToAllElements(rules, ngs.VOL)
    x = _coords(mesh, mp)
    jac = ngs.specialcf.JacobianMatrix(dim)
    det = np.asarray((ngs.Det(jac) if dim > 1 else jac)(mp), dtype=float).ravel()
    if len(rules) == 1:
        w = np.tile(np.asarray(next(iter(rules.values())).weights), len(els))
    else:
        w = np.concatenate([np.asarray(rules[el.type].weights) for el in els])
    return x, w * np.abs(det)


def _edge_probes(fes, bnd, length):
    """Boundary vertices without a Dirichlet condition nearby, at most EDGE_PROBES, the sharpest (largest turn of
    the boundary normal) first: the variance's extremes sit there."""
    from scipy.spatial import cKDTree

    verts, dirichlet, el_normal, coords = bnd["verts"], bnd["dirichlet"], bnd["el_normal"], bnd["coords"]
    inc_e = np.repeat(np.arange(len(verts)), [len(v) for v in verts])
    inc_v = np.concatenate([np.asarray(v, dtype=np.int64) for v in verts])
    robin = ~dirichlet[inc_e]
    inc_e, inc_v = inc_e[robin], inc_v[robin]
    nv = len(coords)
    # the mean of the distinct incident normals (elements on one flat piece count once, however many they are)
    _, first = np.unique(np.column_stack([inc_v, np.round(el_normal[inc_e], 6)]), axis=0, return_index=True)
    mean = np.zeros((nv, coords.shape[1]))
    np.add.at(mean, inc_v[first], el_normal[inc_e[first]])
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-300)
    flat = np.full(nv, np.inf)
    np.minimum.at(flat, inc_v, (el_normal[inc_e] * mean[inc_v]).sum(1))
    cand = np.unique(inc_v)
    if dirichlet.any():
        cand = cand[cKDTree(bnd["x"][~bnd["robin"]]).query(coords[cand])[0] >= length]
    cand = cand[np.argsort(flat[cand], kind="stable")][:EDGE_PROBES]
    return coords[cand]


def _sizes(coords, verts, centres=False):
    """Largest coordinate extent of each element (and its vertex centroid), grouped by vertex count."""
    size, centre = np.empty(len(verts)), np.empty((len(verts), coords.shape[1]))
    lengths = np.array([len(v) for v in verts])
    for k in np.unique(lengths):
        idx = np.flatnonzero(lengths == k)
        pts = coords[np.array([verts[i] for i in idx])]  # (n, k, dim)
        size[idx] = np.ptp(pts, axis=1).max(axis=1)
        centre[idx] = pts.mean(axis=1)
    return (size, centre) if centres else size


def _vertex_coords(mesh):
    return np.asarray(mesh.ngmesh.Coordinates(), dtype=float).reshape(mesh.nv, -1)[:, :mesh.dim]


def _coords(mesh, mp):
    cf = ngs.CF((ngs.x, ngs.y, ngs.z)[:mesh.dim]) if mesh.dim > 1 else ngs.x
    return np.asarray(cf(mp), dtype=float).reshape(-1, mesh.dim)


def _fingerprint(fes) -> str:
    """Identifies a space for the cache: vertex coordinates, free dofs, type, order, sizes, curvature."""
    mesh = fes.mesh
    h = blake2b(digest_size=16)
    h.update(np.ascontiguousarray(mesh.ngmesh.Coordinates(), dtype=float).tobytes())
    h.update(np.array(fes.FreeDofs(), dtype=bool).tobytes())
    h.update(f"{fes.type} {fes.globalorder} {fes.ndof} {mesh.ne} {mesh.nedge} {mesh.GetCurveOrder()}".encode())
    return h.hexdigest()
