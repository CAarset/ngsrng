"""Stationary Matérn fields on bounded domains: constant pointwise variance by construction.

A Matérn field built from the SPDE on the domain itself needs a boundary condition, and every local one (Neumann,
Dirichlet, Robin with any coefficient, constant or varying along the boundary) distorts it near the boundary: its
pointwise variance is not constant and its correlations are not those of the Matérn covariance.
:class:`StationaryMatern` gives the field on the domain whose law does
not depend on position. Two constructions:

``construction="torus"``
    The Matérn SPDE on a periodic box (a torus) around the domain, restricted to the domain. Shifts of the torus
    commute with the Laplacian, so the field is exactly stationary, k(x, y) = f(x - y), and its variance exactly
    constant -- for any padding and any domain inside the box, convex or not. f is the periodised Matérn
    sum_n k_M(x - y + n L): it is not isotropic, and it differs from the Matérn covariance by the images of the other
    points, which lie at least ``padding`` away; the padding is chosen so that their correlation is at most ``tol``.
    Method "spde": ngsrng's Gaussian pipeline (white noise and sparse solves, the rational route for fractional
    powers) on a structured periodic mesh of the box. Cost: the box's dofs, (extent + padding)^d / |domain| times the
    domain's at equal resolution.

``construction="restrict"``
    The Matérn covariance of R^d restricted to the domain, k(x, y) = k_M(|x - y|): exactly stationary and
    isotropic. Methods:

    - "dense": the kernel matrix at the quadrature points and its eigendecomposition. Exact; O(n^3) time and O(n^2)
      memory in the number n of points, so for small problems (``max_dense`` points). The reference.
    - "fft": circulant embedding (Wood & Chan 1994; Dietrich & Newsam 1997; Graham et al. 2018) on a regular grid over
      the domain's bounding box, embedded in a periodic grid that is enlarged until the embedding is positive
      semidefinite. Exact at the grid nodes when it is; the field at the quadrature points is the multilinear
      interpolant of the grid values. O(N log N) per sample in the grid size N, which depends on the bounding box and
      the grid spacing, not on the mesh.

    "auto": dense up to ``max_dense`` points, fft beyond.

Either field is represented on the given FE space by its L2 projection: the values g at the quadrature points of the
space give the coefficients M^-1 B g, as for ``representation="projection"`` of the distributions. The rule has degree
4p (16 points on a P2 triangle): with the degree-2p rule (as many points as basis functions) M^-1 B would nearly
interpolate the rough point values and overshoot at the vertices. The coefficient covariance is exact for that
projection; the FE field's pointwise variance is constant up to the projection's discretisation error (and, for "fft",
the interpolation from the grid).

**Dirichlet boundaries: the boundary update.** On a space with Dirichlet dofs, the field is the stationary field
*conditioned* on its trace there taking the values of ``mean`` (zero by default): Gaussian conditioning, the unique
law consistent with the stationary prior and the boundary information. Exact, in Matheron form: with c the projected
field on all dofs, F the free and D the Dirichlet dofs, a sample is c_F - Sigma_FD Sigma_DD^-1 c_D (plus the mean).
Sigma_DD costs n_D applications of the factor and its transpose; Sigma_FD Sigma_DD^-1 is kept when it is small and
applied matrix-free otherwise (one more application of the factor and its transpose per batch). The variance is zero
on the Dirichlet boundary and returns to its interior value within about a correlation length. This is not the
Dirichlet SPDE: on the half-line with nu = 3/2 the conditioned variance is 1 - e^-2kx (1 + kx)^2, the SPDE's
1 - e^-2kx (1 + 2kx). Pointwise families (uniform, gamma, ...) built on such a field are standardised by the field
before the boundary update, as those built on a posterior are by its prior (``unconditioned``): on the boundary they
take the transform of the data, and their spread grows back within about a correlation length.

**Spaces on part of the mesh** (``definedon``, with or without ``Compress``): the field lives on the elements that
carry the space. Its quadrature points, the extent of the torus box or FFT grid, and the default resolution come from
those elements only.

Samplers have the interface of the other Gaussians (``apply``/``apply_t``/``nrand``/``mean``/``space``, in-place
``sample(n, out=)``, seeding, pickling) and can be passed as ``correlation=`` to :func:`ngsrng.default_rng`.
Scalar spaces with one element type. On a GPU (``device="cuda"``) everything runs there: the FFT, the torus
Gaussian, the projection and the boundary update.
"""

import copy
import math
import time
import warnings

import ngsolve as ngs
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import torch
from scipy.optimize import brentq
from scipy.special import kv

from .backend import DeviceCache, Rebuildable, available_memory, device_copy, resolve_device, sparse_to_torch
from .inplace import InPlace
from .quadrature import QuadratureRule
from .seeding import as_source, inherit
from .solvers import NGSolveDirect
from .space import SpaceInfo, coefficient_vector, mass_form

CONSTRUCTIONS = {"torus": ("spde",), "restrict": ("dense", "fft")}
MAX_DENSE = 3000  # quadrature points up to which construction="restrict" uses the exact dense kernel by default (an
# eigendecomposition of about 20 s on a laptop; the cost grows with the cube)
DENSE_CONDITIONING_VALUES = 2**24  # Sigma_FD Sigma_DD^-1 kept (n_free x n_D values) up to this size
EMBEDDING_GROWTH = 1.25  # circulant embedding: enlarge the periodic grid by this factor until it is PSD
EMBEDDING_MAX_FACTOR = 8  # ... up to this many times the grid (per direction)
EMBEDDING_TOL = 1e-10  # negative embedding eigenvalues below this fraction of the largest count as rounding
FFT_BLOCK_VALUES = 2**22  # complex values per FFT block (64 MB): the columns of a block share one transform
FFT_BLOCK_VALUES_CUDA = 2**24  # on a GPU (256 MB): larger batched transforms
F64 = torch.float64


def _sparse_pair(cache, key, mat, device):
    """A SciPy matrix and its transpose as float64 sparse tensors on ``device``, made once per device."""
    k = (key, str(device))
    if k not in cache:
        cache[k] = (sparse_to_torch(mat, device, F64), sparse_to_torch(mat.T.tocsr(), device, F64))
    return cache[k]


def matern_function(r, length, nu):
    """Unit-variance Matérn correlation at distances ``r``: 2^(1-nu)/Gamma(nu) (kappa r)^nu K_nu(kappa r),
    kappa = sqrt(8 nu) / length (``length`` is the practical range, as in :func:`ngsrng.matern_correlation`)."""
    x = math.sqrt(8.0 * nu) / length * np.asarray(r, dtype=float)
    out = np.ones_like(x)
    pos = x > 0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        vals = 2.0 ** (1 - nu) / math.gamma(nu) * x[pos] ** nu * kv(nu, x[pos])
    out[pos] = np.nan_to_num(vals, nan=0.0)
    return out


def padding_for(length, nu, tol) -> float:
    """The distance beyond which the Matérn correlation is below ``tol``."""
    if not 0 < tol < 1:
        raise ValueError("tol must lie in (0, 1)")
    return float(brentq(lambda r: matern_function(r, length, nu) - tol, 1e-9 * length, 100.0 * length,
                        xtol=1e-6 * length))


def _variance_scale(dim, length, nu) -> float:
    """1/sqrt of the variance of (kappa^2 - Laplace)^(-alpha/2) W on R^dim, alpha = nu + dim/2."""
    kappa = math.sqrt(8.0 * nu) / length
    var = math.gamma(nu) / (math.gamma(nu + dim / 2) * (4 * math.pi) ** (dim / 2) * kappa ** (2 * nu))
    return 1.0 / math.sqrt(var)


def _median_edge(mesh, active=None) -> float:
    """Median edge length of the mesh, or of its ``active`` elements (a mask over the volume elements)."""
    xyz = np.array([v.point for v in mesh.vertices], dtype=float)
    if active is None or active.all():
        lens = [np.linalg.norm(xyz[e.vertices[0].nr] - xyz[e.vertices[1].nr]) for e in mesh.edges]
        if not lens:  # 1D meshes: the element lengths
            lens = [np.linalg.norm(xyz[el.vertices[0].nr] - xyz[el.vertices[1].nr]) for el in mesh.Elements(ngs.VOL)]
        return float(np.median(lens))
    els = [mesh[ngs.ElementId(ngs.VOL, int(i))] for i in np.flatnonzero(active)]
    if mesh.dim == 1:
        pairs = [(el.vertices[0].nr, el.vertices[1].nr) for el in els]
    else:
        edges = {e.nr for el in els for e in el.edges}
        pairs = [tuple(v.nr for v in mesh[ngs.NodeId(ngs.EDGE, e)].vertices) for e in edges]
    return float(np.median([np.linalg.norm(xyz[a] - xyz[b]) for a, b in pairs]))


class _UsedDofs(SpaceInfo):
    """The space of the field before its boundary update: the same FE space with every *used* dof free, those on a
    Dirichlet boundary too. Dofs no element uses (outside ``definedon``) stay out. The same space object rather than
    a rebuilt twin, so that compressed and other wrapped spaces keep their dof numbering."""

    def __post_init__(self):
        self.free = np.flatnonzero(self.mass.diagonal() > 0)


def _active_elements(space, quad) -> np.ndarray:
    """Mask of the mesh's volume elements that carry used dofs of ``space`` (every element, unless the space lives
    on part of the mesh: ``definedon``, ``Compress``)."""
    if (abs(quad.b).sum(axis=0) > 0).all():  # every quadrature point carries basis functions: the whole mesh
        return np.ones(space.mesh.ne, dtype=bool)
    used = np.zeros(space.ndof, dtype=bool)
    used[space.free] = True
    out = np.zeros(space.mesh.ne, dtype=bool)
    for el in space.fes.Elements(ngs.VOL):
        dofs = [d for d in el.dofs if d >= 0]
        out[el.nr] = bool(dofs) and bool(used[dofs].any())
    return out


def _points_of(quad, active) -> np.ndarray:
    """Indices of the quadrature points (IntegrationRuleSpace dofs) on the ``active`` elements, in increasing order."""
    if active.all():
        return np.arange(quad.npoints)
    keep = [quad.irs.GetDofNrs(ngs.ElementId(ngs.VOL, int(i))) for i in np.flatnonzero(active)]
    return np.sort(np.concatenate([np.asarray(k, dtype=int) for k in keep]))


def basis_at(fes, points) -> sp.csr_matrix:
    """E (npoints x ndof): the values of the basis functions of ``fes`` at ``points`` (inside the mesh), so that
    E c are the values of the FE function with coefficients c. Points are located with NGSolve's vectorised search and
    evaluated with the elements' shape functions."""
    mesh = fes.mesh
    d = mesh.dim
    pts = np.asarray(points, dtype=float)
    mips = mesh(*[pts[:, i] for i in range(d)])
    nrs = np.asarray(mips["nr"])
    if (nrs < 0).any():
        bad = pts[np.flatnonzero(nrs < 0)[0]]
        raise ValueError(f"point {tuple(bad)} lies outside the mesh")
    ref = [np.asarray(mips[c]) for c in ("x", "y", "z")[:d]]
    order = np.argsort(nrs, kind="stable")
    cuts = np.flatnonzero(np.diff(nrs[order])) + 1
    rows, cols, vals = [], [], []
    for group in np.split(order, cuts):
        el = ngs.ElementId(ngs.VOL, int(nrs[group[0]]))
        fe = fes.GetFE(el)
        dofs = np.asarray(fes.GetDofNrs(el))
        keep = dofs >= 0
        for i in group:
            shape = np.asarray(fe.CalcShape(*[r[i] for r in ref]))
            rows.append(np.full(keep.sum(), i))
            cols.append(dofs[keep])
            vals.append(shape[keep])
    e = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(pts), fes.ndof))
    e.sum_duplicates()
    return e


class _TorusSource:
    """The Matérn SPDE on a structured periodic box mesh, evaluated at the target points."""

    method = "spde"

    def __init__(self, points, lo, hi, length, nu, tol, resolution, order, rtol, gauss_method, device="cpu",
                 torch_engine=False):
        from ngsolve.meshes import Make1DMesh, MakeStructured2DMesh, MakeStructured3DMesh

        from .gaussian import Gaussian
        from .specs import matern

        d = points.shape[1]
        self.padding = padding_for(length, nu, tol)
        extent = hi - lo
        self.box = extent + self.padding  # period in each direction: images at least `padding` away
        self.origin = lo - self.padding / 2
        n = [max(2, int(math.ceil(b / resolution))) for b in self.box]
        o, b = self.origin, self.box
        if d == 1:
            mesh = Make1DMesh(n[0], mapping=_Affine1(o[0], b[0]), periodic=True)
        elif d == 2:
            mesh = MakeStructured2DMesh(quads=False, nx=n[0], ny=n[1], periodic_x=True, periodic_y=True,
                                        mapping=_Affine2(o, b))
        else:
            mesh = MakeStructured3DMesh(hexes=False, nx=n[0], ny=n[1], nz=n[2], periodic_x=True, periodic_y=True,
                                        periodic_z=True, mapping=_Affine3(o, b))
        fes = ngs.Periodic(ngs.H1(mesh, order=order))
        kappa = math.sqrt(8.0 * nu) / length
        spec = matern(kappa, nu + d / 2, _variance_scale(d, length, nu))
        spec.rtol = rtol
        if torch_engine:  # the Gaussian computes on the same device as the field (a CPU device: the torch engine)
            gauss_method = {**(gauss_method or {}), "engine": "torch"}
        self.gauss = Gaussian(fes, spec, dtype="float64", seed=0, method=gauss_method,  # randomness comes as eta
                              device=device)
        self.e = basis_at(fes, points)[:, self.gauss.space.free].tocsr()
        self._device = DeviceCache()
        self.nrand = self.gauss.nrand
        self.info = {"padding": self.padding, "box": [float(x) for x in self.box], "cells": n,
                     "ndof": int(self.gauss.space.nfree), "order": order, "gaussian": self.gauss.plan}

    def values(self, eta):
        return self.e @ self.gauss.apply(torch.from_numpy(eta)).numpy()

    def values_t(self, g):
        return self.gauss.apply_t(torch.from_numpy(np.ascontiguousarray(self.e.T @ g))).numpy()

    def values_torch(self, eta):
        e, _ = _sparse_pair(self._device, "e", self.e, eta.device)
        return e @ self.gauss.apply(eta).to(F64)

    def values_t_torch(self, g):
        _, et = _sparse_pair(self._device, "e", self.e, g.device)
        return self.gauss.apply_t(et @ g).to(F64)


class _Affine1:  # mesh mappings: callable classes, not lambdas (samplers pickle)
    def __init__(self, o, b):
        self.o, self.b = float(o), float(b)

    def __call__(self, x):
        return self.o + self.b * x


class _Affine2:
    def __init__(self, o, b):
        self.o, self.b = [float(v) for v in o], [float(v) for v in b]

    def __call__(self, x, y):
        return self.o[0] + self.b[0] * x, self.o[1] + self.b[1] * y


class _Affine3:
    def __init__(self, o, b):
        self.o, self.b = [float(v) for v in o], [float(v) for v in b]

    def __call__(self, x, y, z):
        return self.o[0] + self.b[0] * x, self.o[1] + self.b[1] * y, self.o[2] + self.b[2] * z


class _DenseSource:
    """The exact Matérn kernel at the target points: K = V diag(lam) V^T, factor V diag(sqrt(lam)) (eigenvalues above
    ``rtol`` times the largest)."""

    method = "dense"

    def __init__(self, points, length, nu, max_points, rtol=1e-12):
        n = len(points)
        if n > max_points:
            raise MemoryError(f"the dense Matérn kernel needs {n} x {n} values (limit max_dense={max_points}): "
                              "use method='fft' or construction='torus'")
        from scipy.spatial.distance import cdist

        k = matern_function(cdist(points, points), length, nu)
        lam, vec = sla.eigh(k, overwrite_a=True, check_finite=False)
        keep = lam > rtol * lam[-1]
        self.r = vec[:, keep] * np.sqrt(lam[keep])
        self.nrand = self.r.shape[1]
        self.info = {"points": n, "rank": int(self.nrand), "dropped_trace": float(lam[~keep].clip(0).sum() / n)}

    def values(self, eta):
        return self.r @ eta

    def values_t(self, g):
        return self.r.T @ g

    def _r(self, device):
        cache = self.__dict__.setdefault("_device", DeviceCache())
        if str(device) not in cache:
            cache[str(device)] = torch.from_numpy(np.ascontiguousarray(self.r)).to(device)
        return cache[str(device)]

    def values_torch(self, eta):
        return self._r(eta.device) @ eta

    def values_t_torch(self, g):
        return self._r(g.device).T @ g


class _GridSource:
    """Circulant embedding of the Matérn kernel on a regular grid over the bounding box [lo, hi], interpolated
    multilinearly to the target points.

    The grid (n_i nodes per direction, spacing h_i) is embedded in a periodic grid of m_i >= 2 (n_i - 1) nodes with
    the kernel at the wrapped distances; its eigenvalues lam (an FFT of the first column) are real. If they are
    nonnegative, Re F(a (xi_1 + i xi_2)), a = sqrt(lam / M), xi_1, xi_2 ~ N(0, I_M), has exactly the kernel as
    covariance on the periodic grid, hence on the original grid. The periodic grid is enlarged until the negative
    eigenvalues are rounding; what is left is clipped and reported (``info["clipped"]``, relative to the trace)."""

    method = "fft"

    def __init__(self, points, lo, hi, length, nu, spacing):
        extent = np.maximum(hi - lo, 0.0)
        self.n = [max(2, int(math.ceil(e / spacing)) + 1) for e in extent]
        self.h = [e / (k - 1) if e > 0 else spacing for e, k in zip(extent, self.n)]
        self.lo = np.asarray(lo, dtype=float)
        m = [2 * (k - 1) for k in self.n]
        attempts, clipped = 0, 0.0
        while True:
            attempts += 1
            axes = [np.minimum(np.arange(mi), mi - np.arange(mi)) * hi_ for mi, hi_ in zip(m, self.h)]
            r2 = sum(np.meshgrid(*[a**2 for a in axes], indexing="ij"))
            lam = np.fft.fftn(matern_function(np.sqrt(r2), length, nu)).real
            ok = lam.min() >= -EMBEDDING_TOL * lam.max()
            if ok or any(mi >= EMBEDDING_MAX_FACTOR * ni for mi, ni in zip(m, self.n)):
                break
            m = [int(math.ceil(EMBEDDING_GROWTH * mi)) for mi in m]
        if not ok:
            clipped = float(-lam[lam < 0].sum() / lam.sum())
            warnings.warn(f"circulant embedding not positive semidefinite after {attempts} enlargements: negative "
                          f"eigenvalues clipped ({clipped:.1e} of the trace); the grid covariance is approximate",
                          RuntimeWarning, stacklevel=4)
        self.m = m
        self.M = int(np.prod(m))
        self.a = np.sqrt(np.clip(lam, 0, None) / self.M)
        self.nrand = 2 * self.M
        self.w = self._interpolation(points)
        self.info = {"grid": self.n, "spacing": [float(x) for x in self.h], "embedding": m, "attempts": attempts,
                     "clipped": clipped, "exact_on_grid": bool(ok)}

    def _interpolation(self, points):
        """Multilinear interpolation weights (npoints x prod(n)) from the grid nodes to the points."""
        d = points.shape[1]
        t = (points - self.lo) / np.asarray(self.h)
        idx = np.clip(np.floor(t).astype(int), 0, np.asarray(self.n) - 2)
        frac = np.clip(t - idx, 0.0, 1.0)
        strides = np.cumprod([1] + self.n[::-1][:-1])[::-1]  # C order: last axis fastest
        rows, cols, vals = [], [], []
        for corner in range(2**d):
            bits = [(corner >> k) & 1 for k in range(d)]
            wgt = np.ones(len(points))
            lin = np.zeros(len(points), dtype=np.int64)
            for k, bit in enumerate(bits):
                wgt *= frac[:, k] if bit else 1.0 - frac[:, k]
                lin += (idx[:, k] + bit) * strides[k]
            rows.append(np.arange(len(points)))
            cols.append(lin)
            vals.append(wgt)
        w = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(points), int(np.prod(self.n))))
        w.sum_duplicates()
        return w

    def _crop(self):
        return tuple(slice(0, k) for k in self.n)

    def _blocks(self, k):
        step = max(1, FFT_BLOCK_VALUES // self.M)  # columns per FFT: bounded complex temporaries
        return [(j, min(k, j + step)) for j in range(0, k, step)]

    def values(self, eta):
        k = eta.shape[1]
        out = np.empty((self.w.shape[0], k))
        axes = tuple(range(len(self.m)))
        for j0, j1 in self._blocks(k):
            xi = (eta[: self.M, j0:j1] + 1j * eta[self.M:, j0:j1]).reshape(*self.m, j1 - j0)
            y = np.fft.fftn(self.a[..., None] * xi, axes=axes).real[self._crop()]
            out[:, j0:j1] = self.w @ y.reshape(-1, j1 - j0)
        return out

    def values_t(self, g):
        k = g.shape[1]
        out = np.empty((self.nrand, k))
        a = self.a[..., None]
        for j0, j1 in self._blocks(k):
            grid = np.zeros((*self.m, j1 - j0))
            grid[self._crop()] = (self.w.T @ g[:, j0:j1]).reshape(*self.n, j1 - j0)
            u = np.fft.fftn(grid, axes=tuple(range(len(self.m))))
            out[: self.M, j0:j1] = (a * u.real).reshape(self.M, -1)
            out[self.M:, j0:j1] = (-a * u.imag).reshape(self.M, -1)
        return out

    # On a torch device: the same maps with the sample axis first, (k, *m), so that cuFFT batches over it.

    def _a(self, device):
        cache = self.__dict__.setdefault("_device", DeviceCache())
        if ("a", str(device)) not in cache:
            cache[("a", str(device))] = torch.from_numpy(np.ascontiguousarray(self.a)).to(device)
        return cache[("a", str(device))]

    def _blocks_torch(self, k, device):
        values = FFT_BLOCK_VALUES_CUDA if torch.device(device).type == "cuda" else FFT_BLOCK_VALUES
        step = max(1, values // self.M)
        return [(j, min(k, j + step)) for j in range(0, k, step)]

    def values_torch(self, eta):
        dev, k, d = eta.device, eta.shape[1], len(self.m)
        w, _ = _sparse_pair(self.__dict__.setdefault("_device", DeviceCache()), "w", self.w, dev)
        a, crop = self._a(dev), (slice(None),) + self._crop()
        out = torch.empty((self.w.shape[0], k), dtype=F64, device=dev)
        for j0, j1 in self._blocks_torch(k, dev):
            xi = torch.complex(eta[: self.M, j0:j1].T, eta[self.M:, j0:j1].T).reshape(j1 - j0, *self.m)
            y = torch.fft.fftn(a * xi, dim=tuple(range(1, d + 1))).real[crop]
            out[:, j0:j1] = w @ y.reshape(j1 - j0, -1).T.contiguous()
        return out

    def values_t_torch(self, g):
        dev, k, d = g.device, g.shape[1], len(self.m)
        _, wt = _sparse_pair(self.__dict__.setdefault("_device", DeviceCache()), "w", self.w, dev)
        a, crop = self._a(dev), (slice(None),) + self._crop()
        out = torch.empty((self.nrand, k), dtype=F64, device=dev)
        for j0, j1 in self._blocks_torch(k, dev):
            grid = torch.zeros((j1 - j0, *self.m), dtype=F64, device=dev)
            grid[crop] = (wt @ g[:, j0:j1].contiguous()).T.reshape(j1 - j0, *self.n)
            u = torch.fft.fftn(grid, dim=tuple(range(1, d + 1)))
            out[: self.M, j0:j1] = (a * u.real).reshape(j1 - j0, self.M).T
            out[self.M:, j0:j1] = (-a * u.imag).reshape(j1 - j0, self.M).T
        return out


class StationaryMatern(InPlace, Rebuildable):
    """Matérn Gaussian random function on ``fes`` whose law does not depend on position (see the module docstring):
    unit pointwise variance up to the FE projection, correlation range ``length``, smoothness ``nu``.

    ``construction`` ("torus" or "restrict") has no default: the two are different fields (periodised or exact
    Matérn), and the right one depends on the application. ``method``: "auto", or "spde" (torus), "dense" / "fft"
    (restrict). ``tol``: for "torus", the correlation between a point and the nearest image of another (it sets the
    padding). ``resolution``: the torus mesh size or the FFT grid spacing (default: the median edge length of
    ``fes``'s mesh; a quarter of it for the grid in 1D and 2D, half in 3D). ``mean``: a GridFunction,
    CoefficientFunction or coefficient vector; on a space with Dirichlet dofs its values there are the boundary data
    the field is conditioned on.
    ``method_options``: passed to the torus Gaussian as its ``method`` (solvers, noise). ``seed``: as for every
    sampler (:mod:`ngsrng.seeding`).

    ``device``: "cpu" or a torch device such as "cuda": where the field is computed and sampled. On a GPU everything
    runs there (the FFT, the torus Gaussian's solves, the projection and the boundary update), the samples are
    tensors there and their random numbers are drawn there (so, as on the torch engine, seeded draws depend on how
    a batch is split). ``engine``: "numpy" (the CPU's default) or "torch" (a GPU's; on the CPU only useful for
    testing the torch code path).
    """

    def __init__(self, fes, length, nu=1.0, *, construction=None, method="auto", mean=None, seed=None, tol=1e-3,
                 resolution=None, max_dense=MAX_DENSE, rtol=1e-7, method_options=None, device="cpu", engine=None):
        if construction not in CONSTRUCTIONS:
            raise ValueError("construction has no default: 'torus' (the Matérn SPDE on a periodic box around the "
                             "domain: exactly stationary, periodised Matérn, sparse) or 'restrict' (the Matérn "
                             "covariance of R^d restricted to the domain: exact, dense or FFT on a grid)")
        if length <= 0 or nu <= 0:
            raise ValueError("length and nu must be positive")
        if method == "auto":
            method = "spde" if construction == "torus" else None
        if method is not None and method not in CONSTRUCTIONS[construction]:
            raise ValueError(f"method {method!r} is not available for construction={construction!r} "
                             f"(available: {', '.join(CONSTRUCTIONS[construction])}, or 'auto')")
        if fes.TrialFunction().dim != 1:
            raise NotImplementedError("StationaryMatern needs a scalar space")
        self.device = resolve_device(device)
        engine = engine or ("numpy" if self.device.type == "cpu" else "torch")
        if engine not in ("numpy", "torch"):
            raise ValueError("engine must be 'numpy' or 'torch'")
        if engine == "numpy" and self.device.type != "cpu":
            raise ValueError("the NumPy engine runs on the CPU only")
        self._torch, self._device = engine == "torch", DeviceCache()
        start = time.perf_counter()
        self.length, self.nu, self.construction, self.tol = float(length), float(nu), construction, float(tol)
        self.space = SpaceInfo(fes)
        self._full = _UsedDofs(fes)  # the field before the boundary update lives on every used dof
        mesh = fes.mesh
        d = mesh.dim
        quad = QuadratureRule(self._full)  # degree 4p: a projection, not an interpolation of rough point values
        # only the elements that carry the space (definedon, Compress): their points, extent and mesh size
        active = _active_elements(self._full, quad)
        keep = _points_of(quad, active)
        points = quad.points[keep]
        self._b = quad.b.tocsr() if active.all() else quad.b.tocsc()[:, keep].tocsr()  # ndof x nq: w_q phi_i(x_q)
        lo, hi = points.min(0), points.max(0)
        corners = np.unique([v.nr for el in mesh.Elements(ngs.VOL) if active[el.nr] for v in el.vertices])
        xyz = np.array([mesh.vertices[int(v)].point for v in corners], dtype=float)[:, :d]
        lo, hi = np.minimum(lo, xyz.min(0)), np.maximum(hi, xyz.max(0))
        h = _median_edge(mesh, active) if resolution is None else float(resolution)
        if method is None:
            method = "dense" if len(points) <= max_dense else "fft"
        if method == "spde":
            self._src = _TorusSource(points, lo, hi, self.length, self.nu, self.tol, h, max(1, self.space.order),
                                     rtol, method_options, self.device, self._torch)
        elif method == "dense":
            self._src = _DenseSource(points, self.length, self.nu, max_dense)
        else:
            spacing = h / (4 if d <= 2 else 2) if resolution is None else h
            self._src = _GridSource(points, lo, hi, self.length, self.nu, spacing)
        self.method = method
        self._mass = NGSolveDirect(self._full, mass_form)
        self._mean = np.zeros(fes.ndof) if mean is None else coefficient_vector(mean, self.space)
        self.source = as_source(seed)
        self._prior = None
        self._maps, self._post_maps, self._conditioning = [], [], []  # derived samplers
        # the boundary update: condition on the Dirichlet dofs (the used dofs that are not free)
        self._dir = np.setdiff1d(self._full.free, self.space.free)
        self._cond = None
        if self._dir.size:
            self._setup_conditioning()
        self.plan = {"space": self.space.describe(), "construction": construction, "method": method,
                     "quadrature_points": int(len(points)), "source": self._src.info,
                     "conditioning": None if self._cond is None else {"dirichlet_dofs": int(self._dir.size),
                                                                       "mode": self._cond[0]},
                     "device": str(self.device), "engine": engine, "setup_s": None}
        self.plan["setup_s"] = round(time.perf_counter() - start, 3)
        # pointwise families standardise a conditioned field by the field it was conditioned from (as posteriors by
        # their prior): here the field before the boundary update (ngsrng.distributions.pointwise)
        self.unconditioned = None if self._cond is None else self._unconditioned()

    def _unconditioned(self):
        """The field before its boundary update: same source, mean and randomness, all used dofs free."""
        view = copy.copy(self)
        view.space, view._cond, view._dir, view.unconditioned = self._full, None, np.empty(0, dtype=int), None
        view._maps, view._post_maps, view._conditioning, view._prior = [], [], [], None
        view.plan = {**self.plan, "space": self._full.describe(), "conditioning": None}
        return view

    # ---- the factor on all dofs, before the boundary update: eta -> M^-1 B g(eta) ----
    def _project(self, eta):
        g = self._src.values(np.ascontiguousarray(eta, dtype=np.float64))
        bg = self._b @ g
        used = self._full.free  # unused dofs (outside definedon) carry nothing
        out = np.zeros_like(bg)
        out[used] = self._mass.apply(torch.from_numpy(np.ascontiguousarray(bg[used]))).numpy()
        return out

    def _project_t(self, y):
        used, y = self._full.free, np.asarray(y, dtype=np.float64)
        z = np.zeros(y.shape)
        z[used] = self._mass.apply(torch.from_numpy(np.ascontiguousarray(y[used]))).numpy()
        return self._src.values_t(np.ascontiguousarray(self._b.T @ z))

    # ---- the same on a torch device (float64 tensors there) ----
    @property
    def torch_device(self):
        """The torch device this field computes on, or None (NumPy on the CPU)."""
        return self.device if self._torch else None

    def _projection_ops(self, device):
        """B on the used dofs (n_used x npoints) and its transpose, a mass solver (cuDSS, or Jacobi PCG without
        it) and the used dofs' indices, on ``device``."""
        key = ("projection", str(device))
        if key not in self._device:
            from .solvers import device_spd_solver

            used = self._full.free
            b = self._b[used].tocsr()
            self._device[key] = (sparse_to_torch(b, device, F64), sparse_to_torch(b.T.tocsr(), device, F64),
                                 device_spd_solver(self._full.mass_free, device), torch.from_numpy(used).to(device))
        return self._device[key]

    def _project_torch(self, eta):
        b, _, mass, used = self._projection_ops(eta.device)
        out = torch.zeros((self.space.ndof, eta.shape[1]), dtype=F64, device=eta.device)
        out[used] = mass.apply(b @ self._src.values_torch(eta))
        return out

    def _project_t_torch(self, y):
        _, bt, mass, used = self._projection_ops(y.device)
        return self._src.values_t_torch((bt @ mass.apply(y[used])).contiguous())

    def _sigma_torch(self, x_full):
        """Sigma x for full-dof columns x (before the boundary update), on x's device."""
        return self._project_torch(self._project_t_torch(x_full))

    def _cond_ops(self, device):
        """The boundary update's dense regression, or the Cholesky factor of Sigma_DD (lower), on ``device``."""
        key = ("conditioning", str(device))
        if key not in self._device:
            mode, op = self._cond
            mat = op if mode == "dense" else np.tril(op[0])
            self._device[key] = torch.from_numpy(np.ascontiguousarray(mat)).to(device)
        return self._device[key]

    def _base_apply_torch(self, x):
        free = device_copy(self, self.space.free, x.device)
        c = self._project_torch(x)
        out = c[free]
        if self._cond is not None:
            op, dirichlet = self._cond_ops(x.device), device_copy(self, self._dir, x.device)
            cd = c[dirichlet]
            if self._cond[0] == "dense":
                out = out - op @ cd
            else:
                z = torch.zeros_like(c)
                z[dirichlet] = torch.cholesky_solve(cd, op)
                out = out - self._sigma_torch(z)[free]
        return out

    def _base_apply_t_torch(self, v):
        free = device_copy(self, self.space.free, v.device)
        full = torch.zeros((self.space.ndof, v.shape[1]), dtype=F64, device=v.device)
        full[free] = v
        if self._cond is not None:
            op, dirichlet = self._cond_ops(v.device), device_copy(self, self._dir, v.device)
            if self._cond[0] == "dense":
                full[dirichlet] = -op.T @ v
            else:
                z = torch.zeros_like(full)
                z[free] = v
                full[dirichlet] = -torch.cholesky_solve(self._sigma_torch(z)[dirichlet], op)
        return self._project_t_torch(full)

    def _setup_conditioning(self):
        """Sigma_DD (n_D applications of the factor and its transpose) and its Cholesky factor; the regression
        Sigma_FD Sigma_DD^-1 when it is small."""
        nd, nf, ndof = self._dir.size, self.space.nfree, self.space.ndof
        if self._torch:  # the n_D applications on the device; Sigma[:, D] comes back once
            e = torch.zeros((ndof, nd), dtype=F64, device=self.device)
            e[torch.from_numpy(self._dir).to(self.device), torch.arange(nd, device=self.device)] = 1.0
            cols = self._sigma_torch(e).cpu().numpy()
        else:
            e = np.zeros((ndof, nd))
            e[self._dir, np.arange(nd)] = 1.0
            cols = self._project(self._project_t(e))  # Sigma[:, D], all dofs
        sdd = cols[self._dir]
        sdd = 0.5 * (sdd + sdd.T)
        chol = sla.cho_factor(sdd + 1e-14 * np.trace(sdd) / nd * np.eye(nd), lower=True)
        if nf * nd <= DENSE_CONDITIONING_VALUES:
            reg = sla.cho_solve(chol, cols[self.space.free].T).T  # n_F x n_D
            self._cond = ("dense", reg)
        else:
            mem = available_memory()
            if mem is not None and nd * nd * 8 > 0.5 * mem:
                raise MemoryError(f"conditioning on {nd} Dirichlet dofs needs a {nd} x {nd} dense block")
            self._cond = ("matrix-free", chol)

    def _sigma_rows(self, x_full, rows):
        """(Sigma x)[rows] for full-dof columns x."""
        return self._project(self._project_t(x_full))[rows]

    @property
    def nrand(self) -> int:
        return self._src.nrand

    @property
    def mean(self):
        return self._mean.copy()

    @property
    def info(self) -> dict:
        return self.plan

    def apply(self, eta: torch.Tensor) -> torch.Tensor:
        """eta (nrand x k) -> zero-mean free-dof coefficients (nfree x k). Derived samplers (posteriors) apply their
        noise-space maps before and their FE-space maps after the field's factor, as ``Gaussian`` does."""
        x = eta.detach().to(self.device, F64) if self._torch else eta.detach().cpu().double()
        squeeze = x.ndim == 1
        x = x[:, None] if squeeze else x
        for m in reversed(self._maps):
            x = m.apply_torch(x)
        c = self._base_apply_torch(x) if self._torch else self._base_apply(x)
        for pm in self._post_maps:
            c = pm.apply_torch(c)
        return c[:, 0] if squeeze else c

    def apply_t(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint of ``apply``: (nfree x k) -> (nrand x k)."""
        x = y.detach().to(self.device, F64) if self._torch else y.detach().cpu().double()
        squeeze = x.ndim == 1
        x = x[:, None] if squeeze else x
        for pm in reversed(self._post_maps):
            x = pm.apply_t_torch(x)
        eta = self._base_apply_t_torch(x) if self._torch else self._base_apply_t(x)
        for m in self._maps:
            eta = m.apply_torch(eta)
        return eta[:, 0] if squeeze else eta

    def _base_apply(self, eta: torch.Tensor) -> torch.Tensor:
        """The field's own factor (before any derived maps)."""
        e = eta.detach().cpu().double().numpy()
        squeeze = e.ndim == 1
        e = e[:, None] if squeeze else e
        c = self._project(e)
        out = c[self.space.free]
        if self._cond is not None:
            mode, op = self._cond
            cd = c[self._dir]
            if mode == "dense":
                out = out - op @ cd
            else:
                x = np.zeros_like(c)
                x[self._dir] = sla.cho_solve(op, cd)
                out = out - self._sigma_rows(x, self.space.free)
        return torch.from_numpy(out[:, 0] if squeeze else out)

    def _base_apply_t(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint of ``_base_apply``."""
        v = y.detach().cpu().double().numpy()
        squeeze = v.ndim == 1
        v = v[:, None] if squeeze else v
        full = np.zeros((self.space.ndof, v.shape[1]))
        full[self.space.free] = v
        if self._cond is not None:
            mode, op = self._cond
            if mode == "dense":
                full[self._dir] = -op.T @ v
            else:
                x = np.zeros_like(full)
                x[self.space.free] = v
                full[self._dir] = -sla.cho_solve(op, self._sigma_rows(x, self._dir))
        out = self._project_t(full)
        return torch.from_numpy(out[:, 0] if squeeze else out)

    def sample(self, n=None, out=None):
        """One sample (GridFunction) or n samples (ndof x n, float64; on the field's device). ``out``: an ndof x n
        buffer filled in place and returned (:mod:`ngsrng.inplace`)."""
        if out is not None:
            return self._sample_out(n, out)
        ns = 1 if n is None else int(n)
        if self._torch:  # the noise on the device: nrand is large for the FFT construction (twice the periodic grid)
            dev = self.device
            eta = torch.randn((self.nrand, ns), generator=self.source.streams().torch(dev), device=dev, dtype=F64)
            x = device_copy(self, self._mean, dev, F64)[:, None].repeat(1, ns)
            x[device_copy(self, self.space.free, dev)] += self.apply(eta)
            if n is None:
                gf = ngs.GridFunction(self.space.fes)
                gf.vec.FV().NumPy()[:] = x[:, 0].cpu().numpy()
                return gf
            return x
        zeta = self.source.streams().numpy().standard_normal((ns, self.nrand)).T  # sample by sample: chunking-free
        x = np.repeat(self._mean[:, None], ns, axis=1)
        x[self.space.free] += self.apply(torch.from_numpy(zeta)).numpy()
        if n is None:
            gf = ngs.GridFunction(self.space.fes)
            gf.vec.FV().NumPy()[:] = x[:, 0]
            return gf
        return torch.from_numpy(x)

    def _derive(self, label, mean=None, *, eta_map=None, post_map=None, seed=None, exact=None):
        """A sampler sharing this one's set-up with an extra noise-space or FE-space map and/or a new mean: how
        :class:`ngsrng.LinearObservations` builds posteriors (``exact`` as for ``Gaussian._derive``)."""
        new = copy.copy(self)
        new._conditioning = self._conditioning + [exact if exact is not None else ("unknown", label)]
        new._maps = self._maps + ([eta_map] if eta_map is not None else [])
        new._post_maps = self._post_maps + ([post_map] if post_map is not None else [])
        new._mean = self._mean.copy() if mean is None else np.asarray(mean, dtype=float).copy()
        new._prior = self if self._prior is None else self._prior
        new.plan = {**self.plan, "posterior": self.plan.get("posterior", []) + [label]}
        new.source = inherit(seed, self)
        return new

    def condition(self, *args, **kwargs):
        """Shortcut for ``LinearObservations(self, ...).posterior(...)``: the exact posterior (see ``Gaussian``)."""
        from .posterior import LinearObservations

        data = kwargs.pop("data", None)
        seed = kwargs.pop("seed", None)
        form = kwargs.pop("form", "fe")
        return LinearObservations(self, *args, **kwargs).posterior(data=data, seed=seed, form=form)

    def __repr__(self):
        return f"StationaryMatern(length={self.length}, nu={self.nu}, construction={self.construction!r}, " \
               f"method={self.method!r}, nrand={self.nrand}, device={str(self.device)!r})"
