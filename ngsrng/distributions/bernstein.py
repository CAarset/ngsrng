"""Bernstein-Bezier form of an FE space: the bound-preserving representation of pointwise-defined fields.

Every H1 or L2 space of degree p on simplices (P_p) or on quadrilaterals / hexahedra (Q_p) has a basis of Bernstein
polynomials B_j: nonnegative, summing to one on every element, and (for H1) continuous across elements when their
coefficients are shared at the "domain points" x_j. A function sum_j b_j B_j therefore takes values in the convex
hull of its coefficients at **every** point (convex hull property). With b_j = f(x_j) for a field f with values in a
convex set (an interval, the simplex, the SPD cone), the FE function

    Q_h f = sum_j f(x_j) B_j

stays in that set everywhere, in any dimension, order and (curved) geometry: the argument is made in reference
coordinates, where the basis functions are polynomials. Q_h is a positive linear operator that reproduces constants
and (on each element, in reference coordinates) affine functions; it equals f at the mesh vertices and converges
uniformly, ||Q_h f - f||_inf <= C omega(f, h) (O(h^2) for smooth f).

Everything here is combinatorial or exact linear algebra: the domain points are identified across elements by their
integer weights on the element's global vertices; values at domain points come from values at the points of a Gauss
rule of order 2p by an exact local polynomial fit (the rule is unisolvent for P_p / Q_p); and NGSolve's (hierarchical)
coefficients of Q_h f come from an exact same-rule L2 projection. Supported: segments, triangles, tetrahedra,
quadrilaterals, hexahedra (one element type per mesh), H1 and L2 spaces of uniform order.
"""

import itertools
import math

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
import torch

from ..backend import (
    DeviceCache,
    Rebuildable,
    Scratch,
    available_memory,
    device_free_memory,
    ngs_sparse,
    parallel_ranges,
    sparse_to_torch,
)
from ..quadrature import QuadratureRule
from ..space import SpaceInfo

SIMPLICES = (ngs.ET.SEGM, ngs.ET.TRIG, ngs.ET.TET)
TENSOR = (ngs.ET.QUAD, ngs.ET.HEX)


def _reference_vertices(mesh, et):
    """Reference coordinates of the local vertices, read off NGSolve's lowest-order H1 shape functions."""
    fe = ngs.H1(mesh, order=1).GetFE(ngs.ElementId(ngs.VOL, 0))
    d = mesh.dim
    if et in SIMPLICES:
        candidates = np.vstack([np.zeros(d), np.eye(d)])
    else:
        candidates = np.array(list(itertools.product([0.0, 1.0], repeat=d)))
    verts = np.full((fe.ndof, d), np.nan)
    for c in candidates:
        shape = np.asarray(fe.CalcShape(*c, *([0.0] * (3 - d))))
        if np.isclose(shape.max(), 1.0) and np.isclose(shape.sum(), 1.0):
            verts[int(np.argmax(shape))] = c
    if np.isnan(verts).any():
        raise RuntimeError(f"could not identify the reference vertices of {et}")
    return verts


def unsupported(space: SpaceInfo):
    """Why ``space`` has no Bernstein-Bezier description, or None if it has one."""
    if space.fes.type not in ("h1ho", "l2ho"):
        return "Bernstein representation: H1 and L2 spaces only"
    if not space.single_element_type:
        return "Bernstein representation: one element type per mesh"
    et = next(iter(space.mesh.Elements(ngs.VOL))).type
    if et not in SIMPLICES + TENSOR:
        return f"Bernstein representation: {et} elements are not supported"
    return None


def default_ok(space: SpaceInfo) -> bool:
    """Whether the Bernstein representation can be chosen by default: a supported space of uniform order, checked
    cheaply (the broken space has the full local P_p / Q_p count on every element) before any set-up."""
    if unsupported(space) is not None:
        return False
    p, d = int(space.order), space.dim
    simplex = next(iter(space.mesh.Elements(ngs.VOL))).type in SIMPLICES
    nloc = math.comb(p + d, d) if simplex else (p + 1) ** d
    return space.broken.ndof == space.mesh.ne * nloc


class BernsteinForm:
    """Bernstein-Bezier description of ``space`` (see the module docstring).

    Attributes: ``npoints`` (number of global domain points, = ndof), ``evaluation`` (npoints x ndof: values of an
    FE function at the domain points), and :meth:`to_coefficients` (Bernstein coefficients -> NGSolve coefficients).
    """

    def __init__(self, space: SpaceInfo):
        mesh, fes = space.mesh, space.fes
        reason = unsupported(space)
        if reason is not None:
            raise NotImplementedError(reason)
        et = next(iter(mesh.Elements(ngs.VOL))).type
        self.space, self.p, self.simplex, self.dim = space, int(space.order), et in SIMPLICES, mesh.dim
        p, d = self.p, self.dim
        self.ref_vertices = _reference_vertices(mesh, et)
        self.rule = QuadratureRule(space, order=max(2 * p, 1))
        qpts = np.array([ip.point[:d] for ip in self.rule.rule])
        self.lattice = self._lattice()
        self.weights = self._vertex_weights()  # (nloc, nverts) integers: the domain point's weight on each vertex
        self.ref_points = self._domain_points()  # (nloc, d) reference coordinates
        self.bq = self.values(qpts)  # (nloc, nq): Bernstein basis at the rule points
        nloc, nq = self.bq.shape
        if nq < nloc:
            raise RuntimeError("quadrature rule too small to determine the local polynomials")
        # values at domain points from values at rule points: exact for P_p / Q_p (unisolvent rule)
        self.fit = self.values(self.ref_points).T @ np.linalg.pinv(self.bq.T)  # (nloc, nq)
        verts = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(ngs.VOL)])
        self.ne = len(verts)
        self._number(verts, conforming=fes.type == "h1ho")
        rows = sp.kron(sp.identity(self.ne, format="csr"), sp.csr_matrix(self.fit), format="csr")
        self.evaluation = (rows @ self.rule.evaluation_full)[self.first].tocsr()  # (npoints x ndof)
        bm = self.rule.b
        self._mass = SymmetricFactor(bm @ sp.diags(1.0 / self.rule.weights) @ bm.T)  # same-rule mass, all dofs
        self._mapped = None
        self._ops = None  # the conversion's sparse operators (made on first use; see _operators)
        self._device = DeviceCache()  # the conversion's operators on torch devices (to_coefficients of tensors)

    def __getstate__(self):  # the operators are rebuilt on first use (P is about as large as the rule's matrices)
        return {**self.__dict__, "_ops": None}

    # ---------------------------------------------------------------- reference element

    def _lattice(self):
        p, d = self.p, self.dim
        if self.simplex:
            return [a for a in itertools.product(range(p + 1), repeat=d + 1) if sum(a) == p]
        return list(itertools.product(range(p + 1), repeat=d))

    def _vertex_weights(self):
        p = self.p
        if self.simplex:
            return np.array(self.lattice, dtype=np.int64)  # alpha_i on the i-th local vertex
        corner = self.ref_vertices > 0.5
        return np.array(
            [[np.prod([ix[k] if corner[v, k] else p - ix[k] for k in range(self.dim)]) for v in range(len(corner))]
             for ix in self.lattice],
            dtype=np.int64,
        )

    def _domain_points(self):
        if self.p == 0:
            return self.ref_vertices.mean(axis=0, keepdims=True)
        w = self.weights / self.weights.sum(axis=1, keepdims=True)
        return w @ self.ref_vertices if self.simplex else np.array(self.lattice, dtype=float) / self.p

    def values(self, pts):
        """Bernstein basis functions (nloc x npts) at reference points (npts x d)."""
        p, d = self.p, self.dim
        if self.simplex:
            a = np.vstack([self.ref_vertices.T, np.ones(len(self.ref_vertices))])
            lam = np.linalg.solve(a, np.vstack([np.asarray(pts).T, np.ones(len(pts))]))  # barycentric (d+1, npts)
            return np.array([
                math.factorial(p) / math.prod(math.factorial(k) for k in al)
                * np.prod([lam[i] ** al[i] for i in range(d + 1)], axis=0)
                for al in self.lattice
            ])
        s = np.asarray(pts)
        return np.array([
            np.prod([math.comb(p, ix[k]) * s[:, k] ** ix[k] * (1 - s[:, k]) ** (p - ix[k]) for k in range(d)], axis=0)
            for ix in self.lattice
        ])

    # ---------------------------------------------------------------- global numbering

    def _number(self, verts, conforming):
        nloc = len(self.lattice)
        if conforming:
            # a domain point is identified by its (global vertex, weight) pairs, sorted by vertex number
            vid = np.repeat(verts[:, None, :], nloc, axis=1)
            w = np.broadcast_to(self.weights[None], vid.shape)
            vkey = np.where(w > 0, vid, -1)
            order = np.argsort(vkey, axis=2, kind="stable")
            k1 = np.take_along_axis(vkey, order, 2)
            k2 = np.take_along_axis(w, order, 2) * (k1 >= 0)
            keys = np.concatenate([k1, k2], axis=2).reshape(self.ne * nloc, -1)
            _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
            self.local_to_global, self.first = inv.reshape(self.ne, nloc), first
        else:
            self.local_to_global = np.arange(self.ne * nloc).reshape(self.ne, nloc)
            self.first = np.arange(self.ne * nloc)
        self.npoints = len(self.first)
        if self.npoints != self.space.ndof:
            raise NotImplementedError(
                f"the space is not the full P_p / Q_p space ({self.npoints} Bernstein functions, {self.space.ndof} "
                "dofs): variable orders or special flags are not supported"
            )

    # ---------------------------------------------------------------- maps

    def cf_values(self, cf) -> np.ndarray:
        """A CoefficientFunction at the domain points (npoints,)."""
        if self._mapped is None:
            pts = [tuple(r) + (0.0,) * (3 - self.dim) for r in self.ref_points]
            ir = ngs.IntegrationRule(pts, [1.0] * len(pts))
            self._mapped = self.space.mesh.MapToAllElements(ir, ngs.VOL)
        return np.asarray(cf(self._mapped), dtype=float).reshape(-1)[self.first]

    def to_coefficients(self, b):
        """Bernstein coefficients (npoints, k) -> NGSolve coefficients (ndof, k) of sum_j b_j B_j, and the largest
        deviation at the rule points (a measure of the rounding error of the conversion). A torch tensor is converted
        on its device in float64 (cuDSS on NVIDIA GPUs, Jacobi PCG elsewhere), giving torch tensors.

        The conversion is one sparse product with the precomposed R = B P and a mass solve: no values at the rule
        points (P b, ne nq per column: 79 per dof in 3D P1) are formed for it. Only the rounding check needs them; it
        runs in one block when that fits in a quarter of the free host memory (half of the free device memory), in
        column blocks otherwise."""
        if isinstance(b, torch.Tensor):
            return self._to_coefficients_torch(b)
        b = np.asarray(b, dtype=float)
        p, r = self._operators()
        c = self._mass.solve(np.asarray(r @ b))
        e, avail = self.rule.evaluation_full, available_memory()
        step = self._check_step(b.shape[1], None if avail is None else 0.25 * avail)
        resid = np.concatenate([np.abs(e @ c[:, j:j + step] - p @ b[:, j:j + step]).max(axis=0)
                                for j in range(0, b.shape[1], step)])
        return c, resid

    def _operators(self):
        """P (Bernstein coefficients -> values at the rule points: ne nq x npoints) and R = B P (-> the conversion's
        load: ndof x npoints), sparse, made on first use."""
        if self._ops is None:
            ne, (nloc, nq) = self.ne, self.bq.shape
            rows = (np.arange(ne)[:, None, None] * nq + np.arange(nq)[None, None, :]).repeat(nloc, 1)
            cols = np.broadcast_to(self.local_to_global[:, :, None], rows.shape)
            vals = np.broadcast_to(self.bq[None], rows.shape)
            p = sp.csr_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(ne * nq, self.npoints))
            r = (self.rule.b @ p).tocsr()
            for m in (p, r):  # sorted now: copying them to a device (sparse_to_torch sorts in place) changes nothing
                m.sort_indices()
            self._ops = (p, r)
        return self._ops

    def _check_step(self, k, budget):
        """Columns per block of the rounding check: all k when its temporaries (about four values per rule point and
        column) fit in ``budget`` bytes (None: unknown, all), fewer otherwise."""
        if budget is None:
            return max(1, k)
        per_column = 32.0 * self.ne * self.bq.shape[1]
        return max(1, min(k, int(budget // per_column)))

    def _device_ops(self, device):
        key = str(device)
        if key not in self._device:
            from ..solvers import device_spd_solver

            p, r = self._operators()
            f64 = torch.float64
            self._device[key] = (sparse_to_torch(p, device, f64), sparse_to_torch(r, device, f64),
                                 sparse_to_torch(self.rule.evaluation_full, device, f64),
                                 device_spd_solver(self._mass.a, device))
        return self._device[key]

    def _to_coefficients_torch(self, b):
        p, r, evaluation, mass = self._device_ops(b.device)
        b = b.to(torch.float64)
        c = mass.apply(r @ b)
        if b.device.type == "cuda":
            budget = 0.5 * device_free_memory(b.device)
        else:
            avail = available_memory()
            budget = None if avail is None else 0.25 * avail
        step = self._check_step(b.shape[1], budget)
        resid = torch.cat([(evaluation @ c[:, j:j + step] - p @ b[:, j:j + step]).abs().amax(dim=0)
                           for j in range(0, b.shape[1], step)])
        return c, resid


class SymmetricFactor(Rebuildable):
    """Sparse Cholesky factorisation (NGSolve) of an SPD scipy matrix, applied to many right-hand sides in parallel
    threads. SciPy's SuperLU took ~30 ms per right-hand side for the Bernstein mass matrices of a 5k-dof P2 space;
    this takes ~1 ms single-threaded."""

    _transient = ("mat", "inv")  # NGSolve objects: factorised again after unpickling

    def __init__(self, a):
        self.a = sp.csr_matrix(a)
        self._factorize()
        self._vectors = Scratch(self._new_vectors)

    def _factorize(self):
        self.mat = ngs_sparse(self.a)
        everything = ngs.BitArray(self.a.shape[0])
        everything.Set()
        self.inv = self.mat.Inverse(everything, "sparsecholesky")

    _restore = _factorize

    def _new_vectors(self):
        return self.mat.CreateColVector(), self.mat.CreateColVector()

    def solve(self, b):
        b = np.asarray(b, dtype=float)
        cols = b[:, None] if b.ndim == 1 else b
        out = np.empty((cols.shape[1], cols.shape[0]))

        def work(lo, hi):
            rhs, sol = self._vectors.get()
            for k in range(lo, hi):
                rhs.FV().NumPy()[:] = cols[:, k]
                sol.data = self.inv * rhs
                out[k] = sol.FV().NumPy()

        parallel_ranges(cols.shape[1], work)
        return out[0] if b.ndim == 1 else out.T
