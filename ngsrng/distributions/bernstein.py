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
        self.verts, self.conforming = verts, fes.type == "h1ho"  # element vertex numbers (ne x vertices per element)
        self._number(verts, conforming=self.conforming)
        rows = sp.kron(sp.identity(self.ne, format="csr"), sp.csr_matrix(self.fit), format="csr")
        self.evaluation = (rows @ self.rule.evaluation_full)[self.first].tocsr()  # (npoints x ndof)
        bm = self.rule.b
        self._mass = SymmetricFactor(bm @ sp.diags(1.0 / self.rule.weights) @ bm.T)  # same-rule mass, all dofs
        self._mapped = None
        self._ops = None  # the conversion's sparse operators (made on first use; see _operators)
        self._device = DeviceCache()  # the conversion's operators on torch devices (to_coefficients of tensors)
        self._topology = None  # per dof its node, per support the domain points (dependence; made on first use)

    def __getstate__(self):  # the operators are rebuilt on first use (P is about as large as the rule's matrices)
        return {**self.__dict__, "_ops": None}

    def restricted(self, dofs) -> "BernsteinPart":
        """The part of this form that the NGSolve coefficients ``dofs`` depend on (e.g. the DoFs of a trace): see
        :class:`BernsteinPart`. Not kept here (the part refers to the form: no reference cycle); callers cache it."""
        return BernsteinPart(self, np.unique(np.asarray(dofs, dtype=np.int64)))

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

    # ---------------------------------------------------------------- which coefficients depend on which

    def dependence(self, dofs) -> sp.csr_matrix:
        """Which Bernstein coefficients the NGSolve coefficients ``dofs`` depend on: a 0/1 matrix (len(dofs) x
        npoints), exact (from the mesh topology, not from rounding).

        H1: NGSolve's basis is hierarchical, a coefficient belongs to a mesh node (vertex, edge, face, cell) and is a
        linear function of the FE function on that node's closure (the other basis functions vanish there, or belong
        to the closure's own nodes). On that closure only the Bernstein functions of the domain points in it do not
        vanish, so the coefficient depends exactly on those points: the points whose vertices (those with a positive
        weight) are vertices of the node. L2: a coefficient depends on the domain points of its element."""
        nodes, by_support = self._nodes()
        dofs = np.asarray(dofs, dtype=np.int64).ravel()
        rows, cols, found = [], [], {}
        for r, d in enumerate(dofs):
            node = nodes[d]
            pts = found.get(node)
            if pts is None:
                if node[0] == "element":
                    pts = self.local_to_global[node[1]]
                else:
                    subsets = (w for k in range(1, len(node) + 1) for w in itertools.combinations(node, k))
                    pts = np.unique(np.concatenate([by_support.get(w, np.zeros(0, np.int64)) for w in subsets]))
                found[node] = pts
            rows.append(np.full(len(pts), r))
            cols.append(pts)
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        return sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(dofs.size, self.npoints))

    def _nodes(self):
        """(per dof: the sorted vertex numbers of its node, or ("element", number) for L2; per sorted vertex tuple:
        the domain points with exactly those vertices), made on first use."""
        if self._topology is None:
            mesh, fes = self.space.mesh, self.space.fes
            nodes = [None] * fes.ndof
            if self.conforming:
                for nt in (ngs.VERTEX, ngs.EDGE, ngs.FACE, ngs.CELL):
                    for i in range(mesh.nnodes(nt)):
                        nid = ngs.NodeId(nt, i)
                        dofs = fes.GetDofNrs(nid)
                        if not dofs:
                            continue
                        verts = (i,) if nt == ngs.VERTEX else tuple(sorted(v.nr for v in mesh[nid].vertices))
                        for d in dofs:
                            if d >= 0:
                                nodes[d] = verts
                nloc = len(self.lattice)
                positive = self.weights > 0  # (nloc, vertices per element): the vertices of each local point
                support = {}
                for j, flat in enumerate(self.first):
                    el, loc = divmod(int(flat), nloc)
                    support.setdefault(tuple(sorted(int(v) for v in self.verts[el][positive[loc]])), []).append(j)
                by_support = {k: np.asarray(v, dtype=np.int64) for k, v in support.items()}
            else:
                for el in mesh.Elements(ngs.VOL):
                    for d in fes.GetDofNrs(el):
                        if d >= 0:
                            nodes[d] = ("element", el.nr)
                by_support = {}
            if any(n is None for n in nodes):
                raise NotImplementedError("some dofs belong to no mesh node: an unsupported space for dense samplers")
            self._topology = (nodes, by_support)
        return self._topology

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


class BernsteinPart:
    """The domain points of a :class:`BernsteinForm` that the NGSolve coefficients ``dofs`` depend on, with the
    form's interface on those points only (a field built on it lives at them).

    The points are ``rows = form.dependence(dofs)``: exact, from the mesh topology. So a function sum_j b_j B_j has
    at ``dofs`` the coefficients of sum_{j in rows} b_j B_j, whatever the b_j at the other points: for the DoFs of a
    trace (``bnd=``), ``rows`` are the domain points on the boundary, and a pointwise field restricted to that
    trace needs its values (and its standardisation) only there. :meth:`to_coefficients` gives the coefficients at
    ``dofs`` (zero elsewhere, as ``bnd=`` keeps them) element by element, at the cost of the part, not of the space.
    """

    def __init__(self, form: BernsteinForm, dofs):
        self.form, self.dofs, self.space = form, np.asarray(dofs, dtype=np.int64), form.space
        self.rows = np.flatnonzero(np.asarray(form.dependence(self.dofs).sum(axis=0)).ravel())
        self.npoints = int(self.rows.size)
        self.evaluation = form.evaluation[self.rows].tocsr()  # (npoints x ndof): rows of the form's evaluation
        self._map = None  # (T, resid): the conversion, made on first use (_local_map)
        self._device = DeviceCache()

    @property
    def full_npoints(self) -> int:
        """The number of domain points of the whole form."""
        return self.form.npoints

    def at_rows(self, values):
        """Values at every domain point of the form (full_npoints, ...) restricted to this part's points."""
        return values[self.rows]

    def dependence(self, dofs) -> sp.csr_matrix:
        """As :meth:`BernsteinForm.dependence`, on this part's points (len(dofs) x npoints); dofs outside the part's
        depend on nothing (their coefficients are zero)."""
        dofs = np.asarray(dofs, dtype=np.int64).ravel()
        dep = self.form.dependence(dofs)[:, self.rows]
        dep = (sp.diags(np.isin(dofs, self.dofs).astype(float)) @ dep).tocsr()
        dep.eliminate_zeros()
        return dep

    def cf_values(self, cf) -> np.ndarray:
        return self.form.cf_values(cf)[self.rows]

    def _local_map(self):
        """(T, resid), made on first use: T (len(dofs) x npoints, CSR) maps the Bernstein coefficients at this part's
        points to the NGSolve coefficients at ``dofs``; resid (npoints,) bounds the conversion's rounding per unit
        Bernstein coefficient (at the rule points of the elements used).

        A coefficient of an FE function is that of its restriction to any element containing the coefficient's node,
        and there the function is the polynomial sum_j b_j B_j: its local coefficients are the exact least-squares fit
        of its values at the element's rule points (unisolvent for P_p / Q_p; the curved geometry enters through the
        basis values at the points, as in the global conversion). One element per node of ``dofs``: the set-up costs
        the boundary, not the space. Entries outside ``form.dependence`` (exact zeros, computed as rounding) are
        dropped, and their size is added to resid."""
        if self._map is not None:
            return self._map
        form = self.form
        fes = form.space.fes
        nodes, _ = form._nodes()
        e = form.rule.evaluation_full
        nq = form.bq.shape[1]
        pos = np.full(form.npoints, -1)
        pos[self.rows] = np.arange(self.npoints)
        dep = form.dependence(self.dofs)  # (len(dofs) x npoints of the form), exact
        verts = form.verts
        flat = np.argsort(verts.ravel(), kind="stable")
        starts = np.searchsorted(verts.ravel()[flat], np.arange(verts.max() + 2))
        by_element, owner = {}, {}
        for r, d in enumerate(self.dofs):
            node = nodes[d]
            if node not in owner:
                if node[0] == "element":
                    owner[node] = node[1]
                else:  # an element having every vertex of the node
                    cands = flat[starts[node[0]]:starts[node[0] + 1]] // verts.shape[1]
                    owner[node] = next(int(t) for t in cands if set(node) <= set(verts[t].tolist()))
            by_element.setdefault(owner[node], []).append(r)
        rows, cols, vals = [], [], []
        resid = np.zeros(self.npoints)
        for t, members in by_element.items():
            dn = np.asarray(fes.GetDofNrs(ngs.ElementId(ngs.VOL, t)), dtype=np.int64)
            if (dn < 0).any():
                raise NotImplementedError("a Bernstein part needs every element dof to be a dof of the space")
            phi = e[t * nq:(t + 1) * nq][:, dn].toarray()  # (nq x nloc): the element's basis at its rule points
            conv = np.linalg.lstsq(phi, form.bq.T, rcond=None)[0]  # local Bernstein -> local NGSolve coefficients
            glob = form.local_to_global[t]
            inside = pos[glob] >= 0
            resid[pos[glob[inside]]] = np.maximum(resid[pos[glob[inside]]],
                                                  np.abs(phi @ conv - form.bq.T).max(axis=0)[inside])
            scale = np.abs(phi).max()
            local = {int(g): k for k, g in enumerate(dn)}
            for r in members:
                row = conv[local[int(self.dofs[r])]]
                keep = np.isin(glob, dep.indices[dep.indptr[r]:dep.indptr[r + 1]])
                rows.append(np.full(int(keep.sum()), r))
                cols.append(pos[glob[keep]])
                vals.append(row[keep])
                drop = ~keep & inside
                resid[pos[glob[drop]]] += scale * np.abs(row[drop])
        t_map = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(self.dofs.size, self.npoints))
        self._map = (t_map, resid)
        return self._map

    def to_coefficients(self, b):
        """Bernstein coefficients at this part's points (npoints, k) -> NGSolve coefficients (ndof, k): those of
        sum_{j in rows} b_j B_j at ``dofs``, zero elsewhere; and per column a bound of the conversion's rounding,
        sum_j |b_j| resid_j (the role of the measured deviation in :meth:`BernsteinForm.to_coefficients`)."""
        t_map, resid = self._local_map()
        if isinstance(b, torch.Tensor):
            key = str(b.device)
            if key not in self._device:
                self._device[key] = (sparse_to_torch(t_map, b.device, torch.float64),
                                     torch.from_numpy(resid).to(b.device),
                                     torch.from_numpy(self.dofs).to(b.device))
            tt, rt, dofs = self._device[key]
            b = b.to(torch.float64)
            c = torch.zeros((self.space.ndof, b.shape[1]), dtype=torch.float64, device=b.device)
            c[dofs] = tt @ b
            return c, rt @ b.abs()
        b = np.asarray(b, dtype=float)
        c = np.zeros((self.space.ndof, b.shape[1]))
        c[self.dofs] = t_map @ b
        return c, resid @ np.abs(b)


class SymmetricFactor(Rebuildable):
    """Sparse Cholesky factorisation (NGSolve) of an SPD scipy matrix, applied to many right-hand sides in parallel
    threads. SciPy's SuperLU took ~30 ms per right-hand side for the Bernstein mass matrices of a 5k-dof P2 space;
    this takes ~1 ms single-threaded."""

    _transient = ("mat", "inv")  # NGSolve objects: factorised again after unpickling

    def __init__(self, a):
        self.a = sp.csr_matrix(a)
        self._factorize()
        self._vectors = Scratch(type(self)._new_vectors)

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
            rhs, sol = self._vectors.get(self)
            for k in range(lo, hi):
                rhs.FV().NumPy()[:] = cols[:, k]
                sol.data = self.inv * rhs
                out[k] = sol.FV().NumPy()

        parallel_ranges(cols.shape[1], work)
        return out[0] if b.ndim == 1 else out.T
