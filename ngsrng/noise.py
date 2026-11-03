"""Samplers for z ~ N(0, A) on the free dofs, where A is an assembled SPD (or SPSD) matrix.

For A = M (mass) this is FE white noise; for A = K (a locally PSD stiffness) it is "precision-shaped" noise.
Every sampler is a linear map z = H eta with eta ~ N(0, I_nrand), so the exact covariance H H^T of any sampler is
computable for testing. Two engines implement it:

* torch (``setup``/``apply``): free-dof tensors of shape (nfree, ns), used on CUDA (and on CPU for testing);
* NGSolve (``ngs_setup``/``ngs_apply``): one sample at a time on full-length NGSolve vectors (Dirichlet entries
  zero), with multithreaded NGSolve sparse kernels. This is the fast CPU engine.
"""

import warnings
from collections import deque

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .backend import Rebuildable, Scratch, available_memory, ngs_sparse, sparse_to_torch, spmm
from .space import SpaceInfo, assemble, mass_form


class NoiseSampler(Rebuildable):
    """White-noise sampler. Its NGSolve objects (``_ngs_*``, made by ``ngs_setup``) do not pickle: they are made
    again after unpickling."""

    name = "abstract"
    exact = True
    nrand: int
    nfree: int

    def setup(self, device, dtype):
        self.device, self.dtype = device, dtype
        return self

    def apply(self, eta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_t(self, z: torch.Tensor) -> torch.Tensor:
        """Adjoint map H^T: free-dof vectors (nfree x k) -> noise space (nrand x k). Torch engine (any device)."""
        raise NotImplementedError

    def ngs_setup(self, space: SpaceInfo):
        """Default for samplers that store a sparse factor ``self.h`` (nfree x nrand)."""
        self.ngs_space = space
        self._ngs_h = ngs_sparse(self.h, rows=space.free, nrows=space.ndof)
        return self

    def _restore(self):
        if getattr(self, "ngs_space", None) is not None:
            self.ngs_setup(self.ngs_space)

    def ngs_apply(self, eta, out):
        """out (ndof) = H eta (nrand), as NGSolve BaseVectors."""
        out.data = self._ngs_h * eta

    def factor(self) -> sp.csr_matrix:
        """H as a scipy matrix (nfree x nrand). Intended for testing on small problems."""
        eye = torch.eye(self.nrand, device=self.device, dtype=torch.float64)
        old = self.dtype
        self.setup(self.device, torch.float64)
        h = self.apply(eye).cpu().numpy()
        self.setup(self.device, old)
        return sp.csr_matrix(h)

    def info(self) -> dict:
        return {"noise": self.name, "nrand": self.nrand, "exact": self.exact}

    def accuracy(self) -> str:
        """How far Cov(z) is from A, in words (approximate samplers state their error)."""
        return "exact (Cov = A)"


class DiagonalNoise(NoiseSampler):
    """A diagonal (e.g. L2 mass on affine meshes): z = sqrt(diag A) * eta."""

    name = "diagonal"

    def __init__(self, a_free: sp.spmatrix):
        self.d = np.sqrt(a_free.diagonal())
        self.nfree = self.nrand = len(self.d)
        self.h = sp.diags(self.d).tocsr()

    def setup(self, device, dtype):
        super().setup(device, dtype)
        self._d = torch.from_numpy(self.d).to(device=device, dtype=dtype)
        return self

    def apply(self, eta):
        return self._d[:, None] * eta if eta.ndim == 2 else self._d * eta

    apply_t = apply


class LumpedNoise(DiagonalNoise):
    """Row-sum lumping. Approximate: Cov = diag(rowsum A) != A."""

    name = "lumped"
    exact = False

    def __init__(self, a_free: sp.spmatrix):
        rs = np.asarray(a_free.sum(axis=1)).ravel()
        if (rs <= 0).any():
            raise ValueError("row-sum lumping yields non-positive entries for this space")
        self.d = np.sqrt(rs)
        self.nfree = self.nrand = len(self.d)
        self.h = sp.diags(self.d).tocsr()
        # max |Cov - A| / max |A|, Cov = diag(row sums): the off-diagonal of A and the diagonal shift
        diff = abs(a_free - sp.diags(rs))
        self.rel_error = float(diff.max() / abs(a_free).max()) if diff.nnz else 0.0

    def accuracy(self):
        return f"Cov = diag(row sums of A), relative covariance error {self.rel_error:.1e} (max |Cov - A| / max |A|)"


class SparseFactorNoise(NoiseSampler):
    """z = H eta for an explicitly stored sparse factor H."""

    def __init__(self, h: sp.spmatrix, name: str):
        self.h = h.tocsr()
        self.name = name
        self.nfree, self.nrand = self.h.shape

    def setup(self, device, dtype):
        super().setup(device, dtype)
        self._h = sparse_to_torch(self.h, device, dtype)
        self._ht = sparse_to_torch(self.h.T.tocsr(), device, dtype)
        return self

    def apply(self, eta):
        return spmm(self._h, eta)

    def apply_t(self, z):
        return spmm(self._ht, z)

    def factor(self):
        return self.h


def factor_error(h: sp.spmatrix, a: sp.spmatrix, probes: int = 4) -> float:
    """Relative error of H H^T against A, without forming H H^T: the larger of the diagonal error (exact,
    max |diag(H H^T - A)| / max |diag A|) and the probe estimate ||(H H^T - A) X||_F / ||A X||_F of the relative
    Frobenius error, X Gaussian (fixed seed). Costs 2 ``probes`` products with H."""
    dh = np.asarray(h.multiply(h).sum(axis=1)).ravel()
    da = a.diagonal()
    diag = np.abs(dh - da).max() / np.abs(da).max()
    x = np.random.default_rng(0).standard_normal((a.shape[0], probes))
    ax = a @ x
    full = np.linalg.norm(h @ (h.T @ x) - ax) / np.linalg.norm(ax)
    return float(max(diag, full))


def dense_cholesky_noise(a_free: sp.spmatrix) -> SparseFactorNoise:
    """Baseline for small problems: lower Cholesky of the full matrix. O(n^2) memory (peak ~24 n^2 bytes: the dense
    matrix, its factor and the sparse copy of the factor); raises MemoryError up front if that exceeds the available
    memory. Never chosen automatically."""
    n = a_free.shape[0]
    need, avail = 24.0 * n**2, available_memory()
    if avail is not None and need > avail:
        raise MemoryError(f"dense Cholesky of a {n} x {n} matrix needs ~{need / 2**30:.1f} GiB, "
                          f"{avail / 2**30:.1f} GiB available")
    return SparseFactorNoise(sp.csr_matrix(np.linalg.cholesky(a_free.toarray())), "dense-cholesky")


def quadrature_noise(space: SpaceInfo, order: int | None = None, max_nnz: float = 4e8,
                     rtol: float = 1e-8) -> SparseFactorNoise:
    """White noise sampled at quadrature points: b_i = sum_q sqrt(w_q) phi_i(x_q) xi_q (see ``ngsrng.quadrature``).

    Exact (Cov = M) whenever the rule integrates phi_i phi_j exactly: order >= p on affine elements (the rule has degree
    2 order; see ``ngsrng.quadrature``), and on curved elements only order = p, the rule NGSolve assembles M with. The
    default (``order=None``) is p, which is exact on affine, mapped and curved meshes alike and has the fewest points
    (6 instead of 25 per P2 triangle at 2p); should it ever miss M, 2p is tried before giving up. Every rule is checked
    against the assembled M (see :func:`factor_error`), and one that misses it by more than ``rtol`` raises: a
    covariance error of 1e-3 is invisible in samples. Raises MemoryError before assembly if the factor would exceed
    ``max_nnz`` nonzeros (~12 bytes each).
    """
    from .quadrature import QuadratureRule

    errors = {}
    for o in [space.order, 2 * space.order] if order is None else [order]:
        if o in errors:  # p = 0
            continue
        rule = QuadratureRule(space, o, max_nnz)
        h = rule.b[space.free] @ sp.diags(1.0 / np.sqrt(rule.weights))
        errors[o] = factor_error(h, space.mass_free)
        if errors[o] <= rtol:
            break
    else:
        tried = ", ".join(f"{err:.1e} at order {o}" for o, err in errors.items())
        raise RuntimeError(
            f"quadrature noise is not exact for {space.kind}(order={space.order}): relative covariance error {tried} "
            f"(> {rtol:.0e}). The rule must integrate products of basis functions exactly: order >= {space.order} on "
            f"affine elements, order = {space.order} on curved ones"
        )
    noise = SparseFactorNoise(h, "quadrature")
    noise.quad_order = rule.order
    noise.element_of_column = np.repeat(np.arange(space.mesh.ne), rule.npoints // space.mesh.ne)
    return noise


class ChebyshevNoise(NoiseSampler):
    """z ~= D^{1/2} (D^{-1/2} A D^{-1/2})^{1/2} eta, the symmetric root applied by a Chebyshev polynomial.

    Approximate but linear in eta. Needs only mat-vecs with A. The degree is chosen so that the relative error of the
    scalar root is below ``tol`` on the estimated spectrum of the Jacobi-scaled matrix.
    """

    name = "chebyshev"
    exact = False

    def __init__(self, a_free: sp.spmatrix, tol=1e-6, max_degree=400):
        a = a_free.tocsr()
        self.nfree = self.nrand = a.shape[0]
        dinv = 1.0 / np.sqrt(a.diagonal())
        self.dsqrt = 1.0 / dinv
        self.at = sp.diags(dinv) @ a @ sp.diags(dinv)
        hi = float(np.abs(self.at).sum(axis=1).max())
        # Lower spectral bound: a few LOBPCG steps overestimate lambda_min, so keep a wide safety margin.
        # Underestimating only raises the degree; overestimating would break the approximation.
        if self.nfree > 8:
            x0 = np.random.default_rng(0).standard_normal((self.nfree, 2))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lam = spla.lobpcg(self.at, x0, largest=False, maxiter=60, tol=1e-3)[0]
            lo = 0.5 * float(lam.min())
        else:
            lo = 0.5 * float(np.linalg.eigvalsh(self.at.toarray()).min())
        self.lo, self.hi = lo, hi
        self.degree, self.coef = self._fit(tol, max_degree)
        self.tol = tol

    def _fit(self, tol, max_degree):
        x = np.cos(np.linspace(0, np.pi, 2000))
        lam = 0.5 * (self.hi - self.lo) * x + 0.5 * (self.hi + self.lo)

        def attempt(deg):
            cheb = np.polynomial.chebyshev.Chebyshev.interpolate(
                lambda t: np.sqrt(0.5 * (self.hi - self.lo) * t + 0.5 * (self.hi + self.lo)), deg
            )
            return np.max(np.abs(cheb(x) - np.sqrt(lam)) / np.sqrt(lam)) < tol, cheb.coef

        hi_deg = 2
        while hi_deg < max_degree and not attempt(hi_deg)[0]:
            hi_deg = min(2 * hi_deg, max_degree)
        lo_deg = max(2, hi_deg // 2)
        while lo_deg < hi_deg:
            mid = (lo_deg + hi_deg) // 2
            if attempt(mid)[0]:
                hi_deg = mid
            else:
                lo_deg = mid + 1
        return hi_deg, attempt(hi_deg)[1]

    def setup(self, device, dtype):
        super().setup(device, dtype)
        self._a = sparse_to_torch(self.at, device, dtype)
        self._d = torch.from_numpy(self.dsqrt).to(device=device, dtype=dtype)
        return self

    def _poly(self, x):
        alpha, beta = 2.0 / (self.hi - self.lo), (self.hi + self.lo) / (self.hi - self.lo)

        def t_op(y):
            return alpha * (self._a @ y) - beta * y

        b1 = torch.zeros_like(x)
        b2 = torch.zeros_like(x)
        for c in self.coef[:0:-1]:
            b1, b2 = 2 * t_op(b1) - b2 + c * x, b1
        return t_op(b1) - b2 + self.coef[0] * x

    def apply(self, eta):
        squeeze = eta.ndim == 1
        x = eta[:, None] if squeeze else eta
        y = self._d[:, None] * self._poly(x)
        return y[:, 0] if squeeze else y

    def apply_t(self, z):
        squeeze = z.ndim == 1
        x = z[:, None] if squeeze else z
        y = self._poly(self._d[:, None] * x)
        return y[:, 0] if squeeze else y

    def ngs_setup(self, space: SpaceInfo):
        self.ngs_space = space
        self._ngs_a = ngs_sparse(self.at)
        self._ngs_out = ngs_sparse(sp.diags(self.dsqrt).tocsr(), rows=space.free, nrows=space.ndof)
        self._ngs_tmp = Scratch(self._new_tmp)  # per thread
        return self

    def _new_tmp(self):
        return [self._ngs_a.CreateColVector() for _ in range(4)]

    def ngs_apply(self, eta, out):
        alpha, beta = 2.0 / (self.hi - self.lo), (self.hi + self.lo) / (self.hi - self.lo)
        b1, b2, b0, av = self._ngs_tmp.get()
        b1[:] = 0.0
        b2[:] = 0.0
        # Coefficients as Python floats: numpy scalars hijack ``scalar * BaseVector`` and break the expression.
        for c in map(float, self.coef[:0:-1]):
            av.data = self._ngs_a * b1
            b0.data = (2 * alpha) * av - (2 * beta) * b1 - b2 + c * eta
            b2, b1, b0 = b1, b0, b2
        av.data = self._ngs_a * b1
        b0.data = alpha * av - beta * b1 - b2 + float(self.coef[0]) * eta
        out.data = self._ngs_out * b0

    def info(self):
        return {**super().info(), "degree": self.degree, "spectrum": (self.lo, self.hi)}

    def accuracy(self):
        return (f"Cov = A up to a relative error of ~{2 * self.tol:.0e} (degree-{self.degree} Chebyshev square root, "
                f"relative error < {self.tol:.0e} on the estimated spectrum)")


class LocalGramNoise(NoiseSampler):
    """Element-wise noise for any assembled matrix A = sum_e P_e^T A_e P_e with PSD element matrices A_e.

    z ~ N(0, A_ff) exactly (Croci et al. 2018 for A = M; any locally PSD precision in general). The local roots
    live in the element-local ("broken") basis, y = blockdiag(A_e^{1/2}) eta, and z = E_f^T y with the embedding
    E of the space into its broken counterpart (the element dof map), so Cov(z) = E_f^T A_broken E_f = A_ff.

    Elements touching Dirichlet dofs instead factor their reduced Gram matrix E_ef^T A_e E_ef directly on their
    free dofs, so no random numbers are spent on Dirichlet dofs.

    Interior elements whose local matrix equals s_e D_e R_k D_e (D_e diagonal +-1, s_e > 0) for one of a few
    reference matrices R_k (orientation classes; the mass matrix on affine meshes) share that class's Cholesky
    factor: one GEMM per class and O(#elements) storage. Remaining elements store individual roots.
    """

    name = "local-gram"

    def __init__(self, space: SpaceInfo, form=mass_form, reference=True, rtol=1e-10, max_classes=64):
        self.space = space
        self.nfree = space.nfree
        a_b = assemble(space.broken, form).tocsr()
        e = space.embedding.tocsr()
        nb = e.shape[0]
        freepos = np.full(space.ndof, -1, dtype=np.int64)
        freepos[space.free] = np.arange(space.nfree)
        ecoo = e.tocoo()
        dir_row = np.zeros(nb, dtype=bool)
        np.logical_or.at(dir_row, ecoo.row, freepos[ecoo.col] < 0)

        starts, sizes = _block_structure(a_b, space)
        ref_parts, broken_roots, boundary = [], [], []
        ref_elems = []
        in_y = np.zeros(nb, dtype=bool)
        for n in np.unique(sizes):
            elem = np.flatnonzero(sizes == n)  # blocks are ordered by element
            s = starts[sizes == n]
            blocks = _extract_blocks(a_b, s, n)
            rows = s[:, None] + np.arange(n)
            interior = ~dir_row[rows].any(axis=1)
            in_y[rows[interior].ravel()] = True
            todo = np.flatnonzero(interior)
            if reference:
                for _ in range(max_classes):
                    if len(todo) == 0:
                        break
                    ok, scale, dsign, l_ref = _reference_fit(blocks[todo], rtol)
                    if not ok.any() or (ok.sum() == 1 and len(todo) > 1):
                        break
                    ref_parts.append((l_ref, np.sqrt(scale[ok])[:, None] * dsign[ok], rows[todo[ok]]))
                    ref_elems.append(elem[todo[ok]])
                    todo = todo[~ok]
            if len(todo):
                broken_roots.append((blocks[todo], rows[todo], elem[todo]))
            bnd = np.flatnonzero(~interior)
            if len(bnd):
                boundary.extend(_reduced_grams(blocks[bnd], rows[bnd], elem[bnd], e, freepos))

        self.ref_parts = ref_parts
        self.n_classes = len(ref_parts)
        self.nref = sum(p[2].size for p in ref_parts)
        self.h_broken, broken_cols = _root_factor(broken_roots, nb, rtol)
        self.h_boundary, boundary_cols = _root_factor(_group_by_size(boundary), self.nfree, rtol)
        # element of every white-noise column (for element-wise operations such as type-G variance mixing)
        self.element_of_column = np.concatenate(
            [np.tile(el, p[1].shape[1]) for el, p in zip(ref_elems, ref_parts)] + [broken_cols, boundary_cols]
        ).astype(np.int64)
        self.eft = (sp.diags(in_y.astype(float)) @ e)[:, space.free].T.tocsr()
        self.eft.eliminate_zeros()
        self.nbroken = nb
        self.nrand = self.nref + self.h_broken.shape[1] + self.h_boundary.shape[1]
        self.n_ref_elements = sum(len(p[1]) for p in ref_parts)
        self.n_batched_elements = sum(len(b[0]) for b in broken_roots)
        self.n_boundary_elements = len(boundary)

    def setup(self, device, dtype):
        super().setup(device, dtype)
        self._ref = [
            (
                torch.from_numpy(l).to(device=device, dtype=dtype),
                torch.from_numpy(sd).to(device=device, dtype=dtype),
                torch.from_numpy(rows.reshape(-1)).to(device=device),
            )
            for l, sd, rows in self.ref_parts
        ]
        self._hb = sparse_to_torch(self.h_broken, device, dtype) if self.h_broken.nnz else None
        self._hg = sparse_to_torch(self.h_boundary, device, dtype) if self.h_boundary.nnz else None
        self._eft = sparse_to_torch(self.eft, device, dtype)
        for attr in ("_eftt", "_hbt", "_hgt"):
            self.__dict__.pop(attr, None)
        return self

    def apply(self, eta):
        squeeze = eta.ndim == 1
        x = eta[:, None] if squeeze else eta
        ns = x.shape[1]
        y = torch.zeros(self.nbroken, ns, device=x.device, dtype=x.dtype)
        off = 0
        for l, sd, idx in self._ref:
            ne, n = sd.shape
            # One 2D GEMM, never batched matmul: NGSolve's bundled OpenBLAS exports cblas_*gemm_batch symbols that
            # shadow MKL's and crash torch.bmm on CPU when ngsolve is imported first.
            v = (l @ x[off : off + ne * n].reshape(n, ne * ns)).reshape(n, ne, ns).permute(1, 0, 2)
            y.index_copy_(0, idx, (v * sd[:, :, None]).reshape(ne * n, ns))
            off += ne * n
        if self._hb is not None:
            m = self.h_broken.shape[1]
            y += self._hb @ x[off : off + m]
            off += m
        z = self._eft @ y
        if self._hg is not None:
            z += self._hg @ x[off:]
        return z[:, 0] if squeeze else z

    def apply_t(self, z):
        squeeze = z.ndim == 1
        zz = z[:, None] if squeeze else z
        ns = zz.shape[1]
        if not hasattr(self, "_eftt"):
            self._eftt = sparse_to_torch(self.eft.T.tocsr(), self.device, self.dtype)
            self._hbt = sparse_to_torch(self.h_broken.T.tocsr(), self.device, self.dtype) if self.h_broken.nnz else None
            self._hgt = (
                sparse_to_torch(self.h_boundary.T.tocsr(), self.device, self.dtype) if self.h_boundary.nnz else None
            )
        y = self._eftt @ zz
        parts = []
        for l, sd, idx in self._ref:
            ne, n = sd.shape
            v = (y.index_select(0, idx).reshape(ne, n, ns) * sd[:, :, None]).permute(1, 0, 2).reshape(n, ne * ns)
            parts.append((l.T @ v).reshape(n * ne, ns))
        if self._hbt is not None:
            parts.append(self._hbt @ y)
        if self._hgt is not None:
            parts.append(self._hgt @ zz)
        x = torch.cat(parts, dim=0)
        return x[:, 0] if squeeze else x

    def explicit(self) -> SparseFactorNoise:
        """The same linear map as one stored sparse factor H = [E_f^T blockdiag(L_e) | H_boundary].

        Faster for small local blocks (one SpMV instead of GEMM + scatter); needs ~n^2/2 nonzeros per element.
        The column order of H matches ``apply``, so both give identical samples for the same eta.
        """
        cols, rows, vals = [], [], []
        off = 0
        for l, sd, r in self.ref_parts:
            ne, n = sd.shape
            i, j = np.nonzero(l)
            rows.append(r[:, i].ravel())
            cols.append((off + j[None, :] * ne + np.arange(ne)[:, None]).ravel())
            vals.append((sd[:, i] * l[i, j][None, :]).ravel())
            off += ne * n
        lb = (
            sp.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.nbroken, off)
            )
            if rows
            else sp.csr_matrix((self.nbroken, 0))
        )
        h = sp.hstack([self.eft @ sp.hstack([lb, self.h_broken]), self.h_boundary]).tocsr()
        out = SparseFactorNoise(h, "local-gram-explicit")
        out.element_of_column = self.element_of_column
        return out

    def ngs_setup(self, space: SpaceInfo):
        self.ngs_space = space
        self._ngs_ref = [
            (l, np.ascontiguousarray(sd.T), np.ascontiguousarray(rows.T)) for l, sd, rows in self.ref_parts
        ]
        self._ngs_hb = ngs_sparse(self.h_broken) if self.h_broken.nnz else None
        self._ngs_hg = ngs_sparse(self.h_boundary, rows=space.free, nrows=space.ndof) if self.h_boundary.nnz else None
        self._ngs_eft = ngs_sparse(self.eft, rows=space.free, nrows=space.ndof)
        self._ngs_y = Scratch(self._new_broken_vector)  # per thread
        return self

    def _new_broken_vector(self):
        return ngs.la.BaseVector(self.nbroken)

    def ngs_apply(self, eta, out):
        e, y = eta.FV().NumPy(), self._ngs_y.get()
        y[:] = 0.0
        yn = y.FV().NumPy()
        off = 0
        for l, sdt, rows_t in self._ngs_ref:
            n, ne = sdt.shape
            yn[rows_t] = (l @ e[off : off + ne * n].reshape(n, ne)) * sdt
            off += ne * n
        if self._ngs_hb is not None:
            m = self.h_broken.shape[1]
            y.data += self._ngs_hb * eta.Range(off, off + m)
            off += m
        out.data = self._ngs_eft * y
        if self._ngs_hg is not None:
            out.data += self._ngs_hg * eta.Range(off, self.nrand)

    def info(self):
        return {
            **super().info(),
            "reference_classes": self.n_classes,
            "reference_elements": self.n_ref_elements,
            "batched_elements": self.n_batched_elements,
            "boundary_elements": self.n_boundary_elements,
        }


def _reduced_grams(blocks, rows, elems, e: sp.csr_matrix, freepos):
    """For elements touching Dirichlet dofs: (E_ef^T A_e E_ef, free indices, element) per element."""
    out = []
    for blk, r, el in zip(blocks, rows, elems):
        sub = e[r].tocoo()
        fp = freepos[sub.col]
        keep = fp >= 0
        if not keep.any():
            continue
        cols, local = np.unique(fp[keep], return_inverse=True)
        ef = np.zeros((len(r), len(cols)))
        np.add.at(ef, (sub.row[keep], local), sub.data[keep])
        out.append((ef.T @ blk @ ef, cols, el))
    return out


def _group_by_size(items):
    groups = {}
    for g, idx, el in items:
        groups.setdefault(len(idx), ([], [], []))
        groups[len(idx)][0].append(g)
        groups[len(idx)][1].append(idx)
        groups[len(idx)][2].append(el)
    return [(np.array(gs), np.array(ids), np.array(els)) for gs, ids, els in groups.values()]


def _block_structure(a: sp.csr_matrix, space: SpaceInfo):
    """Element blocks (start, size) of the broken space; checks that ``a`` has no numerically nonzero coupling."""
    n, ne = a.shape[0], space.mesh.ne
    if n % ne == 0 and space.single_element_type:
        starts = np.arange(ne) * (n // ne)
    else:
        starts = np.array([min(el.dofs) for el in space.broken.Elements(ngs.VOL)])
    order = np.argsort(starts)
    starts = starts[order]
    sizes = np.diff(np.r_[starts, n])
    blk = np.repeat(np.arange(len(starts)), sizes)
    coo = a.tocoo()
    big = np.abs(coo.data) > 1e-14 * np.abs(coo.data).max()
    if (blk[coo.row[big]] != blk[coo.col[big]]).any():
        raise RuntimeError("form couples different elements (e.g. facet terms); no element-local factorisation")
    return starts, sizes


def _extract_blocks(a: sp.csr_matrix, starts: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(starts), n, n))
    rows = (starts[:, None] + np.arange(n)).ravel()
    sub = a[rows].tocoo()
    blk = sub.row // n
    local = sub.col - starts[blk]
    inside = (local >= 0) & (local < n)
    out[blk[inside], sub.row[inside] % n, local[inside]] = sub.data[inside]
    return out


def _reference_fit(blocks: np.ndarray, rtol: float):
    """Find s_e, D_e with blocks[e] = s_e D_e R D_e for R = blocks[0]; returns mask of elements that fit."""
    ne, n, _ = blocks.shape
    r = blocks[0]
    try:
        l_ref = np.linalg.cholesky(r / np.trace(r))
    except np.linalg.LinAlgError:
        return np.zeros(ne, dtype=bool), None, None, None
    scale = np.trace(blocks, axis1=1, axis2=2) / np.trace(r)
    pattern = np.abs(r) > 1e-12 * np.abs(r).max()
    parent = np.full(n, -1)
    seen = np.zeros(n, dtype=bool)
    visit = []
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        queue = deque([root])
        while queue:
            i = queue.popleft()
            visit.append(i)
            for j in np.flatnonzero(pattern[i] & ~seen):
                seen[j] = True
                parent[j] = i
                queue.append(j)
    d = np.ones((ne, n))
    for i in visit:
        p = parent[i]
        if p >= 0:
            d[:, i] = np.sign(blocks[:, i, p] * r[i, p]) * d[:, p]
    recon = scale[:, None, None] * d[:, :, None] * r[None] * d[:, None, :]
    err = np.abs(recon - blocks).max(axis=(1, 2)) / np.abs(blocks).max(axis=(1, 2))
    ok = (err <= rtol) & (scale > 0)
    return ok, scale * np.trace(r), d, l_ref


def _root_factor(groups, nrows, rtol):
    """Sparse H with H H^T = sum_e P_e^T A_e P_e from groups of (blocks (G, n, n), row indices (G, n), elements (G,)).

    Roots come from eigendecompositions, dropping eigenvalues below rtol * max, since local matrices may be singular.
    Returns H and the element of each column.
    """
    rows, cols, vals, col_elem = [], [], [], []
    ncol = 0
    for blocks, idx, elems in groups:
        lam, vec = np.linalg.eigh(blocks)
        keep = lam > rtol * lam.max(axis=1, keepdims=True)
        root = vec * np.sqrt(np.where(keep, lam, 0.0))[:, None, :]
        e_idx, i_idx, j_idx = np.nonzero(np.broadcast_to(keep[:, None, :], root.shape))
        colid = np.cumsum(keep.ravel()).reshape(keep.shape) - 1
        rows.append(idx[e_idx, i_idx])
        cols.append(ncol + colid[e_idx, j_idx])
        vals.append(root[e_idx, i_idx, j_idx])
        col_elem.append(np.repeat(elems, keep.sum(axis=1)))
        ncol += int(keep.sum())
    if not rows:
        return sp.csr_matrix((nrows, 0)), np.zeros(0, dtype=np.int64)
    h = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nrows, ncol))
    return h, np.concatenate(col_elem)
