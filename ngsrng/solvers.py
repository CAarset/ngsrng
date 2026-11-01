"""Linear solvers on the free dofs, all exposed as ``apply(X) -> A^{-1} X`` for torch tensors (nfree x ns)."""

import os
import warnings

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
import torch
from ngsolve.krylovspace import CGSolver

from .backend import Scratch, parallel_ranges, sparse_to_torch
from .space import SpaceInfo, mass_form


class Solver:
    name = "abstract"
    exact = True

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def info(self) -> dict:
        return {"solver": self.name}


class _NGSolveBase(Solver):
    threadsafe = True  # concurrent solves on separate vectors (checked against sequential results in the tests)

    def __init__(self, space: SpaceInfo, form):
        self.space = space
        self.form = form
        # mass solves on the dgjumps-free twin (same dofs): no facet couplings in the graph, hence no needless fill
        self.fes = space.plain if form is mass_form else space.fes
        self._vin = None
        self._vectors = Scratch(lambda: (self.mat.CreateColVector(), self.mat.CreateColVector()))

    def _columns(self, x, solve_one):
        """Solve column by column; columns are spread over threads (NGSolve releases the GIL), or, for a single
        column, solved with NGSolve's own task parallelism."""
        squeeze = x.ndim == 1
        xs = (x[:, None] if squeeze else x).detach().to("cpu", torch.float64).numpy()
        free = self.space.free
        out = np.empty((xs.shape[1], xs.shape[0]))  # rows: contiguous writes from the threads

        def work(lo, hi):
            rhs, sol = self._vectors.get()
            rhs_np, sol_np = rhs.FV().NumPy(), sol.FV().NumPy()
            for k in range(lo, hi):
                rhs_np[:] = 0.0
                rhs_np[free] = xs[:, k]
                solve_one(rhs, sol)
                out[k] = sol_np[free]

        if xs.shape[1] == 1 or not self.threadsafe:
            with ngs.TaskManager():
                work(0, xs.shape[1])
        else:
            parallel_ranges(xs.shape[1], work)
        res = torch.from_numpy(out.T).to(device=x.device, dtype=x.dtype)
        return res[:, 0] if squeeze else res


class NGSolveDirect(_NGSolveBase):
    """Sparse direct factorisation (NGSolve), computed once and reused for every sample."""

    def __init__(self, space: SpaceInfo, form, inverse="sparsecholesky"):
        super().__init__(space, form)
        self.name = f"ngsolve-direct({inverse})"
        u, v = self.fes.TnT()
        a = ngs.BilinearForm(self.fes)
        a += form(u, v)
        with ngs.TaskManager():
            a.Assemble()
            self.inv = a.mat.Inverse(space.fes.FreeDofs(), inverse=inverse)
        self.mat = a.mat

    def apply(self, x):
        return self._columns(x, self.ngs_apply)

    def ngs_apply(self, rhs, out):
        out.data = self.inv * rhs


class NGSolveCG(_NGSolveBase):
    """Preconditioned CG with an NGSolve preconditioner (bddc, h1amg, multigrid, local)."""

    exact = False
    threadsafe = False  # NGSolve preconditioners are not verified for concurrent application

    def __init__(self, space: SpaceInfo, form, precond="bddc", tol=1e-10, maxiter=5000):
        super().__init__(space, form)
        self.name = f"ngsolve-cg({precond})"
        self.tol, self.maxiter = tol, maxiter
        u, v = self.fes.TnT()
        a = ngs.BilinearForm(self.fes)
        a += form(u, v)
        self.pre = ngs.Preconditioner(a, precond)
        with ngs.TaskManager():
            a.Assemble()
        self.mat = a.mat
        self.iterations = []

    def apply(self, x):
        return self._columns(x, self.ngs_apply)

    def ngs_apply(self, rhs, out):
        cg = CGSolver(self.mat, self.pre.mat, tol=self.tol, maxiter=self.maxiter)
        out.data = cg * rhs
        self.iterations.append(cg.iterations)

    def info(self):
        its = self.iterations[-100:]
        return {**super().info(), "tol": self.tol, "mean_iterations": float(np.mean(its)) if its else None}


def cudss_available() -> bool:
    try:
        import nvmath.sparse.advanced  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _cudss_mtlayer():
    import glob
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    for base in list(spec.submodule_search_locations or []) if spec else []:
        hits = glob.glob(os.path.join(base, "*", "lib", "libcudss_mtlayer_gomp.so*"))
        if hits:
            return hits[0]
    return None


class CudssDirect(Solver):
    """GPU sparse Cholesky (NVIDIA cuDSS via nvmath-python), factorised once and reused for every sample.

    cuDSS fixes the right-hand-side shape per solver object, so factorisations are kept for widths 1 and ``width``
    (created on first use); batches are solved in chunks of ``width``, with small remainders one column at a time.
    """

    def __init__(self, a_free: sp.spmatrix, device, dtype, width=32):
        import logging

        import nvmath.sparse.advanced as adv

        self.name = "cudss"
        self.adv, self.width = adv, width
        self.a = sparse_to_torch(a_free, device, dtype)
        self.n, self.device, self.dtype = a_free.shape[0], device, dtype
        logger = logging.getLogger("ngsrng.cudss")
        logger.setLevel(logging.WARNING)
        self.options = adv.DirectSolverOptions(
            sparse_system_type=adv.DirectSolverMatrixType.SPD, multithreading_lib=_cudss_mtlayer(), logger=logger
        )
        self._solvers = {}
        self._get(1)

    def _get(self, k):
        if k not in self._solvers:
            b = torch.zeros(k, self.n, device=self.device, dtype=self.dtype).T
            s = self.adv.DirectSolver(self.a, b, options=self.options)
            s.plan()
            s.factorize()
            self._solvers[k] = s
        return self._solvers[k]

    def _solve(self, b):
        k = b.shape[1]
        s = self._get(k)
        # cuDSS checks strides exactly; .contiguous() is a no-op for size-1 dimensions, so copy explicitly into a
        # column-major (n x k) buffer with strides (1, n).
        rhs = torch.empty(k, self.n, device=self.device, dtype=self.dtype)
        rhs.copy_(b.T)
        s.reset_operands(b=rhs.T)
        return s.solve()

    def apply(self, x):
        squeeze = x.ndim == 1
        x = (x[:, None] if squeeze else x).to(self.dtype)
        k = x.shape[1]
        out = torch.empty(self.n, k, device=self.device, dtype=self.dtype)
        for start in range(0, k, self.width):
            block = x[:, start : start + self.width]
            r = block.shape[1]
            if r == self.width or r == 1:
                out[:, start : start + r] = self._solve(block)
            else:  # a remainder: zero-padded to the full width, one call (column by column was up to 25x slower)
                padded = torch.zeros(self.n, self.width, device=self.device, dtype=self.dtype)
                padded[:, :r] = block
                out[:, start : start + r] = self._solve(padded)[:, :r]
        return out[:, 0] if squeeze else out

    def __del__(self):
        for s in getattr(self, "_solvers", {}).values():
            try:
                s.free()
            except Exception:  # noqa: BLE001
                pass


class TorchPCG(Solver):
    """Batched Jacobi-preconditioned CG in torch: all columns iterate together (CPU or CUDA)."""

    exact = False

    def __init__(self, a_free: sp.spmatrix, device, dtype, tol=1e-6, maxiter=10000):
        self.name = "torch-pcg(jacobi)"
        self.tol, self.maxiter = tol, maxiter
        self.a = sparse_to_torch(a_free, device, dtype)
        self.dinv = torch.from_numpy(1.0 / a_free.diagonal()).to(device=device, dtype=dtype)
        self.iterations = []

    def apply(self, b):
        squeeze = b.ndim == 1
        b = b[:, None] if squeeze else b
        x = torch.zeros_like(b)
        r = b.clone()
        z = self.dinv[:, None] * r
        p = z.clone()
        rz = (r * z).sum(0)
        bnorm = torch.linalg.vector_norm(b, dim=0).clamp_min(torch.finfo(b.dtype).tiny)
        it, res = 0, None
        for it in range(1, self.maxiter + 1):
            ap = self.a @ p
            pap = (p * ap).sum(0)
            # a column with r = 0 (zero right-hand side, or converged exactly) would divide 0 / 0: it takes no step
            alpha = torch.where(pap > 0, rz / pap, torch.zeros_like(rz))
            x += alpha * p
            r -= alpha * ap
            res = (torch.linalg.vector_norm(r, dim=0) / bnorm).max()
            if res <= self.tol:
                break
            z = self.dinv[:, None] * r
            rz_new = (r * z).sum(0)
            p = z + torch.where(rz > 0, rz_new / rz, torch.zeros_like(rz)) * p
            rz = rz_new
        else:
            warnings.warn(f"torch PCG did not converge in {self.maxiter} iterations (relative residual "
                          f"{float(res):.1e} > tol {self.tol:.0e}); float32 rarely gets below ~1e-6", stacklevel=2)
        self.iterations.append(it)
        return x[:, 0] if squeeze else x

    def info(self):
        its = self.iterations[-100:]
        return {**super().info(), "tol": self.tol, "mean_iterations": float(np.mean(its)) if its else None}


class RationalSolver(Solver):
    """R = c0 M^-1 + sum_l c_l (K + t_l M)^-1, a rational function of M^-1 K applied to a dual vector (symmetric).

    With r(lambda) ~ lambda^-theta (``ngsrng.rational``), R z = r(M^-1 K) M^-1 z ~ (M^-1 K)^-theta M^-1 z.
    ``shifted`` are solvers for K + t_l M, ``mass_inv`` one for M (only used when c0 != 0).
    """

    def __init__(self, rational, shifted, mass_inv):
        self.rational, self.shifted, self.mass_inv = rational, shifted, mass_inv
        self.name = f"rational(theta={rational.theta:g}, {rational.degree} shifts, {shifted[0].name})"
        self.exact = all(sv.exact for sv in shifted)
        self._tmp = None

    def apply(self, x):
        out = self.rational.c0 * self.mass_inv.apply(x) if self.rational.c0 else torch.zeros_like(x)
        for c, sv in zip(self.rational.coeffs, self.shifted):
            out = out + float(c) * sv.apply(x)
        return out

    def ngs_apply(self, rhs, out):
        if self._tmp is None:
            self._tmp = Scratch(rhs.CreateVector)  # per thread
        tmp = self._tmp.get()
        out[:] = 0.0
        if self.rational.c0:
            self.mass_inv.ngs_apply(rhs, tmp)
            out.data += float(self.rational.c0) * tmp
        for c, sv in zip(self.rational.coeffs, self.shifted):
            sv.ngs_apply(rhs, tmp)
            out.data += float(c) * tmp

    def info(self):
        r = self.rational
        return {"solver": self.name, "rational_rel_error": r.rel_error, "spectrum": (r.lmin, r.lmax)}


class MultiShiftRationalSolver(Solver):
    """R = c0 M^-1 + sum_l c_l (Q + t_l M)^-1 for an apply-only Q, all shifts in one multi-shift PCG run."""

    exact = False

    def __init__(self, rational, apply_q, mass_inv, tol, device, dtype, space):
        from .blackbox import MultiShiftPCG

        self.rational, self.mass_inv, self.space = rational, mass_inv, space
        self.device, self.dtype = device, dtype
        self.cg = MultiShiftPCG(apply_q, mass_inv.apply, rational.shifts, rational.coeffs, tol=tol)
        self.name = f"apply-only rational(theta={rational.theta:g}, {rational.degree} shifts, multi-shift PCG)"

    def apply(self, x):
        x = x.to(self.device, self.dtype)
        out = self.cg.solve(x)
        if self.rational.c0:
            out = out + self.rational.c0 * self.mass_inv.apply(x)
        return out

    def ngs_apply(self, rhs, out):
        free = self.space.free
        x = torch.from_numpy(rhs.FV().NumPy()[free].copy())
        res = self.apply(x).double().cpu().numpy()
        out[:] = 0.0
        out.FV().NumPy()[free] = res

    def info(self):
        its = self.cg.iterations[-100:]
        r = self.rational
        return {
            "solver": self.name,
            "rational_rel_error": r.rel_error,
            "spectrum": (r.lmin, r.lmax),
            "mean_iterations": float(np.mean(its)) if its else None,
        }
