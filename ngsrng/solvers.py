"""Linear solvers on the free dofs, all exposed as ``apply(X) -> A^{-1} X`` for torch tensors (nfree x ns)."""

import os
import warnings

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
import torch
from ngsolve.krylovspace import CGSolver

from .backend import Rebuildable, Scratch, fixed_numpy_global_rng, parallel_ranges, sparse_to_torch
from .space import SpaceInfo, mass_form


class Solver:
    name = "abstract"
    exact = True

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def info(self) -> dict:
        return {"solver": self.name}


class _NGSolveBase(Solver, Rebuildable):
    """NGSolve matrices, factorisations and preconditioners do not pickle: they are rebuilt after unpickling
    (``_assemble``)."""

    threadsafe = True  # concurrent solves on separate vectors (checked against sequential results in the tests)

    def __init__(self, space: SpaceInfo, form):
        self.space = space
        self.form = form
        # mass solves on the dgjumps-free twin (same dofs): no facet couplings in the graph, hence no needless fill
        self.fes = space.plain if form is mass_form else space.fes
        self._vin = None
        self._vectors = Scratch(self._new_vectors)

    def _new_vectors(self):
        return self.mat.CreateColVector(), self.mat.CreateColVector()

    def _bilinear_form(self):
        u, v = self.fes.TnT()
        a = ngs.BilinearForm(self.fes)
        a += self.form(u, v)
        return a

    def _restore(self):
        self._assemble()

    def _columns(self, x, solve_one):
        """Solve column by column; columns are spread over threads (NGSolve releases the GIL), each solved
        single-threaded, a single column too, so that results repeat bit for bit. Solvers that must not run
        concurrently (BDDC) solve one column at a time with NGSolve's task parallelism (repeats to rounding)."""
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

        if self.threadsafe:
            parallel_ranges(xs.shape[1], work)
        else:
            with ngs.TaskManager():
                work(0, xs.shape[1])
        res = torch.from_numpy(out.T).to(device=x.device, dtype=x.dtype)
        return res[:, 0] if squeeze else res


# NGSolve direct solvers whose concurrent solves were checked against sequential results. "pardiso" (an MKL build)
# returned samples off by up to 1e67 in the GPU-box benchmark and is not offered.
INVERSES = ("sparsecholesky", "umfpack")


class NGSolveDirect(_NGSolveBase):
    """Sparse direct factorisation (NGSolve), computed once and reused for every sample."""

    def __init__(self, space: SpaceInfo, form, inverse="sparsecholesky"):
        if inverse not in INVERSES:
            raise ValueError(f"unsupported direct solver {inverse!r}; valid: {', '.join(INVERSES)}")
        super().__init__(space, form)
        self.name = f"ngsolve-direct({inverse})"
        self.inverse = inverse
        self._assemble()

    _transient = ("mat", "inv")

    def _assemble(self):
        a = self._bilinear_form()
        with ngs.TaskManager():
            a.Assemble()
            self.inv = a.mat.Inverse(self.space.freedofs, inverse=self.inverse)  # the SpaceInfo's free dofs
        self.mat = a.mat

    def apply(self, x):
        return self._columns(x, self.ngs_apply)

    def ngs_apply(self, rhs, out):
        out.data = self.inv * rhs


# NGSolve preconditioners usable without a mesh hierarchy. "multigrid" needs refinement levels: on a generated mesh it
# crashed the process (segfault) in every benchmarked case, so it is rejected up front.
PRECONDITIONERS = ("bddc", "h1amg", "local")


class NGSolveCG(_NGSolveBase):
    """Preconditioned CG with an NGSolve preconditioner (bddc, h1amg, local)."""

    exact = False

    def __init__(self, space: SpaceInfo, form, precond="bddc", tol=1e-10, maxiter=5000):
        if precond not in PRECONDITIONERS:
            raise ValueError(f"unsupported preconditioner {precond!r}; valid: {', '.join(PRECONDITIONERS)} "
                             "(multigrid needs a mesh hierarchy and crashes on a generated mesh)")
        super().__init__(space, form)
        # Concurrent applications of one BDDC preconditioner corrupted the samples (relative error 0.1 in a batch of
        # threaded samples, 1e-10 sequentially); h1amg gave identical results either way; Jacobi is unverified.
        self.threadsafe = precond == "h1amg"
        self.name = f"ngsolve-cg({precond})"
        self.precond = precond
        self.tol, self.maxiter = tol, maxiter
        self._assemble()
        self.iterations = []

    _transient = ("mat", "pre")

    def _assemble(self):
        a = self._bilinear_form()
        self.pre = ngs.Preconditioner(a, self.precond)
        with ngs.TaskManager():
            a.Assemble()
        self.mat = a.mat

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
    """NVIDIA's cuDSS through nvmath-python, on an NVIDIA GPU (PyTorch's ROCm build also reports "cuda" devices)."""
    if getattr(torch.version, "hip", None):
        return False
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


# Right-hand-side staging buffers shared by all cuDSS solvers of one shape: the solves run one after the other on the
# same stream, so the L shifted solvers of a fractional power need one (n x width) buffer, not L of them.
_STAGING = {}


def _staging(n, width, device, dtype):
    key = (n, width, str(device), dtype)
    if key not in _STAGING:
        # column-major (n x width): cuDSS checks strides exactly, so the buffer is (width x n) transposed
        _STAGING[key] = torch.zeros(width, n, device=device, dtype=dtype)
    return _STAGING[key]


class CudssDirect(Solver, Rebuildable):
    """GPU sparse Cholesky (NVIDIA cuDSS via nvmath-python), factorised once and reused for every sample.

    cuDSS fixes the right-hand-side shape per solver object. One factorisation of width ``width`` is kept; batches
    are solved in blocks of ``width`` columns, and a narrower block (a single sample, a remainder) is zero-padded to
    the full width (one padded call; column by column was up to 25x slower, and a second factorisation for single
    columns doubled the device memory of every solver).
    """

    _transient = ("_solver", "_rhs")

    def __init__(self, a_free: sp.spmatrix, device, dtype, width=32):
        self.name = "cudss"
        self.width = int(width)
        self.n, self.device, self.dtype = a_free.shape[0], device, dtype
        self._a = sp.csr_matrix(a_free)  # host copy: the factorisation is redone after unpickling
        self._factorize()

    def _restore(self):
        self._factorize()

    def _factorize(self):
        import logging

        import nvmath.sparse.advanced as adv

        device, dtype = self.device, self.dtype
        a = sparse_to_torch(self._a, device, dtype)
        logger = logging.getLogger("ngsrng.cudss")
        logger.setLevel(logging.WARNING)
        options = adv.DirectSolverOptions(
            sparse_system_type=adv.DirectSolverMatrixType.SPD, multithreading_lib=_cudss_mtlayer(), logger=logger
        )
        self._rhs = _staging(self.n, self.width, device, dtype)
        self._solver = adv.DirectSolver(a, self._rhs.T, options=options)
        self._solver.plan()
        self._solver.factorize()

    def _solve(self, block):
        """Solve for an (n x r) block, r <= width, through the shared staging buffer; returns (n x r)."""
        r = block.shape[1]
        if r < self.width:
            self._rhs[r:].zero_()
        self._rhs[:r].copy_(block.T)
        self._solver.reset_operands(b=self._rhs.T)
        return self._solver.solve()[:, :r]

    def apply(self, x):
        squeeze = x.ndim == 1
        x = (x[:, None] if squeeze else x).to(self.dtype)
        k = x.shape[1]
        out = torch.empty(self.n, k, device=self.device, dtype=self.dtype)
        for start in range(0, k, self.width):
            out[:, start : start + self.width] = self._solve(x[:, start : start + self.width])
        return out[:, 0] if squeeze else out

    def free(self):
        """Release the factorisation now (device memory), instead of at garbage collection."""
        s, self._solver = getattr(self, "_solver", None), None
        if s is not None:
            try:
                s.free()
            except Exception:  # noqa: BLE001
                pass

    def __del__(self):
        self.free()


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


def device_spd_solver(a: sp.spmatrix, device, dtype=torch.float64, tol=1e-13):
    """A solver for an SPD matrix on a torch device, for mass-type matrices (well conditioned after Jacobi scaling):
    cuDSS (exact, as the CPU's sparse Cholesky) on NVIDIA GPUs with nvmath-python, else Jacobi-preconditioned CG
    to the relative residual ``tol`` (AMD GPUs, the CPU)."""
    device = torch.device(device)
    if device.type == "cuda" and cudss_available():
        return CudssDirect(a, device, dtype)
    return TorchPCG(sp.csr_matrix(a), device, dtype, tol=tol if dtype == torch.float64 else max(tol, 1e-6))


def pyamg_available() -> bool:
    try:
        import pyamg  # noqa: F401
    except ImportError:
        return False
    return True


class TorchAMGPCG(Solver):
    """Batched CG preconditioned by one smoothed-aggregation AMG V-cycle, every operation on the device.

    The hierarchy is built once on the CPU (pyamg) and uploaded as sparse matrices. The V-cycle smooths with damped
    Jacobi (the same sweeps before and after, weight 4/3 over the spectral radius of D^-1 A) and solves the coarsest
    level with a dense Cholesky factor, so the preconditioner is symmetric positive definite and CG applies. All work
    per iteration is sparse-dense products and elementwise operations on all right-hand sides together, so it runs on
    any torch device -- also where no GPU sparse direct solver exists (ROCm) -- and its cost grows linearly with the
    number of unknowns, without the fill-in of a factorisation.
    """

    exact = False

    def __init__(self, a_free: sp.spmatrix, device, dtype, tol=1e-8, maxiter=1000, sweeps=2, max_coarse=500):
        import pyamg
        from pyamg.util.linalg import approximate_spectral_radius

        self.name = "torch-pcg(amg)"
        self.tol, self.maxiter, self.sweeps = tol, maxiter, sweeps
        self.device, self.dtype = device, dtype
        a = sp.csr_matrix(a_free, dtype=np.float64)
        with fixed_numpy_global_rng():  # pyamg's spectral-radius estimates draw from np.random
            ml = pyamg.smoothed_aggregation_solver(a, symmetry="symmetric", max_coarse=max_coarse)
            radii = [approximate_spectral_radius(sp.diags(1.0 / lvl.A.diagonal()) @ lvl.A.tocsr())
                     for lvl in ml.levels[:-1]]
        self.a = sparse_to_torch(a, device, dtype)
        self.levels = []
        for lvl, rho in zip(ml.levels[:-1], radii):
            al = lvl.A.tocsr()
            dinv = 1.0 / al.diagonal()
            omega = 4.0 / (3.0 * rho)
            self.levels.append((sparse_to_torch(al, device, dtype), sparse_to_torch(lvl.P.tocsr(), device, dtype),
                                sparse_to_torch(lvl.P.T.tocsr(), device, dtype),
                                torch.from_numpy(omega * dinv).to(device=device, dtype=dtype)[:, None]))
        coarse = torch.from_numpy(ml.levels[-1].A.toarray()).to(device=device, dtype=torch.float64)
        self.coarse = torch.linalg.cholesky(coarse).to(dtype)
        self.nlevels = len(ml.levels)
        self.iterations = []

    def _vcycle(self, level, b):
        if level == len(self.levels):
            return torch.cholesky_solve(b, self.coarse)
        a, p, r, wd = self.levels[level]
        x = wd * b  # first sweep from x = 0
        for _ in range(self.sweeps - 1):
            x = x + wd * (b - a @ x)
        x = x + p @ self._vcycle(level + 1, r @ (b - a @ x))
        for _ in range(self.sweeps):
            x = x + wd * (b - a @ x)
        return x

    def apply(self, b):
        squeeze = b.ndim == 1
        b = (b[:, None] if squeeze else b).to(self.dtype)
        x = torch.zeros_like(b)
        r = b.clone()
        z = self._vcycle(0, r)
        p = z.clone()
        rz = (r * z).sum(0)
        bnorm = torch.linalg.vector_norm(b, dim=0).clamp_min(torch.finfo(b.dtype).tiny)
        it, res = 0, None
        for it in range(1, self.maxiter + 1):
            ap = self.a @ p
            pap = (p * ap).sum(0)
            alpha = torch.where(pap > 0, rz / pap, torch.zeros_like(rz))
            x += alpha * p
            r -= alpha * ap
            res = (torch.linalg.vector_norm(r, dim=0) / bnorm).max()
            if res <= self.tol:
                break
            z = self._vcycle(0, r)
            rz_new = (r * z).sum(0)
            p = z + torch.where(rz > 0, rz_new / rz, torch.zeros_like(rz)) * p
            rz = rz_new
        else:
            warnings.warn(f"AMG-preconditioned CG did not converge in {self.maxiter} iterations (relative residual "
                          f"{float(res):.1e} > tol {self.tol:.0e})", stacklevel=2)
        self.iterations.append(it)
        return x[:, 0] if squeeze else x

    def info(self):
        its = self.iterations[-100:]
        return {**super().info(), "tol": self.tol, "levels": self.nlevels,
                "mean_iterations": float(np.mean(its)) if its else None}


class RationalSolver(Solver, Rebuildable):
    """R = c0 M^-1 + sum_l c_l (K + t_l M)^-1, a rational function of M^-1 K applied to a dual vector (symmetric).

    With r(lambda) ~ lambda^-theta (``ngsrng.rational``), R z = r(M^-1 K) M^-1 z ~ (M^-1 K)^-theta M^-1 z.
    ``shifted`` are solvers for K + t_l M, ``mass_inv`` one for M (only used when c0 != 0).
    """

    def __init__(self, rational, shifted, mass_inv):
        self.rational, self.shifted, self.mass_inv = rational, shifted, mass_inv
        self.name = f"rational(theta={rational.theta:g}, {rational.degree} shifts, {shifted[0].name})"
        self.exact = all(sv.exact for sv in shifted)
        self.threadsafe = all(getattr(sv, "threadsafe", True) for sv in [*shifted, mass_inv])
        self._tmp = None

    _transient = ("_tmp",)  # per-thread NGSolve vectors, made on first use

    def _restore(self):
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
