"""Gaussian random functions on an NGSolve FE space, with automatic algorithm selection."""

import copy
import math
import warnings

import ngsolve as ngs
import numpy as np
import torch

from . import noise as nz
from . import tuning as tn
from .backend import (
    NormalStream,
    make_generator,
    ngs_sparse,
    numpy_blas_threads,
    resolve_device,
    resolve_dtype,
    sparse_to_torch,
)
from .solvers import CudssDirect, NGSolveCG, NGSolveDirect, TorchPCG, cudss_available
from .space import SpaceInfo, mass_form
from .specs import AssembledPrecision, OperatorPrecision, PDEPrecision

# Largest free-dof count for which a sparse direct factorisation is chosen by default. Placeholder until the
# benchmark-fitted cost model replaces it.
DIRECT_LIMIT = {1: math.inf, 2: 2_000_000, 3: 200_000}


class Gaussian:
    """N(mean, C) as a random function in ``fes``.

    Parameters
    ----------
    fes : FESpace (H1 or L2). Dirichlet dofs, if any, are fixed to the mean.
    precision : PDEPrecision | AssembledPrecision
    mean : GridFunction or CoefficientFunction (interpolated with ``Set``), default 0
    method : optional overrides, e.g. {"noise": "quadrature", "solver": "cg"}. "noise" selects the white-noise
        sampler (even powers), "form_noise" the precision-shaped noise ("local-gram" | "dense-cholesky", odd powers
        and assembled precisions), "solver" one of "direct" | "cg" | "torch-pcg", "engine" one of "ngsolve" (CPU
        default) | "torch" (CUDA default; on CPU only useful for testing).
    device, dtype : torch device ("cpu", "cuda", "auto") and output dtype (default float32)
    tol : relative tolerance for iterative solvers
    """

    def __init__(self, fes, precision, mean=None, method=None, device="cpu", dtype=None, tol=1e-8, seed=None):
        self.space = SpaceInfo(fes)
        self.spec = precision
        self.method = dict(method or {})
        self.dtype = resolve_dtype(dtype)
        tun = self.method.get("tuning")
        self.tuning = None if tun is False else (tn.Tuning.load(tun) if isinstance(tun, str) else tun or tn.active())
        self.plan = {"space": self.space.describe()}
        self.device = self._choose_device(device)
        self.engine = self.method.get("engine", "torch" if self.device.type == "cuda" else "ngsolve")
        if self.engine == "ngsolve" and self.device.type != "cpu":
            raise ValueError("the ngsolve engine runs on the CPU only")
        self.tol = tol
        self.seed = seed
        self._prior = None  # set on Gaussians derived by conditioning: the prior they came from
        self.generator = make_generator(seed, self.device) if self.engine == "torch" else None
        self.stream = NormalStream(seed) if self.engine == "ngsolve" else None
        self.plan.update(engine=self.engine, device=str(self.device), dtype=str(self.dtype))
        if self.tuning is not None:
            self.plan["tuning"] = f"run {self.tuning.run}"
        self._mean = self._build_mean(mean)
        self._check_trace_class()
        if isinstance(precision, PDEPrecision):
            self._build_pde(precision)
        elif isinstance(precision, AssembledPrecision):
            self._build_assembled(precision)
        elif isinstance(precision, OperatorPrecision):
            self._build_operator(precision)
        else:
            raise TypeError(f"unsupported specification {type(precision).__name__}")
        self._finalise()

    # ---------------------------------------------------------------- planning

    @property
    def _platform(self):
        return tn.platform_of(self.engine, self.device.type, self.dtype)

    def _power(self):
        return self.spec.power if isinstance(self.spec, PDEPrecision) else None

    def _tuned_prior(self, platform):
        if self.tuning is None or self._power() is None:
            return None
        s = self.space
        return self.tuning.prior(platform, s.dim, s.kind, s.order, self._power(), s.nfree)

    def _choose_device(self, device):
        if device != "auto":
            return resolve_device(device)
        dev = resolve_device("auto")
        if dev.type == "cuda" and self.tuning is not None:
            cpu = self._tuned_prior("cpu-ngsolve")
            gpu = self._tuned_prior(tn.platform_of("torch", "cuda", self.dtype))
            if cpu and gpu and cpu["per_sample_s"] < gpu["per_sample_s"]:
                dev = resolve_device("cpu")
            self.plan["device_choice"] = "tuning" if cpu and gpu else "first available"
        return dev

    def _build_pde(self, spec: PDEPrecision):
        self.scale = spec.scale
        if not spec.is_integer:
            return self._build_fractional(spec)
        even = spec.power % 2 == 0
        self.noise = self._mass_noise() if even else self._form_noise(spec.form)
        self.solver = self.first = self._solver(spec.form)
        self.n_mass_steps = spec.power // 2 - 1 if even else (spec.power - 1) // 2
        nsolve = spec.power // 2 if even else (spec.power + 1) // 2
        self.plan["structure"] = (
            f"C = A^-{spec.power}: {'white noise' if even else 'stiffness-shaped noise'} + {nsolve} solve(s) per sample"
        )

    def _build_fractional(self, spec: PDEPrecision):
        """C = A^-beta, beta/2 = n + theta: c = (K^-1 M)^n r(M^-1 K) M^-1 z with r(lambda) ~ lambda^-theta rational,
        r(M^-1 K) M^-1 = c0 M^-1 + sum_l c_l (K + t_l M)^-1, z white noise."""
        from .rational import fractional_power, spectral_interval
        from .solvers import RationalSolver

        s = self.space
        alpha = spec.power / 2
        n = int(np.floor(alpha))
        theta = alpha - n
        self.noise = self._mass_noise()
        self.solver = self._solver(spec.form)
        mass_inv = self._solver(mass_form)
        k, m = s.assemble_free(spec.form), s.mass_free

        def via(solver):
            return lambda v: (
                solver.apply(torch.from_numpy(v).to(self._solver_device, self._work_dtype)).double().cpu().numpy()
            )

        lmin, lmax = spectral_interval(via(self.solver), via(mass_inv), k, m)
        rational = fractional_power(theta, lmin, lmax, rtol=spec.rtol)
        shifted = [self._solver(_shifted_form(spec.form, t)) for t in rational.shifts]
        self.first = RationalSolver(rational, shifted, mass_inv)
        self.n_mass_steps = n
        self.plan["structure"] = (
            f"C = A^-{spec.power:g}: white noise + rational (M^-1 K)^-{theta:g} ({rational.degree} shifted solves, "
            f"rel. error {rational.rel_error:.1e} on [{lmin:.3g}, {lmax:.3g}]) + {n} integer solve(s) per sample"
        )

    def _build_operator(self, spec: OperatorPrecision):
        """Apply-only precision: c = r(M^-1 Q) M^-1 z, r ~ lambda^-1/2, all shifts in one multi-shift PCG run."""
        from .blackbox import inverse_iteration_min, lanczos_interval
        from .rational import fractional_power
        from .solvers import MultiShiftRationalSolver

        s = self.space
        dev, wd = self._solver_device, self._work_dtype
        mass_t = sparse_to_torch(s.mass_free, dev, wd)
        if spec.convention == "dual->dual":
            apply_q = lambda c: spec.apply(mass_t @ c)  # noqa: E731
        else:
            apply_q = spec.apply
        self.noise = self._mass_noise()
        mass_inv = self._solver(mass_form)
        if spec.spectrum is not None:
            lmin, lmax = spec.spectrum
        else:
            kw = dict(dtype=wd, device=dev)
            lmin = 0.5 * inverse_iteration_min(apply_q, lambda c: mass_t @ c, mass_inv.apply, s.nfree, **kw)
            _, hi, res = lanczos_interval(apply_q, mass_inv.apply, s.nfree, iters=80, **kw)
            lmax = 2.0 * (hi + res)
        rational = fractional_power(0.5, lmin, lmax, rtol=spec.rtol)
        self.solver = self.first = MultiShiftRationalSolver(rational, apply_q, mass_inv, spec.tol, dev, wd, s)
        self.n_mass_steps, self.scale = 0, spec.scale
        self.plan["structure"] = (
            f"apply-only precision: white noise + (M^-1 Q)^-1/2 by rational approximation ({rational.degree} shifts, "
            f"rel. error {rational.rel_error:.1e} on [{lmin:.3g}, {lmax:.3g}]) in one multi-shift PCG run per sample"
        )

    @property
    def _solver_device(self):
        return self.device if self.engine == "torch" else torch.device("cpu")

    def _build_assembled(self, spec: AssembledPrecision):
        self.noise = self._form_noise(spec.form)
        self.solver = self.first = self._solver(spec.form)
        self.n_mass_steps, self.scale = 0, spec.scale
        self.plan["structure"] = "Sigma^-1 = Q assembled: local Q-noise + 1 solve"

    def _finalise(self):
        s = self.space
        if self.engine == "torch":
            self.noise.setup(self.device, self._work_dtype)
            self._mass = sparse_to_torch(s.mass_free, self.device, self._work_dtype) if self.n_mass_steps else None
        else:
            self.noise.ngs_setup(s)
            self._ngs_mass = ngs_sparse(s.mass) if self.n_mass_steps else None
            self._eta = ngs.la.BaseVector(self.nrand)
            self._z, self._c, self._tmp = (ngs.la.BaseVector(s.ndof) for _ in range(3))
        self._maps = []  # noise-space maps, applied to eta before the pipeline
        self._post_maps = []  # FE-space maps, applied to the pipeline's output
        # NumPy BLAS threads inside the NGSolve engine (see backend.numpy_blas_threads); None = unchanged.
        self._cpu_blas_threads = self.method.get("cpu_blas_threads", 1)
        self.plan.update(**self.noise.info(), **self.first.info())

    @property
    def _work_dtype(self):
        return self.dtype if self.device.type == "cuda" else torch.float64

    def _mass_noise(self):
        s, choice = self.space, self.method.get("noise", "auto")
        if choice == "auto" and self.tuning is not None:
            rec = self.tuning.noise(self._platform, s.dim, s.kind, s.order, s.nfree)
            if rec and (rec != "diagonal" or s.mass_is_diagonal) and not rec.startswith(("chebyshev", "lumped")):
                choice = rec
                self.plan["noise_choice"] = "tuning"
        if choice == "local-gram-explicit":
            n = s.broken.ndof / max(s.mesh.ne, 1)
            if s.mesh.ne * n * (n + 1) / 2 > 4e8:
                choice = "local-gram"
        if choice == "auto":
            if s.mass_is_diagonal:
                choice = "diagonal"
            else:
                # Explicit sparse factor: fastest exact option in the CPU benchmarks (results/summary.md) while its
                # ~n^2/2 nonzeros per element fit the budget; the GEMM variant needs no per-element storage.
                n = s.broken.ndof / max(s.mesh.ne, 1)
                choice = "local-gram-explicit" if s.mesh.ne * n * (n + 1) / 2 <= 2e8 else "local-gram"
        builders = {
            "diagonal": lambda: nz.DiagonalNoise(s.mass_free),
            "local-gram": lambda: nz.LocalGramNoise(s),
            "local-gram-explicit": lambda: nz.LocalGramNoise(s).explicit(),
            "quadrature": lambda: nz.quadrature_noise(s),
            "dense-cholesky": lambda: nz.dense_cholesky_noise(s.mass_free),
            "chebyshev": lambda: nz.ChebyshevNoise(s.mass_free, tol=self.method.get("noise_tol", 1e-6)),
            "lumped": lambda: nz.LumpedNoise(s.mass_free),
        }
        return builders[choice]()

    def _form_noise(self, form):
        s, choice = self.space, self.method.get("form_noise", "auto")
        if choice in ("auto", "local-gram"):
            try:
                return nz.LocalGramNoise(s, form=form)
            except (RuntimeError, NotImplementedError) as err:
                if choice == "local-gram" or s.nfree > 20_000:
                    raise RuntimeError(f"no exact local factorisation of the precision: {err}") from err
                warnings.warn(f"falling back to dense Cholesky of the precision ({err})")
        return nz.dense_cholesky_noise(s.assemble_free(form))

    def _solver(self, form):
        s, choice = self.space, self.method.get("solver", "auto")
        rec = self._tuned_prior(self._platform) if choice == "auto" else None
        if rec and rec["method"].get("engine", self.engine) == self.engine:
            m = rec["method"]
            # Sparse direct factorisations in 3D are only trusted near the measured sizes (fill-in grows fast).
            if not (m["solver"] == "direct" and s.dim == 3 and s.nfree > max(DIRECT_LIMIT[3], 2 * rec["_target"])):
                choice = m["solver"]
                for key in ("inverse", "precond"):
                    if key in m:
                        self.method.setdefault(key, m[key])
                self.plan["solver_choice"] = "tuning"
        if choice == "auto":
            if self.engine == "torch":
                choice = "cudss" if self.device.type == "cuda" and cudss_available() else "torch-pcg"
            else:
                choice = "direct" if s.nfree <= DIRECT_LIMIT[s.dim] else "cg"
        if choice == "direct":
            # NGSolve's sparsecholesky solve is ~45x slower than umfpack on 1D (chain) elimination trees, but
            # 5-10x faster in 2D/3D (bench/bench_prior.py).
            default = "umfpack" if s.dim == 1 else "sparsecholesky"
            return NGSolveDirect(s, form, inverse=self.method.get("inverse", default))
        if choice == "cg":
            pre = self.method.get("precond", "h1amg" if (s.kind == "H1" and s.order == 1) else "bddc")
            return NGSolveCG(s, form, precond=pre, tol=self.tol)
        if choice == "cudss":
            if self.device.type != "cuda":
                raise ValueError("cudss needs a CUDA device")
            return CudssDirect(s.assemble_free(form), self.device, self._work_dtype)
        if choice == "torch-pcg":
            if self.engine != "torch":
                raise ValueError("torch-pcg needs the torch engine")
            tol = max(self.tol, 1e-6 if self._work_dtype == torch.float32 else 1e-14)
            return TorchPCG(s.assemble_free(form), self.device, self._work_dtype, tol=tol)
        raise ValueError(f"unknown solver {choice!r}")

    def _build_mean(self, mean):
        gf = ngs.GridFunction(self.space.fes)
        if mean is not None:
            if isinstance(mean, ngs.GridFunction):
                gf.vec.data = mean.vec
            else:
                gf.Set(mean)
        return gf.vec.FV().NumPy().copy()

    def _check_trace_class(self):
        if isinstance(self.spec, PDEPrecision) and 2 * self.spec.power <= self.space.dim:
            warnings.warn(
                f"C = A^-{self.spec.power} with second-order A is not trace class in {self.space.dim}D: "
                "samples are not L2 functions and their norms grow under refinement"
            )

    # ---------------------------------------------------------------- sampling

    @property
    def nrand(self) -> int:
        return self.noise.nrand

    def _ngs_one(self, out):
        """Run the CPU pipeline on self._eta; writes the zero-mean full-dof sample into ``out`` (BaseVector)."""
        for m in reversed(self._maps):
            m.apply_ngs(self._eta)
        self.noise.ngs_apply(self._eta, self._z)
        self.first.ngs_apply(self._z, self._c)
        for _ in range(self.n_mass_steps):
            self._tmp.data = self._ngs_mass * self._c
            self.solver.ngs_apply(self._tmp, self._c)
        out.data = self.scale * self._c
        for pm in self._post_maps:
            pm.apply_ngs(out)

    def apply(self, eta: torch.Tensor) -> torch.Tensor:
        """Map standard normal eta (nrand x k) to zero-mean free-dof coefficients (nfree x k)."""
        if self.engine == "torch":
            x = eta.to(self.device, self._work_dtype)
            for m in reversed(self._maps):
                x = m.apply_torch(x)
            c = self.first.apply(self.noise.apply(x))
            for _ in range(self.n_mass_steps):
                c = self.solver.apply(self._mass @ c)
            c = self.scale * c
            for pm in self._post_maps:
                c = pm.apply_torch(c)
            return c
        squeeze = eta.ndim == 1
        e = (eta[:, None] if squeeze else eta).detach().cpu().double().numpy()
        out = np.empty((self.space.nfree, e.shape[1]))
        res = ngs.la.BaseVector(self.space.ndof)
        with ngs.TaskManager(), numpy_blas_threads(self._cpu_blas_threads):
            for k in range(e.shape[1]):
                self._eta.FV().NumPy()[:] = e[:, k]
                self._ngs_one(res)
                out[:, k] = res.FV().NumPy()[self.space.free]
        t = torch.from_numpy(out)
        return t[:, 0] if squeeze else t

    def apply_t(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint of ``apply``: free-dof vectors (nfree x k) -> noise space (nrand x k).

        With Sigma = L L^T the covariance of ``apply``, this is L^T. The solve layer is symmetric, so L^T reuses the
        same solves in reverse order followed by the noise adjoint H^T. Used to set up posteriors.
        """
        squeeze = y.ndim == 1
        x = y[:, None] if squeeze else y
        if self.engine == "torch":
            dev, wd, mass = self.device, self._work_dtype, self._mass
        else:
            dev, wd = torch.device("cpu"), torch.float64
            if not getattr(self, "_noise_torch_ready", False):
                self.noise.setup(dev, wd)
                self._noise_torch_ready = True
            mass = self._mass_cpu() if self.n_mass_steps else None
        x = x.to(dev, wd)
        for pm in reversed(self._post_maps):
            x = pm.apply_t_torch(x)
        # L = scale (K^-1 M)^n R H  =>  L^T = scale H^T R (M K^-1)^n  (R = first stage, symmetric)
        v = x
        for _ in range(self.n_mass_steps):
            v = mass @ self.solver.apply(v)
        v = self.first.apply(v)
        eta = self.scale * self.noise.apply_t(v)
        for m in self._maps:
            eta = m.apply_torch(eta)
        return eta[:, 0] if squeeze else eta

    def _mass_cpu(self):
        if getattr(self, "_mass_t_cpu", None) is None:
            self._mass_t_cpu = sparse_to_torch(self.space.mass_free, torch.device("cpu"), torch.float64)
        return self._mass_t_cpu

    def apply_precision(self, c: torch.Tensor) -> torch.Tensor:
        """Prior precision Sigma^-1 (primal -> dual) on free-dof vectors (nfree x k), computed in float64 on the CPU.

        PDE priors: Sigma^-1 = scale^-2 K (M^-1 K)^(power-1), using a sparse factorisation of M (diagonal M is free).
        """
        if self._maps or self._post_maps:
            raise ValueError("apply_precision is defined for priors, not for conditioned Gaussians")
        s = self.space
        x = c.detach().cpu().double().numpy()
        x = x[:, None] if x.ndim == 1 else x
        if not hasattr(self, "_prec_ops"):
            k = s.assemble_free(self.spec.form)
            if s.mass_is_diagonal:
                dm = s.mass_free.diagonal()
                minv = lambda v: v / dm[:, None]  # noqa: E731
            else:
                ms = NGSolveDirect(s, mass_form)
                minv = lambda v: ms.apply(torch.from_numpy(v)).numpy()  # noqa: E731
            self._prec_ops = (k, minv)
        k, minv = self._prec_ops
        v = k @ x
        for _ in range((self.spec.power if isinstance(self.spec, PDEPrecision) else 1) - 1):
            v = k @ minv(v)
        out = torch.from_numpy(np.ascontiguousarray(v / self.scale**2))
        return out[:, 0] if c.ndim == 1 else out

    def _derive(self, label, mean=None, *, eta_map=None, post_map=None, seed=None):
        """A new Gaussian sharing this one's pipeline, with an extra noise-space or FE-space map and/or a new mean."""
        new = copy.copy(self)
        new._maps = self._maps + ([eta_map] if eta_map is not None else [])
        new._post_maps = self._post_maps + ([post_map] if post_map is not None else [])
        new._mean = self._mean.copy() if mean is None else mean
        new._prior = self if self._prior is None else self._prior
        new.plan = {**self.plan, "conditioning": self.plan.get("conditioning", []) + [label]}
        if self.engine == "torch":
            new.generator = make_generator(seed, self.device)
        else:
            new.stream = NormalStream(seed)
        return new

    @property
    def mean(self) -> np.ndarray:
        """Mean coefficient vector (all dofs)."""
        return self._mean.copy()

    def condition(self, *args, **kwargs):
        """Posterior under linear Gaussian observations; see :class:`ngsrng.posterior.LinearObservations`."""
        from .posterior import LinearObservations

        data = kwargs.pop("data", None)
        seed = kwargs.pop("seed", None)
        form = kwargs.pop("form", "fe")
        return LinearObservations(self, *args, **kwargs).posterior(data=data, seed=seed, form=form)

    def downdate(self, b, k, mean=None, seed=None):
        """N(mean, C0 - B K^-1 B*) for a known low-rank downdate of this prior (:func:`ngsrng.posterior.downdate`)."""
        from .posterior import downdate

        return downdate(self, b, k, mean=mean, seed=seed)

    def sample(self, n=None, out=None):
        """One sample as a GridFunction (n=None), or n samples as a (ndof x n) torch tensor.

        ``out="multivector"`` returns an NGSolve MultiVector instead.
        """
        ns = 1 if n is None else int(n)
        if self.engine == "torch":
            return self._sample_torch(n, ns, out)
        gf = ngs.GridFunction(self.space.fes)
        if n is None:
            vecs = [gf.vec]
        elif out == "multivector":
            mv = ngs.MultiVector(gf.vec, ns)
            vecs = [mv[k] for k in range(ns)]
        else:
            arr = np.empty((ns, self.space.ndof))
            res = ngs.la.BaseVector(self.space.ndof)
            vecs = [res] * ns
        eta_np = self._eta.FV().NumPy()
        with ngs.TaskManager(), numpy_blas_threads(self._cpu_blas_threads):
            for k, v in enumerate(vecs):
                self.stream.fill(eta_np)
                self._ngs_one(v)
                vn = v.FV().NumPy()
                vn += self._mean
                if n is not None and out != "multivector":
                    arr[k] = vn
        if n is None:
            return gf
        if out == "multivector":
            return mv
        return torch.from_numpy(arr).T.to(self.dtype)

    def _sample_torch(self, n, ns, out):
        eta = torch.randn(self.nrand, ns, generator=self.generator, device=self.device, dtype=self._work_dtype)
        c = self.apply(eta).to(self.dtype)
        full = torch.from_numpy(self._mean).to(self.device, self.dtype)[:, None].repeat(1, ns)
        full[torch.from_numpy(self.space.free).to(self.device)] += c
        if n is None:
            gf = ngs.GridFunction(self.space.fes)
            gf.vec.FV().NumPy()[:] = full[:, 0].cpu().double().numpy()
            return gf
        if out == "multivector":
            mv = ngs.MultiVector(ngs.GridFunction(self.space.fes).vec, ns)
            data = full.cpu().double().numpy()
            for k in range(ns):
                mv[k].FV().NumPy()[:] = data[:, k]
            return mv
        return full

    def __repr__(self):
        return "Gaussian(\n" + "\n".join(f"  {k}: {v}" for k, v in self.plan.items()) + "\n)"


def _shifted_form(form, t):
    return lambda u, v: form(u, v) + float(t) * u * v * ngs.dx
