"""Gaussian random functions on an NGSolve FE space, with automatic algorithm selection."""

import copy
import difflib
import math
import warnings

import ngsolve as ngs
import numpy as np
import torch

from . import noise as nz
from . import tuning as tn
from .backend import (
    NormalStream,
    Rebuildable,
    Scratch,
    available_threads,
    ngs_sparse,
    numpy_blas_threads,
    parallel_ranges,
    resolve_device,
    resolve_dtype,
    sparse_to_torch,
)
from .inplace import InPlace, as_tensor, chunks, default_work_values, write
from .seeding import as_source, inherit
from .solvers import (
    INVERSES,
    PRECONDITIONERS,
    CudssDirect,
    NGSolveCG,
    NGSolveDirect,
    TorchAMGPCG,
    TorchPCG,
    cudss_available,
    pyamg_available,
)
from .space import SpaceInfo, mass_form
from .specs import AssembledPrecision, OperatorPrecision, PDEPrecision

# Largest free-dof count for which a sparse direct factorisation is chosen by default. Placeholder until the
# benchmark-fitted cost model replaces it.
DIRECT_LIMIT = {1: math.inf, 2: 2_000_000, 3: 200_000}

# Accepted ``method`` keys and, for the algorithm choices, their values. Anything else raises: an ignored key would
# silently change the computation (a misspelt "engine" runs a GPU study on the CPU, in float64).
NOISE_METHODS = ("auto", "diagonal", "local-gram", "local-gram-explicit", "quadrature", "dense-cholesky", "chebyshev",
                 "lumped")
FORM_NOISE_METHODS = ("auto", "local-gram", "rational", "dense-cholesky")
SOLVERS = ("auto", "direct", "cg", "cudss", "amg-pcg", "torch-pcg")
ENGINES = ("auto", "ngsolve", "torch")
METHOD_KEYS = {
    "noise": NOISE_METHODS,
    "form_noise": FORM_NOISE_METHODS,
    "solver": SOLVERS,
    "engine": ENGINES,
    # Arithmetic of the torch engine. float32 solves lose accuracy like cond(K) * 6e-8: the GPU benchmark measured
    # pathwise errors of 1e-4 at 5k dofs and O(1) on fine meshes, so float64 is the default and float32 an opt-in.
    "work_dtype": ("float64", "float32"),
    "tuning": None,  # False, a recommendations.json path, or a ngsrng.tuning.Tuning
    "noise_tol": None,  # chebyshev noise
    "inverse": INVERSES,  # NGSolve direct solvers (pardiso returned garbage in the benchmark)
    "precond": PRECONDITIONERS,  # NGSolve preconditioners (multigrid needs a mesh hierarchy)
    "cudss_width": None,
    "threads": None,
    "cpu_blas_threads": None,
}


def check_method(method) -> dict:
    """A copy of the ``method`` overrides, with every key and algorithm name checked (ValueError if unknown)."""
    if method is None:
        return {}
    if not isinstance(method, dict):
        raise TypeError("method must be a dict of algorithm overrides, e.g. {'solver': 'cg'} (the inversion method "
                        "of a distribution is a per-call option: rng.gamma(2.0, method='table'))")
    for key, value in method.items():
        if key not in METHOD_KEYS:
            close = difflib.get_close_matches(str(key), METHOD_KEYS, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            raise ValueError(f"unknown method key {key!r}{hint}; valid keys: {', '.join(METHOD_KEYS)}")
        choices = METHOD_KEYS[key]
        if choices is not None and value not in choices:
            raise ValueError(f"unknown {key} {value!r}; valid: {', '.join(choices)}")
    return dict(method)


class Gaussian(InPlace, Rebuildable):
    """N(mean, C) as a random function in ``fes``.

    Parameters
    ----------
    fes : FESpace (H1 or L2). Dirichlet dofs, if any, are fixed to the mean.
    precision : PDEPrecision | AssembledPrecision
    mean : GridFunction or CoefficientFunction (interpolated with ``Set``), default 0
    method : optional overrides, e.g. {"noise": "quadrature", "solver": "cg"} (unknown keys or names raise; see
        ``METHOD_KEYS``). "noise" selects the white-noise sampler (even powers; "chebyshev" and "lumped" are
        approximate: lumped noise has the lumped mass matrix as covariance, off by O(1) for H1 P1), "form_noise"
        the route for odd powers and assembled precisions ("local-gram": precision-shaped noise, exact; "rational":
        white noise and a rational approximation of the square root, relative accuracy 1e-7, used automatically
        when the form couples elements; "dense-cholesky": exact, O(n^2) memory, only on request), "solver" one of
        "direct" | "cg" | "cudss" | "amg-pcg" | "torch-pcg" (with "inverse" / "precond" naming the NGSolve direct
        solver / preconditioner; "amg-pcg", AMG-preconditioned CG on the torch engine's device, is the GPU solver
        where cuDSS is unavailable, e.g. on AMD GPUs, and needs pyamg), "engine" one of "ngsolve" (CPU default) |
        "torch" (GPU default; on CPU only useful for testing), "work_dtype" ("float64", the default, or "float32":
        the torch engine's arithmetic; float32 solves are fast but inaccurate on fine meshes). Also "tuning" (False
        disables the calibration, or a recommendations.json path), "noise_tol" (chebyshev noise), "cudss_width",
        "threads" (CPU batches) and "cpu_blas_threads".
    device, dtype : torch device ("cpu", "cuda", "auto") and output dtype (default float32). Every engine computes in
        float64 and casts at the end, unless method={"work_dtype": "float32"}.
    tol : relative tolerance for iterative solvers
    seed : None (default: torch's global generator, read at each draw, so ``torch.manual_seed`` reproduces the
        samples), a ``torch.Generator`` or ``numpy.random.Generator`` (read at each draw), or an int / SeedSequence
        (an own stream). Posteriors and other derived samplers inherit it; see :mod:`ngsrng.seeding`.
    """

    def __init__(self, fes, precision, mean=None, method=None, device="cpu", dtype=None, tol=1e-8, seed=None):
        self.space = SpaceInfo(fes)
        self.spec = precision
        self.method = check_method(method)
        self.dtype = resolve_dtype(dtype)
        tun = self.method.get("tuning")
        self.tuning = None if tun is False else (tn.Tuning.load(tun) if isinstance(tun, str) else tun or tn.active())
        self.plan = {"space": self.space.describe()}
        self.device = self._choose_device(device)
        if self._work_dtype == torch.float32:
            warnings.warn("work_dtype='float32': solves in single precision lose accuracy like cond(K) * 6e-8 "
                          "(pathwise errors from 1e-4 on small meshes to O(1) on fine ones); check against float64",
                          stacklevel=2)
        self.work_values = default_work_values(self.device)  # chunk size of in-place fills on the torch engine
        engine = self.method.get("engine", "auto")
        self.engine = ("torch" if self.device.type == "cuda" else "ngsolve") if engine == "auto" else engine
        if self.engine == "ngsolve" and self.device.type != "cpu":
            raise ValueError("the ngsolve engine runs on the CPU only")
        self.tol = tol
        self.seed = seed
        self._prior = None  # set on Gaussians derived by conditioning: the prior they came from
        self._conditioning = []  # the exact updates of the prior covariance behind it (ngsrng.reference)
        self.source = as_source(seed)
        self.stream = NormalStream(self.source) if self.engine == "ngsolve" else None
        self.plan.update(engine=self.engine, device=str(self.device), dtype=str(self.dtype))
        self.plan["randomness"] = self.source.describe()
        self.plan["tuning"] = "built-in rules"
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
        if tun is None:  # the automatic layers (not an explicit choice): tell an uncalibrated machine how to calibrate
            tn.notice(self._platform, self.tuning, self.plan)

    # ---------------------------------------------------------------- planning

    @property
    def _platform(self):
        name = tn.device_name(self.device) if self.device.type == "cuda" else None
        return tn.platform_of(self.engine, self.device.type, self._work_dtype, name)

    def _tuning_note(self, what, hit, used, why=None):
        """Record in the plan which calibration a choice came from, or why a tuned suggestion was not used."""
        note = f"{what}: {hit['_layer']} calibration (run {hit['_run']})" if used else \
            f"{what}: built-in rule ({hit['_layer']} calibration's {why})"
        notes = [n for n in self.plan.get("tuning", "").split("; ") if n and n != "built-in rules"]
        if note not in notes:
            notes.append(note)
        self.plan["tuning"] = "; ".join(notes)

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
            gpu = self._tuned_prior(tn.platform_of("torch", "cuda", self.method.get("work_dtype", "float64"),
                                                   tn.device_name(dev)))
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
        if self.noise is None:  # odd power without precision-shaped noise: (M^-1 K)^-1/2 by the rational route
            return self._build_fractional(spec)
        self.solver = self.first = self._solver(spec.form)
        self.n_mass_steps = spec.power // 2 - 1 if even else (spec.power - 1) // 2
        nsolve = spec.power // 2 if even else (spec.power + 1) // 2
        self.plan["structure"] = (
            f"C = A^-{spec.power}: {'white noise' if even else 'stiffness-shaped noise'} + {nsolve} solve(s) per sample"
        )

    def _build_fractional(self, spec: PDEPrecision, label=None):
        """C = A^-beta, beta/2 = n + theta: c = (K^-1 M)^n r(M^-1 K) M^-1 z with r(lambda) ~ lambda^-theta rational,
        r(M^-1 K) M^-1 = c0 M^-1 + sum_l c_l (K + t_l M)^-1, z white noise.

        Also the route for odd integer powers (theta = 1/2) whose form couples elements (DG facet terms), where
        precision-shaped noise would need a global factorisation: only white noise, sparse solves and the
        relative-accuracy rational approximation."""
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

        # The spectral interval spans cond(M^-1 K) ~ h^-2: inverse iteration needs float64 solves (in float32 the
        # factorisation of a fine 1D mesh broke down and the estimate came out NaN). Single-precision engines get
        # temporary float64 solvers, released before the shifted ones are built.
        f64 = self._work_dtype == torch.float64
        est_k = self.solver if f64 else self._solver(spec.form, dtype=torch.float64)
        est_m = mass_inv if f64 else self._solver(mass_form, dtype=torch.float64)

        def via(solver):
            return lambda v: solver.apply(torch.from_numpy(v).to(self._solver_device, torch.float64)).double().cpu() \
                .numpy()

        lmin, lmax = spectral_interval(via(est_k), via(est_m), k, m)
        if not f64:
            for sv in (est_k, est_m):
                getattr(sv, "free", lambda: None)()
            del est_k, est_m
        rational = fractional_power(theta, lmin, lmax, rtol=spec.rtol)
        width = self._shifted_width(rational.degree)
        shifted = [self._solver(ShiftedForm(spec.form, t), width=width) for t in rational.shifts]
        self.first = RationalSolver(rational, shifted, mass_inv)
        self.n_mass_steps = n
        self.plan["structure"] = (
            f"{label or f'C = A^-{spec.power:g}'}: white noise + rational (M^-1 K)^-{theta:g} ({rational.degree} "
            f"shifted solves, rel. error {rational.rel_error:.1e} on [{lmin:.3g}, {lmax:.3g}]) + {n} integer "
            "solve(s) per sample"
        )

    def _shifted_width(self, nshift):
        """cuDSS right-hand-side width for the ``nshift`` shifted solvers of a fractional power: every solver keeps
        width x n device buffers, so the width shrinks until all of them fit in a quarter of the free device memory
        (a 1e6-dof 1D problem with ~40 shifts ran out of 40 GB at width 32 in float64)."""
        width = int(self.method.get("cudss_width", 32))
        if self.engine != "torch" or self.device.type != "cuda":
            return width
        free, _ = torch.cuda.mem_get_info(self.device)
        per_column = 2 * self.space.nfree * torch.empty(0, dtype=self._work_dtype).element_size()
        fit = int(0.25 * free // max(1, nshift * per_column))
        if fit < width:
            self.plan["shifted_width"] = f"{max(1, fit)} (device memory)"
        return max(1, min(width, fit))

    def _build_operator(self, spec: OperatorPrecision):
        """Apply-only precision: c = r(M^-1 Q) M^-1 z, r ~ lambda^-1/2, all shifts in one multi-shift PCG run."""
        from .blackbox import inverse_iteration_min, lanczos_interval
        from .rational import fractional_power
        from .solvers import MultiShiftRationalSolver

        s = self.space
        dev, wd = self._solver_device, self._work_dtype
        mass_t = sparse_to_torch(s.mass_free, dev, wd)
        if spec.convention == "dual->dual":
            apply_q = DualToDual(spec.apply, mass_t)
        else:
            apply_q = spec.apply
        self._apply_q = apply_q  # Sigma^-1 * scale^2 (primal -> dual), for apply_precision
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
        if self.noise is None:  # c = scale (M^-1 Q)^-1/2 M^-1 z has covariance scale^2 Q^-1
            self.scale = spec.scale
            return self._build_fractional(PDEPrecision(spec.form, power=1, scale=spec.scale),
                                          label="Sigma^-1 = Q assembled")
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
            # work vectors (eta, z, c, tmp, result) per thread: samples run in parallel threads
            self._work = Scratch(self._new_work)
        self._maps = []  # noise-space maps, applied to eta before the pipeline
        self._post_maps = []  # FE-space maps, applied to the pipeline's output
        # CPU engine: float64 torch copies of the noise sampler and the mass matrix (for apply_t). Shared with derived
        # Gaussians (same pipeline); the noise copy keeps the shared sampler's own set-up untouched.
        self._cpu_torch = {}
        # NumPy BLAS threads inside the NGSolve engine (see backend.numpy_blas_threads); None = unchanged.
        self._cpu_blas_threads = self.method.get("cpu_blas_threads", 1)
        self.plan.update(**self.noise.info(), **self.first.info())

    def _new_work(self):
        return (ngs.la.BaseVector(self.nrand), *(ngs.la.BaseVector(self.space.ndof) for _ in range(4)))

    # pickling (ngsrng.backend.Rebuildable): the NGSolve mass matrix is made again; the solvers and noise samplers
    # rebuild their own NGSolve objects; the device copy of the mean is made on first use
    _transient = ("_mean_t",)

    def _restore(self):
        if self.engine != "torch":
            self._ngs_mass = ngs_sparse(self.space.mass) if self.n_mass_steps else None

    @property
    def _work_dtype(self):
        """Arithmetic of the pipeline: float64, unless the torch engine on a GPU was asked for float32."""
        if self.device.type == "cuda" and self.method.get("work_dtype") == "float32":
            return torch.float32
        return torch.float64

    def _mass_noise(self):
        s, choice = self.space, self.method.get("noise", "auto")
        tuned = None
        if choice == "auto" and self.tuning is not None:
            rec = self.tuning._lookup("noise", self._platform, s.dim, s.kind, s.order, s.nfree)
            # exact samplers only; names a newer or older calibration used but this version does not know are ignored
            if rec and rec["method"] in NOISE_METHODS and rec["method"] not in ("chebyshev", "lumped") \
                    and (rec["method"] != "diagonal" or s.mass_is_diagonal):
                tuned = rec
        builders = {
            "diagonal": lambda: nz.DiagonalNoise(s.mass_free),
            "local-gram": lambda: nz.LocalGramNoise(s),
            "local-gram-explicit": lambda: nz.LocalGramNoise(s).explicit(),
            "quadrature": lambda: nz.quadrature_noise(s),
            "dense-cholesky": lambda: nz.dense_cholesky_noise(s.mass_free),
            "chebyshev": lambda: nz.ChebyshevNoise(s.mass_free, tol=self.method.get("noise_tol", 1e-6)),
            "lumped": lambda: nz.LumpedNoise(s.mass_free),
        }

        def bounded(name):
            if name == "local-gram-explicit":
                n = s.broken.ndof / max(s.mesh.ne, 1)
                if s.mesh.ne * n * (n + 1) / 2 > 4e8:
                    return "local-gram"
            return name

        noise = None
        if tuned is not None:
            try:  # a suggestion: if it cannot be built here (size limits, an exactness check), the rule decides
                noise = builders[bounded(tuned["method"])]()
                self.plan["noise_choice"] = "tuning"
                self._tuning_note("noise", tuned, True)
            except (MemoryError, RuntimeError, ValueError, NotImplementedError) as err:
                self._tuning_note("noise", tuned, False, f"{tuned['method']} failed: {type(err).__name__}")
        if noise is None:
            if choice == "auto":
                if s.mass_is_diagonal:
                    choice = "diagonal"
                else:
                    # Explicit sparse factor: the fastest exact option while its ~n^2/2 nonzeros per element fit the
                    # budget; the GEMM variant needs no per-element storage.
                    n = s.broken.ndof / max(s.mesh.ne, 1)
                    choice = "local-gram-explicit" if s.mesh.ne * n * (n + 1) / 2 <= 2e8 else "local-gram"
            noise = builders[bounded(choice)]()
        if not noise.exact:
            accuracy = noise.accuracy()
            if self._work_dtype == torch.float32:
                accuracy += ", limited to ~1e-7 by float32 arithmetic"
            self.plan["noise_accuracy"] = accuracy
            warnings.warn(f"approximate white noise ({noise.name}): {accuracy}", stacklevel=4)
        return noise

    def _form_noise(self, form):
        """Precision-shaped noise z ~ N(0, A) for the form's matrix A, or None for the rational route (white noise
        and (M^-1 A)^-1/2): on request, or automatically when A has no element-local factorisation (facet terms)."""
        s, choice = self.space, self.method.get("form_noise", "auto")
        if choice == "rational":
            return None
        if choice == "dense-cholesky":  # only on request: O(n^2) memory
            return nz.dense_cholesky_noise(s.assemble_free(form))
        try:
            return nz.LocalGramNoise(s, form=form)
        except (RuntimeError, NotImplementedError) as err:
            if choice == "local-gram":
                raise RuntimeError(f"no exact local factorisation of the precision: {err}") from err
            self.plan["form_noise_choice"] = f"rational ({err})"
            return None

    def _solver_available(self, name) -> bool:
        """Whether solver ``name`` runs on this Gaussian's engine and device (a tuned choice must, too)."""
        if name in ("direct", "cg"):
            return self.engine == "ngsolve"
        if name == "cudss":
            return (self.engine == "torch" and self.device.type == "cuda" and tn.gpu_backend() == "cuda"
                    and cudss_available())
        if name == "amg-pcg":
            return self.engine == "torch" and pyamg_available()
        return name == "torch-pcg" and self.engine == "torch"

    def _rule_solver(self, form) -> str:
        """The built-in rules: a sparse direct solver (cuDSS on NVIDIA GPUs) while it fits, else preconditioned CG."""
        if self.engine == "torch":
            if self._solver_available("cudss"):
                return "cudss"
            # no GPU sparse direct solver (ROCm, or no nvmath-python): AMG-preconditioned CG, whose cost grows
            # linearly; Jacobi PCG only without pyamg
            return "amg-pcg" if self._solver_available("amg-pcg") else "torch-pcg"
        return "direct" if self.space.nfree <= DIRECT_LIMIT[self.space.dim] else "cg"

    def _solver(self, form, dtype=None, width=None):
        """Solver for the form's matrix on the free dofs; ``dtype`` overrides the work dtype of the torch engine (the
        CPU engine is float64 anyway) and ``width`` the cuDSS right-hand-side width. With solver "auto", a calibrated
        choice is used if it is offered and available here and builds without error; otherwise the built-in rule."""
        s, choice = self.space, self.method.get("solver", "auto")
        dtype = self._work_dtype if dtype is None else dtype
        if choice != "auto":
            return self._make_solver(choice, form, dtype, width)
        rec = self._tuned_prior(self._platform)
        if rec is not None:
            m = rec["method"]
            if m.get("engine", self.engine) != self.engine or m.get("solver") not in SOLVERS[1:] \
                    or m.get("precond", "bddc") not in PRECONDITIONERS or m.get("inverse", "umfpack") not in INVERSES:
                why = f"{m.get('solver')} is not offered by this version"  # tables may name removed solvers
            elif not self._solver_available(m["solver"]):
                why = f"{m['solver']} is not available here"
            elif m["solver"] == "direct" and s.dim == 3 and s.nfree > max(DIRECT_LIMIT[3], 2 * rec["_target"]):
                why = "direct solver beyond twice its calibrated size in 3D (fill-in)"
            else:
                try:
                    sv = self._make_solver(m["solver"], form, dtype, width, m.get("inverse"), m.get("precond"))
                    self.plan["solver_choice"] = "tuning"
                    self._tuning_note("solver", rec, True)
                    return sv
                except Exception as err:  # noqa: BLE001 - a suggestion that cannot be built here yields to the rule
                    why = f"{m['solver']} failed: {type(err).__name__}: {err}"[:200]
            self._tuning_note("solver", rec, False, why)
        return self._make_solver(self._rule_solver(form), form, dtype, width)

    def _make_solver(self, name, form, dtype, width=None, inverse=None, precond=None):
        s = self.space
        if name == "direct":
            # NGSolve's sparsecholesky solve is ~45x slower than umfpack on 1D (chain) elimination trees, but
            # 5-10x faster in 2D/3D
            default = "umfpack" if s.dim == 1 else "sparsecholesky"
            return NGSolveDirect(s, form, inverse=self.method.get("inverse") or inverse or default)
        if name == "cg":
            # h1amg for H1 P1, BDDC otherwise: in 3D at 1e5 dofs h1amg cost 350 ms (P1) and 1030 ms (P2) per sample
            # against 138 and 129 ms for BDDC (GPU-box calibration, 2026-11-03)
            pre = self.method.get("precond") or precond or ("h1amg" if (s.kind == "H1" and s.order == 1) else "bddc")
            return NGSolveCG(s, form, precond=pre, tol=self.tol)
        if name == "cudss":
            if self.device.type != "cuda":
                raise ValueError("cudss needs a CUDA device")
            if tn.gpu_backend() != "cuda" or not cudss_available():
                raise RuntimeError("cuDSS is not available: it needs an NVIDIA GPU and nvmath-python "
                                   "(pip install ngsrng[gpu]); on other GPUs, solver='amg-pcg' (pip install pyamg)")
            # right-hand sides per cuDSS call; larger widths amortise the call overhead of big batches
            return CudssDirect(s.assemble_free(form), self.device, dtype,
                               width=int(width or self.method.get("cudss_width", 32)))
        if name in ("amg-pcg", "torch-pcg"):
            if self.engine != "torch":
                raise ValueError(f"{name} needs the torch engine")
            tol = max(self.tol, 1e-6 if dtype == torch.float32 else 1e-14)
            # a mass matrix is spectrally equivalent to its diagonal: Jacobi PCG converges in a few iterations
            if name == "amg-pcg" and form is not mass_form:
                if not pyamg_available():
                    raise RuntimeError("pyamg is not available: amg-pcg needs it (pip install pyamg)")
                return TorchAMGPCG(s.assemble_free(form), self.device, dtype, tol=tol)
            return TorchPCG(s.assemble_free(form), self.device, dtype, tol=tol)
        raise ValueError(f"unknown solver {name!r}")

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

    def _ngs_one(self, work, out):
        """Run the CPU pipeline on the noise in work[0]; writes the zero-mean full-dof sample into ``out``."""
        eta, z, c, tmp = work[:4]
        for m in reversed(self._maps):
            m.apply_ngs(eta)
        self.noise.ngs_apply(eta, z)
        self.first.ngs_apply(z, c)
        for _ in range(self.n_mass_steps):
            tmp.data = self._ngs_mass * c
            self.solver.ngs_apply(tmp, c)
        out.data = self.scale * c
        for pm in self._post_maps:
            pm.apply_ngs(out)

    @property
    def _threads(self) -> int:
        """Threads for batches in the CPU engine: one sample per thread at a time. Below ~1000 dofs a sample is too
        cheap for threads to pay off (the per-sample Python work holds the GIL)."""
        if "threads" in self.method:
            return max(1, int(self.method["threads"]))
        return available_threads() if self.space.ndof >= 1000 else 1

    @property
    def _threadsafe(self) -> bool:
        """Whether samples may run in parallel threads: every solver of the pipeline must allow concurrent solves
        (BDDC-preconditioned CG does not)."""
        return all(getattr(sv, "threadsafe", True) for sv in (self.first, self.solver))

    def _ngs_rows(self, n, out, *, eta=None, free_only=False, evaluate=None, add_mean=True, transform=None):
        """CPU engine: n samples, row k of ``out`` (n x width, float32 or float64, any strides) for sample k.

        A row is the coefficient vector (plus the mean unless ``add_mean=False``; then ``transform``, a function
        changing the float64 coefficient vector in place, e.g. an affine map), its free part (``free_only``), or
        ``evaluate`` (an NGSolve matrix, e.g. point evaluation) applied to it. ``eta`` (nrand x n) replaces fresh
        noise. Samples run in parallel threads; sample k uses the k-th reserved key of the normal stream, so results
        equal a sequential loop of single samples, for any number of threads, and every thread writes its row of
        ``out`` directly (one cast at the end, no intermediate).

        Every sample is computed single-threaded (NGSolve's kernels outside its task manager, BLAS at one thread),
        a single sample too: then a draw repeats bit for bit, and a single draw equals the matching column of a
        batch. NGSolve's task parallelism would speed up a lone sample (1.6-2.7x on 6 cores, measured), but its
        threads add partial results in a varying order (~1e-16 differences from draw to draw). Only solvers that
        must not run concurrently (BDDC-preconditioned CG) keep it, one sample at a time, and repeat to rounding.
        """
        ticket = self.stream.reserve(n) if eta is None else None
        mean, free = self._mean, self.space.free

        def work(lo, hi):
            w = self._work.get()
            e, res = w[0].FV().NumPy(), w[4]
            r = res.FV().NumPy()
            ev = evaluate.CreateColVector() if evaluate is not None else None
            for k in range(lo, hi):
                if eta is None:
                    self.stream.fill_keyed(e, ticket, k)
                else:
                    e[:] = eta[:, k]
                self._ngs_one(w, res)
                if add_mean:
                    r += mean
                if transform is not None:
                    transform(r)
                if ev is not None:
                    ev.data = evaluate * res
                    out[k] = ev.FV().NumPy()
                else:
                    out[k] = r[free] if free_only else r

        with numpy_blas_threads(self._cpu_blas_threads):
            if self._threadsafe:  # one sample per thread, single-threaded kernels: bit for bit, any thread count
                parallel_ranges(n, work, self._threads)
            else:
                with ngs.TaskManager():  # one sample at a time with NGSolve's task parallelism inside (BDDC): repeats
                    work(0, n)           # to rounding, its threads add in a varying order
        return out

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
        rows = self._ngs_rows(e.shape[1], np.empty((e.shape[1], self.space.nfree)), eta=e, free_only=True,
                              add_mean=False)
        t = torch.from_numpy(rows.T)
        return t[:, 0] if squeeze else t

    def apply_t(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint of ``apply``: free-dof vectors (nfree x k) -> noise space (nrand x k).

        With Sigma = L L^T the covariance of ``apply``, this is L^T. The solve layer is symmetric, so L^T reuses the
        same solves in reverse order followed by the noise adjoint H^T. Used to set up posteriors.
        """
        squeeze = y.ndim == 1
        x = y[:, None] if squeeze else y
        if self.engine == "torch":
            dev, wd, mass, noise = self.device, self._work_dtype, self._mass, self.noise
        else:
            dev, wd = torch.device("cpu"), torch.float64
            noise = self._cpu_torch.get("noise")
            if noise is None:  # a torch-side copy: the shared sampler is not re-set-up behind other Gaussians' backs
                noise = self._cpu_torch["noise"] = copy.copy(self.noise).setup(dev, wd)
            mass = self._mass_cpu() if self.n_mass_steps else None
        x = x.to(dev, wd)
        for pm in reversed(self._post_maps):
            x = pm.apply_t_torch(x)
        # L = scale (K^-1 M)^n R H  =>  L^T = scale H^T R (M K^-1)^n  (R = first stage, symmetric)
        v = x
        for _ in range(self.n_mass_steps):
            v = mass @ self.solver.apply(v)
        v = self.first.apply(v)
        eta = self.scale * noise.apply_t(v)
        for m in self._maps:
            eta = m.apply_torch(eta)
        return eta[:, 0] if squeeze else eta

    def _mass_cpu(self):
        if "mass" not in self._cpu_torch:
            self._cpu_torch["mass"] = sparse_to_torch(self.space.mass_free, torch.device("cpu"), torch.float64)
        return self._cpu_torch["mass"]

    def apply_precision(self, c: torch.Tensor) -> torch.Tensor:
        """Prior precision Sigma^-1 (primal -> dual) on free-dof vectors (nfree x k), returned in float64 on the CPU.

        Integer-power PDE priors: Sigma^-1 = scale^-2 K (M^-1 K)^(power-1), using a sparse factorisation of M
        (diagonal M is free). Assembled precisions: scale^-2 Q. Apply-only precisions: their own action (on the
        solver device). Fractional powers are sampled through a rational approximation whose precision has no closed
        form, so they raise NotImplementedError.
        """
        if self._maps or self._post_maps:
            raise ValueError("apply_precision is defined for priors, not for conditioned Gaussians")
        if isinstance(self.spec, PDEPrecision) and not self.spec.is_integer:
            raise NotImplementedError(f"apply_precision needs an integer power (got {self.spec.power:g}): fractional "
                                      "powers are sampled with a rational approximation, whose precision has no "
                                      "closed form")
        if isinstance(self.spec, OperatorPrecision):
            x = c[:, None] if c.ndim == 1 else c
            out = self._apply_q(x.to(self._solver_device, self._work_dtype)).double().cpu() / self.scale**2
            return out[:, 0] if c.ndim == 1 else out
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

    def _derive(self, label, mean=None, *, eta_map=None, post_map=None, seed=None, exact=None):
        """A new Gaussian sharing this one's pipeline, with an extra noise-space or FE-space map and/or a new mean.

        ``exact``: the exact covariance update the maps implement (or approximate, for randomised factorisations),
        recorded for :func:`ngsrng.reference.exact_covariance`: ``("observe", G_free, whitening kind, whitening data,
        noise scaling)`` or ``("downdate", B_free, K)``."""
        new = copy.copy(self)
        new._conditioning = getattr(self, "_conditioning", []) + [exact if exact is not None else ("unknown", label)]
        new.__dict__.pop("_mean_t", None)  # device copy of the parent's mean (see _torch_rows)
        new._maps = self._maps + ([eta_map] if eta_map is not None else [])
        new._post_maps = self._post_maps + ([post_map] if post_map is not None else [])
        new._mean = self._mean.copy() if mean is None else mean
        new._prior = self if self._prior is None else self._prior
        new.plan = {**self.plan, "conditioning": self.plan.get("conditioning", []) + [label]}
        new.source = inherit(seed, self)  # seed=None: a child stream of this one (or the same draw-time generator)
        new.plan["randomness"] = new.source.describe()
        if self.engine != "torch":
            new.stream = NormalStream(new.source)
        return new

    @property
    def generator(self):
        """Torch engine: the torch.Generator of the current draw (the own stream's, which persists, for an explicit
        seed)."""
        return self.source.streams().torch(self.device) if self.engine == "torch" else None

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

        ``out``: a buffer (ndof x n; NumPy or torch, any float dtype and device) filled in place and returned, the
        in-place form (see :meth:`sample_into`); ``"multivector"`` returns an NGSolve MultiVector instead.
        """
        if out is not None and not isinstance(out, str):
            return self._sample_out(n, out)
        ns = 1 if n is None else int(n)
        if self.engine == "torch":
            return self._sample_torch(n, ns, out)
        rows = self._ngs_rows(ns, np.empty((ns, self.space.ndof)))
        if n is None:
            gf = ngs.GridFunction(self.space.fes)
            gf.vec.FV().NumPy()[:] = rows[0]
            return gf
        if out == "multivector":
            mv = ngs.MultiVector(ngs.GridFunction(self.space.fes).vec, ns)
            for k in range(ns):
                mv[k].FV().NumPy()[:] = rows[k]
            return mv
        return torch.from_numpy(rows).T.to(self.dtype)

    def sample_into(self, rows):
        """Fill rows (k x ndof, one sample per row; NumPy or torch, any float dtype and device) in place
        (:mod:`ngsrng.inplace`). The CPU engine writes every sample from its worker thread straight into its row; the
        torch engine works on its device in chunks of ``work_values`` values."""
        rows = as_tensor(rows)
        if tuple(rows.shape[1:]) != (self.space.ndof,):
            raise ValueError(f"out holds {tuple(rows.shape[1:])} values per sample, the Gaussian gives "
                             f"({self.space.ndof},)")
        if self.engine == "ngsolve" and rows.device.type == "cpu" and rows.dtype in (torch.float32, torch.float64):
            self._ngs_rows(rows.shape[0], rows.numpy())
            return rows
        for k0, k1 in chunks(rows.shape[0], self._per_sample(), self.work_values, self._min_chunk()):
            if self.engine == "torch" and rows.device == self.device and rows.dtype == self.dtype:
                self._torch_rows(rows[k0:k1])
            else:
                write(rows[k0:k1], self.sample(k1 - k0))
        return rows

    def _torch_rows(self, rows, loc=None, scale=1.0):
        """Torch engine: fill rows (k x ndof, on the device, in self.dtype) with samples, without the transposed
        intermediate of ``sample``; same random numbers and arithmetic as ``sample(k)``. With ``loc`` (a device
        tensor) and ``scale``: loc + scale (sample - mean)."""
        k = rows.shape[0]
        eta = torch.randn(self.nrand, k, generator=self.generator, device=self.device, dtype=self._work_dtype)
        c = self.apply(eta)
        c = (c if scale == 1.0 else c.mul_(scale)).to(self.dtype)
        # device copy of the mean, keyed by the mean array itself (posteriors replace it: posterior(mean=...))
        cache = getattr(self, "_mean_t", None)
        if cache is None or cache[0] is not self._mean or cache[1].dtype != self.dtype:
            self._mean_t = cache = (self._mean, torch.from_numpy(self._mean).to(self.device, self.dtype),
                                    torch.from_numpy(self.space.free).to(self.device))
        _, mean_t, free_t = cache
        rows.copy_((mean_t if loc is None else loc).expand(k, -1))
        rows[:, free_t] += c.T
        return rows

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


class ShiftedForm:
    """The form a(u, v) + t (u, v): the shifted systems of fractional powers (a class, not a closure: it pickles)."""

    def __init__(self, form, t):
        self.form, self.t = form, float(t)

    def __call__(self, u, v):
        return self.form(u, v) + self.t * u * v * ngs.dx


class DualToDual:
    """Sigma^-1 c = Q_dd (M c) for a precision given as a dual-space operator Q_dd (apply-only specs)."""

    def __init__(self, apply, mass):
        self.apply, self.mass = apply, mass

    def __call__(self, c):
        return self.apply(self.mass @ c)
