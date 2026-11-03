"""Benchmark grid: profiles, spaces, sizes, and every (engine, device, dtype, method, solver) to run.

A *group* is one (suite, dim, kind, order, target size) and runs in one worker process, which builds the mesh and
space once. A *task* is one configuration inside a group and produces one result record.

The default profile, ``calibrate``, exists to choose this machine's white-noise sampler and linear solver: it measures
only candidates that can win, checks every measurement for accuracy, and drops a candidate for larger sizes once it is
inaccurate, fails, or is far slower than the best one (``RETIRE_RATIO``). Its predecessor measured every method on
every grid point and needed days on the GPU box; the measured losers are no longer candidates:

- white noise: ``chebyshev`` (60-3400x slower than the exact samplers), ``lumped`` (covariance error 150% for H1 P1;
  identical to ``diagonal`` where the mass matrix is diagonal; undefined for H1 p >= 2) and ``dense-cholesky``
  (O(n^2) memory) -- the library still offers them on request;
- solvers: ``cg-local`` (up to 390x slower than a direct solver), ``cg-multigrid`` (crashed: needs a mesh hierarchy),
  ``direct-pardiso`` (not available in NGSolve builds without MKL), the apply-only multi-shift route (a feature for
  precisions known only as an action, not a solver choice), float32 GPU arithmetic (pathwise errors up to O(1)).
"""

from dataclasses import asdict, dataclass, field

# Settings of the research suites (posterior, distributions, lowrank, covariance, continuum); every profile can run
# them with --suites, but only "research" does so by default.
_RESEARCH = dict(
    torch_cpu_max_ndof=1e5,
    continuum_max_ndof=5e4,
    continuum_orders={"H1": [1, 2, 3], "L2": [0, 1, 2]},
    posterior_obs=[10, 100],
    posterior_orders={"H1": [1, 2], "L2": [0, 1]},
)
CALIBRATION_SUITES = ["noise", "prior"]
RESEARCH_SUITES = ["noise", "prior", "posterior", "distributions", "lowrank", "covariance", "continuum"]

PROFILES = {
    # a few minutes on any machine: the harness end to end
    "smoke": dict(_RESEARCH, suites=CALIBRATION_SUITES, targets=[2e3], dims=[2, 3], orders={"H1": [1, 2], "L2": [1]},
                  powers=[2], method_timeout=120, timeout_floor=30, setup_budget=30, sample_budget=2.0,
                  torch_cpu_max_ndof=0, continuum_max_ndof=5e3, posterior_obs=[10]),
    # the default: calibrates this machine's automatic choices (2D and 3D, the common orders, 1e4 and 1e5 dofs)
    # plus one probe at 2.5e5 dofs (prior only) where an iterative solver could overtake a factorisation: 3D H1 P1,
    # just above the built-in direct limit (2e5 in 3D)
    "calibrate": dict(_RESEARCH, suites=CALIBRATION_SUITES, targets=[1e4, 1e5], dims=[2, 3],
                      orders={"H1": [1, 2], "L2": [1]}, powers=[2], probes=[(3, "H1", 1, 2.5e5)],
                      method_timeout=300, timeout_floor=30, setup_budget=60, sample_budget=2.0,
                      torch_cpu_max_ndof=0),
    # calibrate, extended to 1e6 dofs (large CPU/GPU boxes)
    "large": dict(_RESEARCH, suites=CALIBRATION_SUITES, targets=[1e4, 1e5, 1e6], dims=[2, 3],
                  orders={"H1": [1, 2], "L2": [1]}, powers=[2], method_timeout=900, timeout_floor=60,
                  setup_budget=300, sample_budget=5.0, torch_cpu_max_ndof=0),
    # research data (all suites, more orders and powers, 1D): hours, not part of calibration
    "research": dict(_RESEARCH, suites=RESEARCH_SUITES, targets=[1e3, 1e4, 1e5], dims=[1, 2, 3],
                     orders={"H1": [1, 2, 4], "L2": [0, 1, 2]}, powers=[2, 3, 1.6], method_timeout=600,
                     timeout_floor=30, setup_budget=120, sample_budget=10.0),
}

# Exact white-noise samplers (the only candidates: an approximate sampler is not a faster version of the same law).
NOISE_METHODS = ["diagonal", "local-gram", "local-gram-explicit", "quadrature"]

# (solver label, Gaussian method overrides); "precond" choices are NGSolve preconditioner names.
CPU_SOLVERS = {
    "direct-sparsecholesky": {"solver": "direct", "inverse": "sparsecholesky"},
    "direct-umfpack": {"solver": "direct", "inverse": "umfpack"},
    "cg-bddc": {"solver": "cg", "precond": "bddc"},
    "cg-h1amg": {"solver": "cg", "precond": "h1amg"},
}
GPU_SOLVERS = {"cudss": {"solver": "cudss"}, "pcg-jacobi": {"solver": "torch-pcg"}}
ITERATIVE = {"cg-bddc", "cg-h1amg", "pcg-jacobi"}

# Accuracy a measurement must reach to count (all calibrated arithmetic is float64): white noise reproduces M to
# max|HH^T - M|/max|M| (or its probe estimate), priors match the float64 direct reference sample for sample.
ACCURACY = {"noise": 1e-10, "prior": 1e-6}
# A candidate slower than RETIRE_RATIO x the best accurate one of its group, in every mode (single samples and
# batches), is not measured at larger sizes. Iterative solvers in 3D are exempt: their cost grows more slowly than a
# factorisation's, so they may win at larger sizes.
RETIRE_RATIO = 20.0
# Samples per timed batch: one per CPU thread (the CPU engine runs one sample per thread; 4 left 32 cores idle and
# understated batch throughput), and 256 on a GPU (realistic batches; 32 is a single cuDSS block).
GPU_BATCH = 256
# One recommendation serves single samples and batches alike, so candidates are ranked by their worst ratio to the
# best single-sample and the best batch time. Candidates within TIE of the best score are tied, and a tie goes to an
# exact solver (in the first calibration, batched PCG won a 4% batch tie while 11x slower for single samples).
MODES = ("per_sample_s", "per_sample_batch_s")
TIE = 1.10


def mode_ratios(recs):
    """Each record of one case and platform with its time ratio to the best one, per mode it measured."""
    def val(r, k):
        v = r.get(k)
        return v if isinstance(v, (int, float)) and v == v and v > 0 else None

    best = {k: min((val(r, k) for r in recs if val(r, k)), default=None) for k in MODES}
    return [(r, {k: val(r, k) / best[k] for k in MODES if best[k] and val(r, k)}) for r in recs]


def choose(recs):
    """The record to recommend among the accurate records of one case and platform (see TIE)."""
    scored = sorted(((max(q.values(), default=float("inf")), i, r) for i, (r, q) in enumerate(mode_ratios(recs))),
                    key=lambda x: (x[0], x[1]))
    top = scored[0][0]
    for score, _, r in scored:
        if score <= TIE * top and r.get("solver") not in ITERATIVE:
            return r
    return scored[0][2]

# Posterior (linear Gaussian observations): m local averages of the prior field. The exact low-rank update applied
# in FE space ("fe-space") or to the white noise ("noise-space"), its randomised rank-32 version by randomised
# subspace iteration ("fe-space-rank32") or by plain power iteration ("fe-space-rank32-power"),
# and pathwise conditioning ("matheron").
# fe-space-reweighted: the fe-space posterior plus the cost of re-weighting the noise (posterior(noise_scaling=...),
# the Gaussian step of IRLS and of Gibbs samplers for heavy-tailed noise)
POSTERIOR_METHODS = ["fe-space", "fe-space-reweighted", "noise-space", "fe-space-rank32", "fe-space-rank32-power",
                     "matheron"]
POSTERIOR_DENSE_LIMIT = 4e8  # setup keeps G (m x ndof) and L^T G^T (nrand x m) dense

# Distribution families on the default Matern prior (power 2), through ngsrng.Generator. Every algorithm variant of a
# family is timed separately (inversion: closed form / Hermite table / scipy ppf / discrete table; constructions:
# squares vs copula, Wishart process vs Bartlett, t-process vs t-copula), so the automatic choices can be fitted.
# name -> (generator method, args, keyword options)
DISTRIBUTION_CASES = {
    "uniform": ("uniform", (), {}),
    "lognormal": ("lognormal", (0.0, 0.5), {}),
    "gamma-table": ("gamma", (2.0,), {"method": "table"}),
    "gamma-ppf": ("gamma", (2.0,), {"method": "ppf"}),
    "beta-table": ("beta", (2.0, 5.0), {"method": "table"}),
    "beta-ppf": ("beta", (2.0, 5.0), {"method": "ppf"}),
    "poisson-table": ("poisson", (3.0,), {"method": "discrete-table"}),
    "poisson-ppf": ("poisson", (3.0,), {"method": "ppf"}),
    "chisquare3-squares": ("chisquare", (3,), {"construction": "squares"}),
    "chisquare3-copula": ("chisquare", (3,), {"construction": "copula"}),
    "wishart2-df5-process": ("wishart", (5, "eye2"), {"construction": "process"}),
    "wishart2-df5-bartlett": ("wishart", (5, "eye2"), {"construction": "bartlett"}),
    "wishart3-df10-process": ("wishart", (10, "eye3"), {"construction": "process"}),
    "wishart3-df10-bartlett": ("wishart", (10, "eye3"), {"construction": "bartlett"}),
    "matrix-lognormal-2": ("matrix_lognormal", (2, 0.5), {}),
    "matrix-lognormal-3": ("matrix_lognormal", (3, 0.5), {}),
    "dirichlet-3": ("dirichlet", ((1.0, 2.0, 3.0),), {}),
    "student-t5-process": ("standard_t", (5,), {"construction": "process"}),
    "student-t5-copula": ("standard_t", (5,), {"construction": "copula"}),
    "nig": ("normal_inverse_gaussian", (0.05,), {}),
    "variance-gamma": ("variance_gamma", (0.05,), {}),
}

# Truncated Karhunen-Loeve expansions of the default Matern prior (power 2): randomised subspace iteration on the
# factor L ("subspace") vs range finder on the covariance with Rayleigh-Ritz ("covariance"), several ranks and
# iteration counts; accuracy against the dense eigendecomposition where it fits (LOWRANK_DENSE_LIMIT free dofs).
LOWRANK_METHODS = ["subspace", "covariance"]
LOWRANK_RANKS = [20, 100]
LOWRANK_ITERATIONS = [1, 2, 4]
LOWRANK_DENSE_LIMIT = 4000

# Covariance diagnostics (ngsrng.covariance) of the default Matern prior and of its posterior under 40 point
# observations: pointwise-variance methods (method, options) and trace estimators (probes), against the exact local
# covariance where it is affordable (COVARIANCE_EXACT_LIMIT free dofs).
COVARIANCE_VARIANCE = {
    "exact": ("exact", {}),
    "lowrank-r100-s32": ("lowrank", {"rank": 100, "samples": 32}),
    "lowrank-r400-s64": ("lowrank", {"rank": 400, "samples": 64}),
    "mc-400": ("mc", {"samples": 400}),
    "mc-2000": ("mc", {"samples": 2000}),
}
COVARIANCE_TRACE = ["hutchinson", "hutch++", "xnystrace"]
COVARIANCE_PROBES = [30, 120]
COVARIANCE_SOURCES = ["prior", "posterior"]
COVARIANCE_EXACT_LIMIT = 20000

DENSE_LIMIT = 3000  # dense-cholesky noise only up to this many free dofs (on explicit request)
DIRECT_3D_LIMIT = 3e5  # sparse direct solvers are not attempted beyond this in 3D (fill-in)


@dataclass
class Task:
    suite: str
    dim: int
    kind: str
    order: int
    target: float
    engine: str = ""
    device: str = ""
    dtype: str = ""
    method: str = ""
    solver: str = ""
    power: int = 0
    options: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return "|".join(
            str(v)
            for v in (
                self.suite,
                self.dim,
                self.kind,
                self.order,
                int(self.target),
                self.engine,
                self.device,
                self.dtype,
                self.method,
                self.solver,
                self.power,
            )
        )

    @property
    def lineage(self) -> str:
        """Identity across sizes: a task is retired for larger sizes once it fails or blows its budget."""
        return "|".join(
            str(v)
            for v in (
                self.suite,
                self.dim,
                self.kind,
                self.order,
                self.engine,
                self.device,
                self.dtype,
                self.method,
                self.solver,
                self.power,
            )
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class Group:
    suite: str
    dim: int
    kind: str
    order: int
    target: float
    tasks: list

    @property
    def id(self):
        return f"{self.suite}|{self.dim}|{self.kind}|{self.order}|{int(self.target)}"

    def to_dict(self):
        return {
            "suite": self.suite,
            "dim": self.dim,
            "kind": self.kind,
            "order": self.order,
            "target": self.target,
            "tasks": [t.to_dict() for t in self.tasks],
        }


def engines(devices, torch_cpu: bool):
    """Calibrated platforms: the CPU engine, optionally the torch engine on the CPU, and the GPU engine in float64
    arithmetic (its default; float64 output, so that the noise suite measures the same arithmetic)."""
    out = [("ngsolve", "cpu", "float64")]
    if torch_cpu:
        out.append(("torch", "cpu", "float64"))
    if "cuda" in devices:
        out.append(("torch", "cuda", "float64"))
    return out


def prior_tasks(d, k, p, t, cfg, devices):
    tasks = []
    for power in cfg["powers"]:
        for name, opts in CPU_SOLVERS.items():
            if name.startswith("direct") and d == 3 and t > DIRECT_3D_LIMIT:
                continue
            if name == "cg-h1amg" and k != "H1":
                continue
            tasks.append(Task("prior", d, k, p, t, "ngsolve", "cpu", "float64", "auto", name, power, opts))
        if "cuda" in devices:
            for name, opts in GPU_SOLVERS.items():
                tasks.append(Task("prior", d, k, p, t, "torch", "cuda", "float64", "auto", name, power, opts))
    return tasks


def build_groups(profile: str, suites, devices, max_ndof=None):
    cfg = PROFILES[profile]
    targets = [t for t in cfg["targets"] if max_ndof is None or t <= max_ndof]
    groups = []
    spaces = [(d, k, p) for d in cfg["dims"] for k in ("H1", "L2") for p in cfg["orders"][k]]
    if "noise" in suites:
        for d, k, p in spaces:
            for t in targets:
                tasks = [Task("noise", d, k, p, t, eng, dev, dt, method=m)
                         for eng, dev, dt in engines(devices, torch_cpu=t <= cfg["torch_cpu_max_ndof"])
                         for m in NOISE_METHODS]
                groups.append(Group("noise", d, k, p, t, tasks))
    if "prior" in suites:
        for d, k, p in spaces:
            for t in targets:
                groups.append(Group("prior", d, k, p, t, prior_tasks(d, k, p, t, cfg, devices)))
        for d, k, p, t in cfg.get("probes", []):
            if max_ndof is None or t <= max_ndof:
                groups.append(Group("prior", d, k, p, t, prior_tasks(d, k, p, t, cfg, devices)))
    if "posterior" in suites:
        pspaces = [(d, k, p) for d in cfg["dims"] for k in ("H1", "L2") for p in cfg["posterior_orders"][k]]
        for d, k, p in pspaces:
            for t in targets:
                tasks = []
                for m in cfg["posterior_obs"]:
                    if m * t > POSTERIOR_DENSE_LIMIT:
                        continue
                    for method in POSTERIOR_METHODS:
                        if method.startswith("fe-space-rank32") and m <= 32:
                            continue
                        platforms = [("ngsolve", "cpu", "float64")]
                        if "cuda" in devices:
                            platforms += [("torch", "cuda", "float64"), ("torch", "cuda", "float32")]
                        for eng, dev, dt in platforms:
                            tasks.append(Task("posterior", d, k, p, t, eng, dev, dt, method, f"obs{m}", 2, {"m": m}))
                groups.append(Group("posterior", d, k, p, t, tasks))
    if "distributions" in suites:
        nspaces = [(d, k, p) for d in cfg["dims"] for k in ("H1", "L2") for p in cfg["posterior_orders"][k]]
        for d, k, p in nspaces:
            for t in targets:
                platforms = [("ngsolve", "cpu", "float64")] + (
                    [("torch", "cuda", "float32")] if "cuda" in devices else []
                )
                tasks = [
                    Task("distributions", d, k, p, t, eng, dev, dt, fam, "", 2)
                    for fam in DISTRIBUTION_CASES
                    for eng, dev, dt in platforms
                ]
                groups.append(Group("distributions", d, k, p, t, tasks))
    if "lowrank" in suites:
        for d, k, p in [(d, k, p) for d in cfg["dims"] for k in ("H1", "L2") for p in cfg["posterior_orders"][k]]:
            for t in targets:
                platforms = [("ngsolve", "cpu", "float64")] + (
                    [("torch", "cuda", "float32")] if "cuda" in devices else []
                )
                tasks = [
                    Task("lowrank", d, k, p, t, eng, dev, dt, method, f"rank{r}-q{q}", 2, {"rank": r, "q": q})
                    for method in LOWRANK_METHODS
                    for r in LOWRANK_RANKS
                    for q in LOWRANK_ITERATIONS
                    for eng, dev, dt in platforms
                ]
                groups.append(Group("lowrank", d, k, p, t, tasks))
    if "covariance" in suites:
        for d, k, p in [(d, k, p) for d in cfg["dims"] for k in ("H1", "L2") for p in cfg["posterior_orders"][k]]:
            for t in targets:
                tasks = [
                    Task("covariance", d, k, p, t, "ngsolve", "cpu", "float64", f"variance:{name}", source, 2)
                    for source in COVARIANCE_SOURCES
                    for name in COVARIANCE_VARIANCE
                ] + [
                    Task("covariance", d, k, p, t, "ngsolve", "cpu", "float64", f"trace:{m}", f"{source}-p{n}", 2,
                         {"probes": n})
                    for source in COVARIANCE_SOURCES
                    for m in COVARIANCE_TRACE
                    for n in COVARIANCE_PROBES
                ]
                groups.append(Group("covariance", d, k, p, t, tasks))
    if "continuum" in suites:
        for d in cfg["dims"]:
            for k in ("H1", "L2"):
                for p in cfg["continuum_orders"][k]:
                    tasks = [
                        Task("continuum", d, k, p, cfg["continuum_max_ndof"], "ngsolve", "cpu", "float64", power=pw)
                        for pw in cfg["powers"]
                        if 2 * pw > d
                    ]
                    groups.append(Group("continuum", d, k, p, cfg["continuum_max_ndof"], tasks))
    return groups
