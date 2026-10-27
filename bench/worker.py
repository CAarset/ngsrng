"""Run one benchmark group in an isolated process (invoked by run_all.py).

Protocol: for each task, prints ``START <task id>`` before and ``DONE <task id>`` after appending its JSON record to
the output file. If the process dies or hangs, the orchestrator attributes the failure to the started task.
A watchdog thread exits with code 99 if the resident memory exceeds ``--mem-limit-gb``, so that this process
fails cleanly before the kernel OOM killer picks a victim.
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
import warnings

import ngsolve as ngs
import numpy as np
import scipy.linalg as sla
import torch

# Benchmarks measure the built-in rule-based "auto" choices, never a previously installed tuning.
os.environ["NGSRNG_TUNING"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no-tuning.json")
sys.path.insert(0, os.path.dirname(__file__))
from cases import DENSE_LIMIT, Task  # noqa: E402
from common import make_space, structured_mesh  # noqa: E402

import ngsrng  # noqa: E402
from ngsrng import noise as nz  # noqa: E402
from ngsrng.backend import NormalStream  # noqa: E402
from ngsrng.reference import covariance_errors, exact_covariance, sampler_covariance  # noqa: E402
from ngsrng.space import SpaceInfo  # noqa: E402

KAPPA = 5.0
EXACT_LIMIT = 1500
MEMLIMIT_EXIT = 99


def watchdog(limit_bytes):
    page = os.sysconf("SC_PAGE_SIZE")

    def run():
        while True:
            try:
                with open("/proc/self/statm") as fh:
                    rss = int(fh.read().split()[1]) * page
            except OSError:
                return
            if rss > limit_bytes:
                print(f"MEMLIMIT rss={rss / 2**30:.1f}GB", flush=True)
                os._exit(MEMLIMIT_EXIT)
            time.sleep(0.05)

    threading.Thread(target=run, daemon=True).start()


def timeit(fn, sync=None, min_time=0.3, max_reps=20, min_reps=2):
    """Median wall time of fn(); repeats until min_time has been spent (at least min_reps calls)."""
    times, spent, out = [], 0.0, None
    while len(times) < min_reps or (spent < min_time and len(times) < max_reps):
        if sync:
            sync()
        t = time.perf_counter()
        out = fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t)
        spent += times[-1]
    return float(np.median(times)), out


def sync_for(device):
    return torch.cuda.synchronize if device == "cuda" else None


def space_for(task: Task):
    kw = {"dgjumps": True} if task.kind == "L2" else {}
    return SpaceInfo(make_space(task.dim, task.kind, task.order, task.target, **kw))


# ------------------------------------------------------------------ noise suite


def noise_sampler(name, space):
    if name == "diagonal":
        if not space.mass_is_diagonal:
            return None
        return nz.DiagonalNoise(space.mass_free)
    if name == "local-gram":
        return nz.LocalGramNoise(space)
    if name == "local-gram-explicit":
        lg = nz.LocalGramNoise(space)
        n = max((p[1].shape[1] for p in lg.ref_parts), default=0)
        if space.mesh.ne * n * (n + 1) / 2 > 4e8:
            raise MemoryError("explicit factor would exceed 4e8 nonzeros")
        return lg.explicit()
    if name == "quadrature":
        return nz.quadrature_noise(space)
    if name.startswith("chebyshev"):
        return nz.ChebyshevNoise(space.mass_free, tol=float(name.split("-", 1)[1]))
    if name == "lumped":
        try:
            return nz.LumpedNoise(space.mass_free)
        except ValueError:
            return None
    if name == "dense-cholesky":
        return nz.dense_cholesky_noise(space.mass_free) if space.nfree <= DENSE_LIMIT else None
    raise ValueError(name)


def run_noise(task: Task, space: SpaceInfo, rec: dict):
    t = time.perf_counter()
    smp = noise_sampler(task.method, space)
    if smp is None:
        rec["status"] = "n/a"
        return
    dtype = getattr(torch, task.dtype)
    if task.engine == "ngsolve":
        smp.ngs_setup(space)
    else:
        smp.setup(torch.device(task.device), dtype)
    rec["setup_s"] = time.perf_counter() - t
    rec["nrand"] = smp.nrand
    rec["exact"] = smp.exact
    for k, v in smp.info().items():
        if k not in ("noise", "nrand", "exact", "spectrum"):
            rec[k] = v
    if task.engine == "ngsolve":
        eta, out = ngs.la.BaseVector(smp.nrand), ngs.la.BaseVector(space.ndof)
        stream = NormalStream(0)
        with ngs.TaskManager():
            rec["rng_s"], _ = timeit(lambda: stream.fill(eta.FV().NumPy()))
            rec["apply_s"], _ = timeit(lambda: smp.ngs_apply(eta, out))
        rec["batch"] = 1
    else:
        dev, sync = torch.device(task.device), sync_for(task.device)
        gen = torch.Generator(device=dev).manual_seed(0)
        batch = int(task.options.get("batch", 32))
        rng1, eta1 = timeit(lambda: torch.randn(smp.nrand, 1, generator=gen, device=dev, dtype=dtype), sync)
        app1, _ = timeit(lambda: smp.apply(eta1), sync)
        rngb, etab = timeit(lambda: torch.randn(smp.nrand, batch, generator=gen, device=dev, dtype=dtype), sync)
        appb, _ = timeit(lambda: smp.apply(etab), sync)
        rec.update(rng_s=rng1, apply_s=app1, batch=batch, per_sample_batch_s=(rngb + appb) / batch)
        del eta1, etab
    rec["per_sample_s"] = rec["rng_s"] + rec["apply_s"]
    if space.nfree <= EXACT_LIMIT * 2:
        # Full factor only when cheap: it costs nrand applications (each a degree-sized polynomial for Chebyshev).
        if smp.nrand <= 2000 and not task.method.startswith("chebyshev"):
            h = noise_factor(smp, task, space, rec)
            m = space.mass_free.toarray()
            err = float(np.abs(h @ h.T - m).max() / np.abs(m).max())
        else:
            err = noise_probe_error(smp, task, space, rec)
        rec["cov_rel_err"] = max(err, rec.get("engine_mismatch", 0.0))


def noise_probe_error(smp, task, space, rec, nprobe=8):
    """max_k |H H^T v_k - M v_k| / |M v_k| on random probes, via the sampler's exact adjoint (cheap for large nrand)."""
    if task.engine == "ngsolve":
        noise_factor(smp, task, space, rec, mismatch_only=True)
        dev, dtype = torch.device("cpu"), torch.float64
        smp.setup(dev, dtype)
    else:
        dev, dtype = torch.device(task.device), getattr(torch, task.dtype)
    v = np.random.default_rng(0).standard_normal((space.nfree, nprobe))
    hht = smp.apply(smp.apply_t(torch.from_numpy(v).to(dev, dtype))).double().cpu().numpy()
    mv = space.mass_free @ v
    return float(np.max(np.linalg.norm(hht - mv, axis=0) / np.linalg.norm(mv, axis=0)))


def noise_factor(smp, task, space, rec, mismatch_only=False):
    """The sampler's matrix H (nfree x nrand), computed with the task's own engine, device and dtype.

    For the NGSolve engine, H comes from the torch path (CPU, float64) and the NGSolve engine is checked against it on
    random probes (``engine_mismatch``); applying it column by column would dominate the benchmark.
    """
    if task.engine == "ngsolve":
        smp.setup(torch.device("cpu"), torch.float64)
        probes = np.random.default_rng(0).standard_normal((smp.nrand, 4))
        ref = smp.apply(torch.from_numpy(probes)).numpy()
        eta, out = ngs.la.BaseVector(smp.nrand), ngs.la.BaseVector(space.ndof)
        mismatch = 0.0
        with ngs.TaskManager():
            for j in range(probes.shape[1]):
                eta.FV().NumPy()[:] = probes[:, j]
                smp.ngs_apply(eta, out)
                got = out.FV().NumPy()[space.free]
                mismatch = max(mismatch, np.linalg.norm(got - ref[:, j]) / np.linalg.norm(ref[:, j]))
        rec["engine_mismatch"] = float(mismatch)
        if mismatch_only:
            return None
        task = Task(**{**task.to_dict(), "engine": "torch", "device": "cpu", "dtype": "float64"})
    dtype = getattr(torch, task.dtype)
    cols = []
    for start in range(0, smp.nrand, 4096):
        k = min(4096, smp.nrand - start)
        eye = torch.zeros(smp.nrand, k, device=task.device, dtype=dtype)
        eye[start + torch.arange(k), torch.arange(k)] = 1
        cols.append(smp.apply(eye).double().cpu().numpy())
    return np.hstack(cols)


# ------------------------------------------------------------------ prior suite


def prior_spec(fes, kind, power):
    return ngsrng.matern(KAPPA, power) if kind == "H1" else ngsrng.matern_dg(fes, KAPPA, power)


class Reference:
    """Pathwise reference: float64 CPU sampler with the same noise and a direct (or tight CG) solve."""

    def __init__(self, fes, task):
        self.fes, self.task, self.cache = fes, task, {}

    def get(self, power, nrand, eta):
        if power not in self.cache:
            space = SpaceInfo(self.fes)
            direct_ok = not (space.dim == 3 and space.nfree > 3e5)
            method = (
                {"solver": "direct"}
                if direct_ok
                else {"solver": "cg", "precond": "bddc" if space.kind == "H1" else "local"}
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = ngsrng.Gaussian(
                    self.fes, prior_spec(self.fes, self.task.kind, power), method=method, tol=1e-12, dtype="float64"
                )
            self.cache[power] = (g, g.apply(eta).numpy() if g.nrand == nrand else None, method)
        return self.cache[power][1], self.cache[power][2]


def apply_only_spec(fes, kind, device, dtype):
    """The power-2 prior as an apply-only precision c -> K M^-1 K c (sparse K, a mass solver)."""
    from ngsrng.backend import sparse_to_torch
    from ngsrng.solvers import CudssDirect, NGSolveDirect

    space = SpaceInfo(fes)
    dev, wd = torch.device(device), getattr(torch, dtype)
    k = sparse_to_torch(space.assemble_free(prior_spec(fes, kind, 2).form), dev, wd)
    if device == "cuda":
        minv = CudssDirect(space.mass_free, dev, wd).apply
    else:
        minv = NGSolveDirect(space, ngsrng.space.mass_form).apply
    return ngsrng.OperatorPrecision(lambda c: k @ minv(k @ c.to(dev, wd)), tol=1e-6 if dtype == "float32" else 1e-8)


def run_prior(task: Task, fes, rec: dict, ref: Reference):
    method = {**task.options, "engine": task.engine}
    tol = 1e-6 if task.dtype == "float32" else 1e-8
    t = time.perf_counter()
    if task.solver == "apply-only":
        spec = apply_only_spec(fes, task.kind, task.device, task.dtype)
    else:
        spec = prior_spec(fes, task.kind, task.power)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = ngsrng.Gaussian(
            fes,
            spec,
            method=method,
            device=task.device,
            dtype=task.dtype,
            tol=tol,
            seed=0,
        )
    rec["setup_s"] = time.perf_counter() - t
    rec.update(nrand=g.nrand, noise=g.plan.get("noise"), solver_name=g.plan.get("solver"), tol=tol)
    sync = sync_for(task.device)
    rec["per_sample_s"], _ = timeit(lambda: g.sample(1), sync, min_time=0.5, max_reps=5)
    batch = 4 if task.engine == "ngsolve" else int(task.options.get("batch", 32))
    tb, _ = timeit(lambda: g.sample(batch), sync, min_time=0.0, max_reps=1, min_reps=1)
    rec.update(batch=batch, per_sample_batch_s=tb / batch)
    its = getattr(g.solver, "iterations", None) or getattr(getattr(g.solver, "cg", None), "iterations", None)
    if its:
        rec["cg_iterations"] = float(np.mean(its))
    eta = torch.randn(g.nrand, 2, generator=torch.Generator().manual_seed(123), dtype=torch.float64)
    c = g.apply(eta).double().cpu().numpy()
    c_ref, ref_method = ref.get(task.power, g.nrand, eta)
    if c_ref is not None:
        rec["pathwise_rel_err"] = float(np.max(np.linalg.norm(c - c_ref, axis=0) / np.linalg.norm(c_ref, axis=0)))
        rec["pathwise_reference"] = ref_method["solver"]
    # Exact law: only for linear pipelines (direct solves) and the batched GPU path. Iterative CPU solvers are judged
    # by their pathwise error; pushing every noise direction through them would dominate the benchmark.
    linear_or_batched = task.solver.startswith("direct") or task.engine == "torch"
    if linear_or_batched and g.space.nfree <= EXACT_LIMIT and g.nrand <= 4 * EXACT_LIMIT:
        rec.update({f"law_{k}": v for k, v in covariance_errors(sampler_covariance(g), exact_covariance(g)).items()})


# ------------------------------------------------------------------ posterior suite


class PosteriorContext:
    """Per-group cache: priors per platform, observation sets, and float64 reference predictive covariances."""

    def __init__(self, fes, kind):
        self.fes, self.kind = fes, kind
        self.priors, self.obs, self.refs = {}, {}, {}

    def prior(self, engine, device, dtype):
        key = (engine, device, dtype)
        if key not in self.priors:
            tol = 1e-6 if dtype == "float32" else 1e-10
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = ngsrng.Gaussian(
                    self.fes,
                    prior_spec(self.fes, self.kind, 2),
                    device=device,
                    dtype=dtype,
                    tol=tol,
                    seed=0,
                    method={"engine": engine},
                )
            g.sample(2)
            sync = sync_for(device)
            g._bench_per_sample, _ = timeit(lambda: g.sample(1), sync, min_time=0.3, max_reps=5)
            self.priors[key] = g
        return self.priors[key]

    def centers(self, m):
        if m not in self.obs:
            dim = self.fes.mesh.dim
            self.obs[m] = np.random.default_rng(m).uniform(0.1, 0.9, (m, dim))
        return self.obs[m]

    def reference(self, m):
        """Float64 CPU Matheron: predictive covariance G Sigma_post G^T and the posterior mean, for cross-checks."""
        if m not in self.refs:
            from ngsrng.posterior import LinearObservations, MatheronPosterior

            prior = self.prior("ngsolve", "cpu", "float64")
            y, var = self.data(m)
            obs = LinearObservations(prior, averages=(self.centers(m), 0.05, "gaussian"), noise=var, factorize=False)
            mat = MatheronPosterior(obs, data=y)
            gs = obs.gf.toarray() @ mat.bmat  # G Sigma0 G^T
            pred = gs - gs @ sla.cho_solve(mat.kd_chol, gs)
            self.refs[m] = (pred, mat.mean)
        return self.refs[m]

    @staticmethod
    def data(m):
        """Data and (heterogeneous) noise variances for m observations."""
        rng = np.random.default_rng(1000 + m)
        return rng.standard_normal(m), 1e-4 * rng.uniform(1.0, 5.0, m)


def run_posterior(task: Task, ctx: PosteriorContext, rec: dict):
    from ngsrng.posterior import LinearObservations, MatheronPosterior

    m = int(task.options["m"])
    prior = ctx.prior(task.engine, task.device, task.dtype)
    sync = sync_for(task.device)
    rec["prior_per_sample_s"] = prior._bench_per_sample
    y, var = ctx.data(m)
    obs_kw = dict(averages=(ctx.centers(m), 0.05, "gaussian"), noise=var)
    t = time.perf_counter()
    if task.method == "matheron":
        obs = LinearObservations(prior, factorize=False, **obs_kw)
        post = MatheronPosterior(obs, data=y)
        rec["setup_s"] = time.perf_counter() - t
    else:
        rank = 32 if task.method.startswith("fe-space-rank32") else None
        form = "noise" if task.method == "noise-space" else "fe"
        low = "power" if task.method.endswith("-power") else "subspace"
        obs = LinearObservations(prior, rank=rank, low_rank_method=low, **obs_kw)
        rec["noise_space_rank"] = obs.rank
        if form == "fe":
            obs._get_lq()  # L Q is part of the FE-space setup
        rec["setup_s"] = time.perf_counter() - t
        t = time.perf_counter()
        post = obs.posterior(data=y, form=form)
        rec["posterior_s"] = time.perf_counter() - t  # forming the posterior (incl. mean) from the factorisation
    rec["per_sample_s"], _ = timeit(lambda: post.sample(1), sync, min_time=0.3, max_reps=5)
    batch = 4 if task.engine == "ngsolve" or task.method == "matheron" else 32
    tb, _ = timeit(lambda: post.sample(batch), sync, min_time=0.0, max_reps=1, min_reps=1)
    rec.update(
        batch=batch, per_sample_batch_s=tb / batch, overhead_vs_prior=rec["per_sample_s"] / rec["prior_per_sample_s"]
    )
    space = prior.space
    if m <= 100:
        pred_ref, mean_ref = ctx.reference(m)
        gf = obs.gf.toarray()
        if task.method == "matheron":
            pred = gf @ post.bmat
            pred = pred - pred @ sla.cho_solve(post.kd_chol, pred)
        else:
            yt = (
                torch.cat(
                    [
                        post.apply_t(torch.from_numpy(np.ascontiguousarray(gf.T[:, i : i + 64])))
                        for i in range(0, m, 64)
                    ],
                    dim=1,
                )
                .double()
                .cpu()
                .numpy()
            )
            pred = yt.T @ yt
        rec["pred_cov_rel_err"] = float(np.linalg.norm(pred - pred_ref) / np.linalg.norm(pred_ref))
        rec["mean_rel_err"] = float(np.abs(post.mean - mean_ref).max() / np.abs(mean_ref).max())
    if space.nfree <= EXACT_LIMIT and post.nrand <= 4 * EXACT_LIMIT:
        s0 = (
            exact_covariance(prior)
            if task.engine == "ngsolve"
            else exact_covariance(ctx.prior("ngsolve", "cpu", "float64"))
        )
        g = obs.gf.toarray()
        gs0 = g @ s0  # Woodbury form; the precision form is ill-conditioned for informative data
        sp_ = s0 - gs0.T @ np.linalg.solve(np.diag(var) + gs0 @ g.T, gs0)
        rec.update({f"law_{k}": v for k, v in covariance_errors(sampler_covariance(post), sp_).items()})


# ------------------------------------------------------------------ distributions suite

DIST_SAMPLES = 400


def _dist_args(args):
    return tuple(np.eye(int(a[3:])) if isinstance(a, str) and a.startswith("eye") else a for a in args)


def _dist_check(case, name, args, x, gauss, rec):
    """Statistical validity of a sample (rows: samples). KS p-values at 5 points where the marginal is known,
    chi^2 KS of a^T W a / a^T S a for Wishart fields, the variance of the domain mean vs theory for FE fields."""
    import scipy.stats as st

    marginals = {
        "uniform": st.uniform(), "lognormal": st.lognorm(0.5), "gamma": st.gamma(2.0), "beta": st.beta(2, 5),
        "chisquare": st.chi2(3), "standard_t": st.t(5),
    }
    probe = np.linspace(0, x.shape[1] - 1, 5, dtype=int)
    if name == "standard_t" and case.endswith("process") or name in ("normal_inverse_gaussian", "variance_gamma"):
        m = gauss.space.mass
        w = np.ones(gauss.space.ndof) @ m
        var = (np.asarray(x) @ w).var()
        ref = (w @ (gauss.sample(DIST_SAMPLES).double().cpu().numpy() - gauss.mean[:, None])).var()
        rec["variance_ratio"] = float(var / ref / (5 / 3 if name == "standard_t" else 1.0))
    elif name in marginals:
        rec["ks_min_pvalue"] = float(min(st.kstest(x[:, q], marginals[name].cdf).pvalue for q in probe))
    elif name == "poisson":
        rec["mean_rel_err"] = float(abs(x.mean() / 3 - 1))
    elif name == "wishart":
        df, p = args[0], args[1].shape[0]
        rec["ks_min_pvalue"] = float(min(st.kstest(x[:, q, 0, 0], st.chi2(df).cdf).pvalue for q in probe))
        rec["min_eig"] = float(np.linalg.eigvalsh(x[:, probe].reshape(-1, p, p)).min())
    elif name == "dirichlet":
        rec["ks_min_pvalue"] = float(min(st.kstest(x[:, q, 0], st.beta(1, 5).cdf).pvalue for q in probe))


def run_distributions(task: Task, fes, rec: dict):
    """Setup and per-sample cost of every family and algorithm relative to its Gaussian, plus a validity check."""
    from cases import DISTRIBUTION_CASES

    name, args, opts = DISTRIBUTION_CASES[task.method]
    args = _dist_args(args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gauss = ngsrng.Gaussian(
            fes,
            prior_spec(fes, task.kind, 2),
            device=task.device,
            dtype=task.dtype,
            seed=0,
            method={"engine": task.engine},
        )
        sync = sync_for(task.device)
        rec["gauss_per_sample_s"], _ = timeit(lambda: gauss.sample(8), sync, min_time=0.2, max_reps=3)
        rec["gauss_per_sample_s"] /= 8
        rng = ngsrng.default_rng(fes, seed=1, correlation=gauss, device=task.device, dtype=task.dtype)
        t = time.perf_counter()
        dist = rng.distribution(name, *args, **opts)
        rec["setup_s"] = time.perf_counter() - t
        dist.sample(1)  # first call: lazy set-up (inversion tables)
        rec["per_sample_s"], _ = timeit(lambda: dist.sample(8), sync, min_time=0.2, max_reps=3)
        rec["per_sample_s"] /= 8
        rec["overhead_vs_gauss"] = rec["per_sample_s"] / rec["gauss_per_sample_s"]
        info = dist.info
        for key in ("law", "inversion", "chi_inversion", "standardisation", "points", "device", "gaussian_fields",
                    "mixing", "algorithm"):
            if key in info:
                rec[key] = info[key]
        x = dist.sample(DIST_SAMPLES, output="numpy")
    _dist_check(task.method, name, args, x, gauss, rec)


# ------------------------------------------------------------------ low-rank suite


class LowRankReference:
    """Dense KL eigenvalues of the default prior on a space (only below LOWRANK_DENSE_LIMIT free dofs)."""

    def __init__(self, fes, kind):
        from cases import LOWRANK_DENSE_LIMIT

        self.fes, self.kind, self.lam = fes, kind, None
        s = SpaceInfo(fes)
        if s.nfree <= LOWRANK_DENSE_LIMIT:
            g = ngsrng.Gaussian(fes, prior_spec(fes, kind, 2), dtype="float64", method={"solver": "direct"}, seed=0)
            w = np.linalg.cholesky(s.mass_free.toarray())
            self.lam = np.sort(np.linalg.eigvalsh(w.T @ exact_covariance(g) @ w))[::-1]


def run_lowrank(task: Task, fes, rec: dict, ref: LowRankReference):
    """Time and accuracy of a truncated KL expansion: eigenvalue errors and the trace-norm error against the optimal
    rank-r truncation."""
    r, q = int(task.options["rank"]), int(task.options["q"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gauss = ngsrng.Gaussian(fes, prior_spec(fes, task.kind, 2), device=task.device, dtype=task.dtype, seed=0,
                                method={"engine": task.engine})
    r = min(r, gauss.space.nfree // 2)
    t = time.perf_counter()
    kl = ngsrng.karhunen_loeve(gauss, r, power_iterations=q, method=task.method, seed=1)
    rec.update(setup_s=time.perf_counter() - t, rank=kl.nrand, applications=(2 * q + 2) if task.method == "subspace"
               else (2 * q + 3))
    if ref.lam is not None:
        lam = ref.lam
        k = min(kl.nrand, r)
        rel = np.abs(kl.eigenvalues[:k] - lam[:k]) / lam[:k]
        rec.update(eig_rel_err_max=float(rel.max()), eig_rel_err_last=float(rel[-1]),
                   trace_captured_ratio=float(kl.eigenvalues[:k].sum() / lam[:k].sum()))


# ------------------------------------------------------------------ covariance suite


class CovarianceContext:
    """Per-group cache: the prior, its posterior under 40 point observations, quadrature-point evaluation, and the
    exact variances and traces (below COVARIANCE_EXACT_LIMIT free dofs)."""

    def __init__(self, fes, kind):
        from cases import COVARIANCE_EXACT_LIMIT

        from ngsrng.quadrature import QuadratureRule

        self.prior = ngsrng.Gaussian(fes, prior_spec(fes, kind, 2), dtype="float64", seed=0)
        lo, hi = 0.1, 0.9
        pts = np.random.default_rng(0).uniform(lo, hi, (40, fes.mesh.dim))
        self.sources = {"prior": self.prior, "posterior": self.prior.condition(points=pts, noise=1e-2)}
        self.eval = QuadratureRule(self.prior.space).evaluation
        self.exact = {}
        self.exact_ok = self.prior.space.nfree <= COVARIANCE_EXACT_LIMIT

    def reference(self, source):
        if source not in self.exact and self.exact_ok:
            t = time.perf_counter()
            lc = ngsrng.local_covariance(self.sources[source], "exact")
            self.exact[source] = (lc.values(self.eval), lc.trace()[0], time.perf_counter() - t)
        return self.exact.get(source)


def run_covariance(task: Task, ctx: CovarianceContext, rec: dict):
    """Pointwise variance (relative errors at the quadrature points) or trace (relative error) of a prior or
    posterior covariance, with the cost in applications of the factor L or L^T."""
    from cases import COVARIANCE_VARIANCE

    kind, name = task.method.split(":")
    source = task.solver.split("-p")[0]
    ref = ctx.reference(source)
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "variance":
            method, opts = COVARIANCE_VARIANCE[name]
            lc = ngsrng.local_covariance(ctx.sources[source], method, seed=1, **opts)
            v, se = lc.values(ctx.eval, stderr=True)
            rec.update(setup_s=time.perf_counter() - t, applications=lc.applications, method_used=lc.method)
            if ref is not None:
                rel = np.abs(v / ref[0] - 1)
                rec.update(rel_err_rms=float(np.sqrt(np.mean(rel**2))), rel_err_max=float(rel.max()),
                           stderr_rel_median=float(np.median(se / ref[0])))
        else:
            est = ngsrng.covariance_trace(ctx.sources[source], name, probes=int(task.options["probes"]), seed=1)
            rec.update(setup_s=time.perf_counter() - t, applications=est.applications, trace=est.value,
                       stderr_rel=est.stderr / abs(est.value))
            if ref is not None:
                rec.update(rel_err=abs(est.value / ref[1] - 1))


# ------------------------------------------------------------------ continuum suite

MODES = {
    1: [(0,), (1,), (3,), (6,)],
    2: [(0, 0), (1, 0), (2, 3), (5, 5)],
    3: [(0, 0, 0), (1, 0, 0), (1, 2, 2), (3, 3, 3)],
}
CELLS = {1: [4, 8, 16, 32, 64, 128, 256, 512], 2: [4, 8, 16, 32, 64, 128, 256], 3: [2, 4, 8, 16, 32, 64]}


def eigenfunction(k, dim):
    cf = ngs.CoefficientFunction(1.0)
    for ki, xi in zip(k, [ngs.x, ngs.y, ngs.z][:dim]):
        cf = cf * ((math.sqrt(2) if ki else 1.0) * ngs.cos(ki * math.pi * xi))
    return cf


def run_continuum(task: Task, out_fh, base: dict):
    """Var <f_h, e_k> = m^T Sigma_h m vs the exact lambda_k^-power (Neumann eigenfunctions, unit cube).

    Integer powers: m^T K^-1 (M K^-1)^(power-1) m with direct solves. Real powers: |L^T m|^2 through the exact adjoint
    of a sampler with a tight rational tolerance (1e-11), i.e. effectively exact.
    """
    from ngsrng.solvers import NGSolveDirect

    integer = float(task.power).is_integer()
    for n in CELLS[task.dim]:
        mesh = structured_mesh(task.dim, n)
        fes = ngs.H1(mesh, order=task.order) if task.kind == "H1" else ngs.L2(mesh, order=task.order, dgjumps=True)
        if fes.ndof > task.target:
            break
        space = SpaceInfo(fes)
        spec = prior_spec(fes, task.kind, task.power)
        t = time.perf_counter()
        if integer:
            solver = NGSolveDirect(space, spec.form, inverse="umfpack" if task.dim == 1 else "sparsecholesky")
            mass = ngs.BilinearForm(fes.TrialFunction() * fes.TestFunction() * ngs.dx).Assemble().mat
        else:
            spec.rtol = 1e-11
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gauss = ngsrng.Gaussian(fes, spec, dtype="float64")
        for k in MODES[task.dim]:
            m = ngs.LinearForm(eigenfunction(k, task.dim) * fes.TestFunction() * ngs.dx).Assemble().vec
            if integer:
                v, w = m.CreateVector(), m.CreateVector()
                with ngs.TaskManager():
                    solver.ngs_apply(m, v)
                    for _ in range(int(task.power) - 1):
                        w.data = mass * v
                        solver.ngs_apply(w, v)
                approx = ngs.InnerProduct(m, v)
            else:
                lt = gauss.apply_t(torch.from_numpy(m.FV().NumPy()[space.free].copy())).double().numpy()
                approx = float(lt @ lt)
            lam = KAPPA**2 + math.pi**2 * sum(ki**2 for ki in k)
            exact = lam ** (-task.power)
            rec = {
                **base,
                "cells": n,
                "h": 1 / n,
                "ndof": fes.ndof,
                "mode": str(k),
                "exact": exact,
                "discrete": approx,
                "rel_err": abs(approx - exact) / exact,
                "status": "ok",
                "time_s": time.perf_counter() - t,
            }
            out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="JSON file with the group and its tasks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem-limit-gb", type=float, default=0)
    args = ap.parse_args()
    if args.mem_limit_gb > 0:
        watchdog(args.mem_limit_gb * 2**30)
    with open(args.group) as fh:
        group = json.load(fh)
    tasks = [Task(**t) for t in group["tasks"]]
    suite = group["suite"]
    space = fes = ref = pctx = None
    if suite == "posterior":
        fes = space_for(tasks[0]).fes
        pctx = PosteriorContext(fes, tasks[0].kind)
    elif suite == "distributions":
        fes = space_for(tasks[0]).fes
    elif suite == "lowrank":
        fes = space_for(tasks[0]).fes
        ref = LowRankReference(fes, tasks[0].kind)
    elif suite == "covariance":
        fes = space_for(tasks[0]).fes
        ref = CovarianceContext(fes, tasks[0].kind)
    elif suite == "noise":
        space = space_for(tasks[0])
        _ = space.mass_free
    elif suite == "prior":
        fes = space_for(tasks[0]).fes
        ref = Reference(fes, tasks[0])
    with open(args.out, "a") as out:
        for task in tasks:
            print(f"START {task.id}", flush=True)
            rec = {**task.to_dict(), "task_id": task.id, "lineage": task.lineage, "status": "ok"}
            rec.pop("options")
            if space is not None or fes is not None:
                s = space if space is not None else SpaceInfo(fes)
                rec.update(ndof=s.ndof, nfree=s.nfree, ne=s.mesh.ne)
            try:
                if suite == "noise":
                    run_noise(task, space, rec)
                elif suite == "prior":
                    run_prior(task, fes, rec, ref)
                elif suite == "posterior":
                    run_posterior(task, pctx, rec)
                elif suite == "distributions":
                    run_distributions(task, fes, rec)
                elif suite == "lowrank":
                    run_lowrank(task, fes, rec, ref)
                elif suite == "covariance":
                    run_covariance(task, ref, rec)
                else:
                    run_continuum(task, out, rec)
                    print(f"DONE {task.id}", flush=True)
                    continue
            except Exception as err:  # noqa: BLE001 - every failure is a benchmark result
                msg = str(err)
                unavailable = any(k in msg for k in ("not available", "no inverse available", "No module named"))
                rec["status"] = "n/a" if unavailable else "error"
                rec["error"] = f"{type(err).__name__}: {err}"[:300]
                rec["traceback"] = traceback.format_exc()[-1500:]
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            out.write(json.dumps(rec, default=float) + "\n")
            out.flush()
            print(f"DONE {task.id}", flush=True)


if __name__ == "__main__":
    main()
