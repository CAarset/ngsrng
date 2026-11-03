"""GPU throughput check: large batches of Gaussian, lognormal and uniform random functions, filled in place.

    python bench/gpu_batch.py                  # CUDA, full grid (~20-40 min)
    python bench/gpu_batch.py --quick          # CUDA, small sizes (a few minutes)
    python bench/gpu_batch.py --device cpu --quick   # dry run of the script on the CPU engine
    python bench/gpu_batch.py --quick --work-values 2**22,2**24,2**26   # in-place chunk-size sweep

1. Checks on the device (they gate the timings): the normal CDF in both tails, Gaussian variances against the exact
   pointwise variance (CPU reference), uniform marginals (KS), and ``out=`` against returned arrays.
2. Throughput of ``rng.<family>(out=buffer)`` with a preallocated buffer on the device, repeated (in-place
   replacement), for 2D P2 and 3D P1 spaces of several sizes: torch engine with cuDSS at right-hand-side widths 32
   and 128 (float64 arithmetic, float32 output: the defaults); the CPU engine (threaded NGSolve) as reference. The
   solver choice itself is calibrated by python -m ngsrng.calibrate, which also rejects batched PCG and float32
   arithmetic. With
   ``--work-values`` every configuration is repeated for each chunk size of the in-place fill
   (:mod:`ngsrng.inplace`); the peak device memory of every measurement is recorded.

Results go to results/runs/<host>_<date>_gpubatch/ (results.jsonl, report.md).
"""

import argparse
import gc
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
import warnings
from datetime import datetime

import ngsolve as ngs
import numpy as np
import scipy.special as sps
import scipy.stats as st
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ngsrng  # noqa: E402
from ngsrng.distributions.pointwise import XP  # noqa: E402
from ngsrng.solvers import cudss_available  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUFFER_BYTES = 6e9  # largest out buffer per measurement (the batch is reduced to fit)


def space(kind, maxh):
    if kind == "2D-P2":
        return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=maxh)), order=2)
    return ngs.H1(ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=maxh)), order=1)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _ints(text):
    """'2**22,2**24,16777216' -> [4194304, 16777216, 16777216]."""
    vals = []
    for v in text.split(","):
        base, _, exp = v.strip().partition("**")
        vals.append(int(float(base)) ** int(exp) if exp else int(float(base)))
    return vals


# ------------------------------------------------------------------ checks


def checks(device, log):
    out = {}
    # normal CDF tails on the device (torch.special.ndtr cancels in the lower tail; ngsrng uses the erfc form)
    z = np.linspace(-37, 8, 200001)
    ref = sps.ndtr(z)
    for dt in (torch.float64, torch.float32):
        val = XP(True).ndtr(torch.from_numpy(z).to(device, dt)).double().cpu().numpy()
        ok = ref > (1e-300 if dt == torch.float64 else 1e-37)
        out[f"ndtr_max_rel_{str(dt)[6:]}"] = float(np.max(np.abs(val[ok] - ref[ok]) / ref[ok]))
    # Gaussian variances and uniform marginals against the exact CPU reference
    fes = space("2D-P2", 0.08)
    corr = ngsrng.matern_correlation(0.25, nu=1.0)
    pts = np.random.default_rng(0).uniform(0.05, 0.95, (40, 2))
    exact = np.diag(ngsrng.functional_covariance(ngsrng.default_rng(fes, correlation=corr, dtype="float64").gaussian(),
                                                 points=pts))
    rng = ngsrng.default_rng(fes, seed=1, correlation=corr, device=device, dtype="float64")
    n = 20000
    vals = rng.normal(size=n, points=pts, output="numpy")
    z = (vals.var(axis=0) / exact - 1) / np.sqrt(2.0 / n)
    out["gauss_var_max_abs_z"] = float(np.abs(z).max())
    u = rng.uniform(size=4000, output="numpy")
    out["uniform_ks_min_p"] = float(min(st.kstest(u[:, j], "uniform").pvalue for j in (0, 1000, 5000)))
    # out= equals the returned arrays (same seeds; two separately built generators: their factorisations, on the
    # GPU as on the CPU, agree to rounding, not bit for bit). Rounding differences of the solves grow with the
    # condition number (~1e4 here), so the gate is relative to the dtype's precision: it is there to catch wrong
    # means, seeds or transforms (O(1) differences), not rounding (measured 3.7e-4 in float32 at 6cb2b82)
    dt = torch.float32  # the generators' default dtype
    diffs = []
    for name, args in (("normal", ()), ("lognormal", (0.0, 0.5)), ("uniform", ())):
        a = getattr(ngsrng.default_rng(fes, seed=2, correlation=corr, device=device, dtype=dt), name)(
            *args, size=5, output="torch")
        buf = torch.empty_like(a)
        getattr(ngsrng.default_rng(fes, seed=2, correlation=corr, device=device, dtype=dt), name)(*args, out=buf)
        diffs.append(float(((buf - a).abs() / a.abs().clamp_min(1e-3)).max()))
    out["out_vs_returned_max_rel"] = max(diffs)
    out["out_vs_returned_tol"] = 1e5 * torch.finfo(dt).eps
    out["passed"] = (out["ndtr_max_rel_float64"] < 1e-11 and out["gauss_var_max_abs_z"] < 5
                     and out["uniform_ks_min_p"] > 1e-4 and out["out_vs_returned_max_rel"] < out["out_vs_returned_tol"])
    log(f"checks: {out}")
    return out


def fractional_checks(device, log, n=100_000, big=1_000_000):
    """Fractional powers on the device (C = A^-1.6 on 1D meshes, the hardest case: cond ~ h^-2): accuracy against the
    CPU engine at 1e5 dofs, the 1e6-dof case that ran out of 40 GB before the shared cuDSS buffers, and float32
    arithmetic, whose spectral-interval estimate failed before it was moved to float64."""
    from ngsolve.meshes import Make1DMesh

    out = {}
    spec = ngsrng.matern(5.0, 1.6)
    fes = ngs.H1(Make1DMesh(n), order=1)
    t0 = time.perf_counter()
    ref = ngsrng.Gaussian(fes, spec, dtype="float64")
    out["frac_cpu_setup_s"] = time.perf_counter() - t0
    eta = torch.randn(ref.nrand, 4, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    c_ref = ref.apply(eta).numpy()
    for label, method in (("f64", None), ("f32work", {"work_dtype": "float32"})):
        try:
            t0 = time.perf_counter()
            g = ngsrng.Gaussian(fes, spec, device=device, dtype="float64", method=method)
            out[f"frac_{label}_setup_s"] = time.perf_counter() - t0
            c = g.apply(eta.to(device)).double().cpu().numpy()
            out[f"frac_{label}_pathwise_rel_err"] = float(np.linalg.norm(c - c_ref) / np.linalg.norm(c_ref))
        except Exception as err:  # noqa: BLE001 - a failure is the result
            out[f"frac_{label}_error"] = f"{type(err).__name__}: {err}"[:200]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.perf_counter()
        g = ngsrng.Gaussian(ngs.H1(Make1DMesh(big), order=1), spec, device=device, dtype="float64")
        out["frac_1e6_setup_s"] = time.perf_counter() - t0
        out["frac_1e6_plan"] = {k: g.plan.get(k) for k in ("structure", "shifted_width")}
        sync(device)
        t0 = time.perf_counter()
        g.sample(32)
        sync(device)
        out["frac_1e6_per_sample_s"] = (time.perf_counter() - t0) / 32
        if device.type == "cuda":
            out["frac_1e6_gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    except Exception as err:  # noqa: BLE001
        out["frac_1e6_error"] = f"{type(err).__name__}: {err}"[:200]
    out["frac_passed"] = (out.get("frac_f64_pathwise_rel_err", 1.0) < 1e-5 and "frac_f32work_setup_s" in out
                          and "frac_1e6_per_sample_s" in out)
    log(f"fractional checks: {out}")
    return out


# ------------------------------------------------------------------ timings


def time_batch(rng, family, fes, batch, repeats, device, dtype, points=None):
    kwargs = {"points": points} if points is not None else {}
    args = (0.0, 0.5) if family == "lognormal" else ()
    d = rng.distribution(family, *args, **kwargs)
    t0 = time.perf_counter()
    first = d.sample(2, output="torch")  # set-up (std, tables) and shape
    setup = time.perf_counter() - t0
    per = first.shape[1:]
    batch = max(1, min(batch, int(BUFFER_BYTES // (math.prod(per) * torch.empty(0, dtype=dtype).element_size()))))
    buf = torch.empty((batch, *per), device=device, dtype=dtype)
    times = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        d.sample(out=buf)
        sync(device)
        times.append(time.perf_counter() - t0)
    return {"setup_s": setup, "batch": batch, "values_per_sample": int(math.prod(per)),
            "per_sample_s": min(times) / batch, "batch_s": min(times),
            "samples_per_s": batch / min(times)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--work-values", type=_ints, default=None,
                    help="chunk sizes of the in-place fill to sweep, e.g. 2**22,2**24,2**26 (default: the library's)")
    ap.add_argument("--out-dir", default=None, help="run directory (default: results/runs/<host>_<date>_gpubatch)")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("no CUDA device available (use --device cpu for a dry run)")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = args.out_dir or os.path.join(ROOT, "results", "runs", f"{socket.gethostname()}_{stamp}_gpubatch")
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "results.jsonl")

    def log(msg):
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

    def record(rec):
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")

    env = {"host": socket.gethostname(), "python": platform.python_version(), "torch": torch.__version__,
           "ngsolve": ngs.__version__, "cpu_threads": ngsrng.backend.available_threads(), "device": str(device),
           "cudss": cudss_available()}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        env.update(gpu=torch.cuda.get_device_name(), gpu_free_gb=free / 1e9, gpu_total_gb=total / 1e9)
    log(f"environment: {env}")
    with open(os.path.join(run_dir, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=1)

    warnings.simplefilter("ignore")
    check = checks(device, log)
    if device.type == "cuda":  # the CPU engine's fractional powers are covered by the tests
        check.update(fractional_checks(device, log))
    record({"kind": "checks", **check})

    # quick: the sizes of the 2026-11-02 throughput numbers (32k and 129k dofs 2D P2, 7k 3D P1)
    sizes = {"2D-P2": [0.012, 0.006] if args.quick else [0.03, 0.012, 0.006],
             "3D-P1": [0.05] if args.quick else [0.1, 0.05]}
    families = [("normal", None), ("lognormal", None), ("uniform", None)]
    if device.type == "cuda":
        # the defaults (float64 arithmetic, float32 output) at two widths, and the float32-arithmetic opt-in: what the
        # accuracy of the default costs
        solvers = ([{"solver": "cudss", "cudss_width": w} for w in (32, 128)]
                   + [{"solver": "cudss", "cudss_width": 32, "work_dtype": "float32"}] if cudss_available()
                   else [{"solver": "torch-pcg"}])
        dtypes = [torch.float32]
    else:
        solvers, dtypes = [{"engine": "ngsolve"}], [torch.float32]
    batch = 500 if args.quick else args.batch
    for kind, hs in sizes.items():
        for h in hs:
            fes = space(kind, h)
            pts = np.random.default_rng(0).uniform(0.05, 0.95, (1000, fes.mesh.dim))
            log(f"{kind}, maxh {h}: {fes.ndof} dofs")
            configs = [(m, dt) for m in solvers for dt in dtypes if not (dt == torch.float64 and m.get("cudss_width",
                                                                                                       32) != 32)]
            configs.append(({"engine": "ngsolve"}, torch.float32))  # CPU reference
            for method, dt in configs:
                cpu = method.get("engine") == "ngsolve"
                dev = torch.device("cpu") if cpu else device
                label = "cpu-ngsolve" if cpu else "+".join(f"{k}={v}" for k, v in method.items())
                if cpu and fes.ndof > (5e4 if args.quick else 2e5):
                    continue  # the reference is too slow at this size
                for family, where in families:
                    for wv in args.work_values or [None]:
                        rec = {"kind": "timing", "space": kind, "maxh": h, "ndof": fes.ndof, "config": label,
                               "dtype": str(dt)[6:], "family": family, "values": where or "quadrature/coefficients",
                               "work_values": wv or "default"}
                        try:
                            rng = ngsrng.default_rng(fes, seed=0, correlation=0.2, device=dev, dtype=dt,
                                                     method=None if cpu else {"engine": "torch", **method},
                                                     work_values=wv)
                            b = min(batch, 500) if cpu else batch
                            if dev.type == "cuda":
                                torch.cuda.reset_peak_memory_stats()
                            rec.update(time_batch(rng, family, fes, b, args.repeats, dev, dt,
                                                  points=pts if where == "points" else None))
                            if dev.type == "cuda":
                                rec["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
                                rec["buffer_gb"] = rec["batch"] * rec["values_per_sample"] * \
                                    torch.empty(0, dtype=dt).element_size() / 1e9
                            rec["status"] = "ok"
                            log(f"  {label:32s} {rec['dtype']:7s} {family:9s} {where or '':6s} wv={wv or 'default'}: "
                                f"{1e3 * rec['per_sample_s']:8.3f} ms/sample (batch {rec['batch']})")
                        except Exception as err:  # noqa: BLE001 - every failure is a result
                            rec.update(status="error", error=f"{type(err).__name__}: {err}"[:300],
                                       traceback=traceback.format_exc()[-1000:])
                            log(f"  {label} {family}: {rec['error']}")
                        finally:
                            rng = None
                            gc.collect()
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                        record(rec)
    report(run_dir)


def report(run_dir):
    recs = [json.loads(line) for line in open(os.path.join(run_dir, "results.jsonl"))]
    lines = ["# GPU batch benchmark\n", f"- run: `{os.path.basename(run_dir)}`\n"]
    chk = [r for r in recs if r["kind"] == "checks"]
    if chk:
        lines.append("## Checks\n\n" + "\n".join(f"- {k}: {v}" for k, v in chk[0].items() if k != "kind") + "\n")
    try:
        import pandas as pd
        import tabulate  # noqa: F401 - used by DataFrame.to_markdown
    except ImportError:
        lines.append("Timing tables need pandas and tabulate (`pip install -e .[bench]`); the records are in "
                     "results.jsonl.\n")
        pd = None
    df = pd.DataFrame([r for r in recs if r["kind"] == "timing" and r.get("status") == "ok"]) if pd else []
    if len(df):
        df["ms/sample"] = 1e3 * df["per_sample_s"]
        df["work_values"] = df["work_values"].astype(str)
        piv = df.pivot_table(index=["space", "ndof", "family", "values"], columns=["config", "dtype", "work_values"],
                             values="ms/sample")
        lines.append("## Milliseconds per sample (batch filled in place)\n\n" + piv.to_markdown(floatfmt=".3f") + "\n")
        if "gpu_peak_gb" in df:
            mem = df.pivot_table(index=["space", "ndof", "family", "values"],
                                 columns=["config", "dtype", "work_values"], values="gpu_peak_gb")
            lines.append("## Peak device memory, GB (including the out buffer)\n\n" + mem.to_markdown(floatfmt=".2f")
                         + "\n")
    with open(os.path.join(run_dir, "report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"report: {os.path.join(run_dir, 'report.md')}")


if __name__ == "__main__":
    main()
