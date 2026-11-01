"""Sparse linear algebra backends for ngsrng's per-sample kernels: products and SPD solves, on the CPU and the GPU.

    python bench/sparse_backends.py --profile smoke --devices cpu   # a few minutes on any machine (harness check)
    python bench/sparse_backends.py                                 # "standard": CPU + CUDA, up to ~1e6 dofs
    python bench/sparse_backends.py --profile full                  # up to ~1e7 dofs (GPU box)
    python bench/sparse_backends.py --report results/runs/<run>     # rebuild report.md from results.jsonl

1. Products Y = A X with k = 1, 32, 256 columns, for the matrices a sample applies: the Matern operator K (and the
   mass matrix M), the explicit local-Gram noise factor H and its transpose (adjoint, posteriors), the quadrature
   noise factor B, point observations G and G^T. Backends: SciPy CSR, NGSolve SparseMatrixd (MultiVector, TaskManager
   threads), torch CSR and COO (CPU, CUDA), cupyx CSR (CUDA), torch-sla SparseTensor.
2. SPD solves K X = R with k = 1 and 32 right-hand sides: NGSolve sparsecholesky / umfpack / pardiso (threaded
   right-hand sides, as in ngsrng), SciPy SuperLU, ngsrng's cuDSS (nvmath) and torch PCG (Jacobi), and torch-sla:
   SciPy LU with a reused factorisation, pytorch CG (Jacobi), cuDSS Cholesky, PyAMG and AmgX. Set-up
   (conversion, factorisation), the first solve and repeated solves are timed separately (does the backend keep its
   factorisation?), and every solution's residual is checked in float64.

Matrices come from ngsrng itself (structured meshes, bench/common.py) and are stored once per run as .npz. Each
(case, backend, device, dtype) group runs in its own process with a timeout: crashes, hangs and out-of-memory
errors are recorded and cost only that group. Optional backends that are not installed are recorded as "n/a"
(torch-sla: ``pip install torch-sla``, without extras so that it uses the installed nvmath-python[cu13];
cupy: ``pip install cupy-cuda13x``). Results go to results/runs/<host>_<date>_sparse/ (results.jsonl,
environment.json, report.md).
"""

import argparse
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (dim, kind, order, target ndof)
PROFILES = {
    "smoke": dict(cases=[(2, "H1", 2, 3e3), (2, "L2", 1, 2e3)], ks=[1, 32], solve_ks=[1, 32], repeats=3,
                  timeout=600),
    "quick": dict(cases=[(2, "H1", 1, 1e5), (2, "H1", 2, 1e5), (2, "H1", 4, 1e5), (3, "H1", 1, 1e5),
                         (3, "H1", 2, 1e5), (2, "L2", 1, 1e5)],
                  ks=[1, 32, 256], solve_ks=[1, 32], repeats=5, timeout=1800),
    "standard": dict(cases=[(d, "H1", p, n) for n in (1e4, 1e5, 1e6) for d, p in ((2, 1), (2, 2), (2, 4), (3, 1),
                                                                                  (3, 2))]
                     + [(2, "L2", 1, 1e5), (2, "L2", 1, 1e6), (3, "L2", 1, 1e5)],
                     ks=[1, 32, 256], solve_ks=[1, 32], repeats=5, timeout=3600),
    "full": dict(cases=[(d, "H1", p, n) for n in (1e4, 1e5, 1e6, 1e7) for d, p in ((2, 1), (2, 2), (2, 4), (3, 1),
                                                                                  (3, 2))]
                 + [(2, "L2", 1, 1e6), (3, "L2", 1, 1e6), (2, "L2", 1, 1e7)],
                 ks=[1, 32, 256], solve_ks=[1, 32], repeats=5, timeout=7200),
}
PRODUCT_BACKENDS = ["scipy", "ngsolve", "torch-csr", "torch-coo", "cupy-csr", "torch-sla"]
SOLVE_BACKENDS = ["ngsolve-sparsecholesky", "ngsolve-umfpack", "ngsolve-pardiso", "scipy-splu", "ngsrng-cudss",
                  "ngsrng-torch-pcg", "torch-sla-scipy-lu", "torch-sla-pytorch-cg", "torch-sla-cudss-cholesky",
                  "torch-sla-pyamg", "torch-sla-amgx"]
CPU_ONLY = {"scipy", "ngsolve", "ngsolve-sparsecholesky", "ngsolve-umfpack", "ngsolve-pardiso", "scipy-splu",
            "torch-sla-scipy-lu"}
CUDA_ONLY = {"cupy-csr", "ngsrng-cudss", "torch-sla-cudss-cholesky", "torch-sla-amgx"}
FLOAT64_ONLY = {"ngsolve", "ngsolve-sparsecholesky", "ngsolve-umfpack", "ngsolve-pardiso", "scipy-splu",
                "torch-sla-scipy-lu"}
DENSE_SOLVE_LIMIT = 3e5  # SciPy SuperLU only up to this many dofs (slow for many right-hand sides, known issue 14)
KAPPA = 10.0


# ------------------------------------------------------------------ matrices (parent process, NGSolve only)


def build_case(case, out_dir):
    """Assemble the case's matrices with ngsrng and store them as .npz; returns metadata."""
    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    import ngsolve as ngs
    from common import make_space

    import ngsrng
    from ngsrng import noise as nz
    from ngsrng.space import SpaceInfo

    dim, kind, order, target = case
    name = f"{dim}D-{kind}-p{order}-{int(target)}"
    path = os.path.join(out_dir, f"{name}.npz")
    if os.path.exists(path):
        return json.load(open(path[:-4] + ".json"))
    t0 = time.perf_counter()
    fes = make_space(dim, kind, order, target, **({"dgjumps": True} if kind == "L2" else {}))
    s = SpaceInfo(fes)
    spec = ngsrng.matern(KAPPA) if kind == "H1" else ngsrng.matern_dg(fes, KAPPA)
    mats = {"K": s.assemble_free(spec.form), "M": s.mass_free}
    if not s.mass_is_diagonal:
        mats["H"] = nz.LocalGramNoise(s).explicit().h
        mats["Ht"] = mats["H"].T.tocsr()
    try:
        mats["B"] = nz.quadrature_noise(s).h
    except (MemoryError, RuntimeError, NotImplementedError) as err:
        print(f"  {name}: no quadrature factor ({err})", flush=True)
    m = int(min(1000, max(10, s.nfree // 10)))
    pts = np.random.default_rng(0).uniform(0.02, 0.98, (m, dim))
    with ngs.TaskManager():
        g = ngsrng.observation_operator(fes, points=pts)[:, s.free].tocsr()
    mats["G"], mats["Gt"] = g, g.T.tocsr()
    np.savez(path, **{f"{k}_{part}": v for k, a in mats.items() for part, v in _parts(a).items()})
    meta = {"case": name, "dim": dim, "kind": kind, "order": order, "ndof": fes.ndof, "nfree": s.nfree,
            "build_s": time.perf_counter() - t0,
            "matrices": {k: {"shape": list(a.shape), "nnz": int(a.nnz)} for k, a in mats.items()}}
    json.dump(meta, open(path[:-4] + ".json", "w"), indent=1)
    return meta


def _parts(a):
    a = a.tocsr()
    a.sort_indices()
    return {"data": a.data, "indices": a.indices, "indptr": a.indptr, "shape": np.array(a.shape)}


def load_matrices(path):
    z = np.load(path)
    names = sorted({k.rsplit("_", 1)[0] for k in z.files})
    return {n: sp.csr_matrix((z[f"{n}_data"], z[f"{n}_indices"], z[f"{n}_indptr"]), shape=tuple(z[f"{n}_shape"]))
            for n in names}


# ------------------------------------------------------------------ backends (worker process)


class Unavailable(Exception):
    pass


def _torch_dtype(dtype):
    import torch

    return {"float64": torch.float64, "float32": torch.float32}[dtype]


def _sync(device):
    if device == "cuda":
        import torch

        torch.cuda.synchronize()
        try:
            import cupy

            cupy.cuda.Device().synchronize()
        except ImportError:
            pass


def _torch_sla():
    try:
        import torch_sla
    except ImportError as err:
        raise Unavailable("torch-sla not installed") from err
    return torch_sla


def _sla_tensor(a, device, dtype):
    import torch

    ts = _torch_sla()
    c = a.tocoo()
    t = ts.SparseTensor(torch.from_numpy(c.data).to(_torch_dtype(dtype)), torch.from_numpy(c.row.astype(np.int64)),
                        torch.from_numpy(c.col.astype(np.int64)), c.shape)
    return t.cuda() if device == "cuda" else t


def product_backend(name, a, device, dtype):
    """(apply(X) -> Y, to_input(numpy) -> backend array, to_numpy(Y) -> numpy)."""
    import torch

    tdt = _torch_dtype(dtype)
    if name == "scipy":
        m = a.astype(dtype)
        return lambda x: m @ x, lambda x: x.astype(dtype), lambda y: y
    if name == "ngsolve":
        import ngsolve as ngs

        from ngsrng.backend import ngs_sparse

        m = ngs_sparse(a)
        cache = {}

        def to_input(x):
            k = x.shape[1]
            mi = ngs.MultiVector(m.CreateRowVector(), k)
            for j in range(k):
                mi[j].FV().NumPy()[:] = x[:, j]
            cache["out"] = ngs.MultiVector(m.CreateColVector(), k)
            return mi

        def apply(mi):
            with ngs.TaskManager():
                cache["out"][:] = m * mi
            return cache["out"]

        return apply, to_input, lambda mo: np.column_stack([mo[j].FV().NumPy() for j in range(len(mo))])
    if name in ("torch-csr", "torch-coo"):
        from ngsrng.backend import sparse_to_torch

        m = sparse_to_torch(a, device, tdt)
        if name == "torch-coo":
            m = m.to_sparse_coo().coalesce()
        return lambda x: m @ x, lambda x: torch.from_numpy(x).to(device, tdt), lambda y: y.double().cpu().numpy()
    if name == "cupy-csr":
        try:
            import cupy
            import cupyx.scipy.sparse as csp
        except ImportError as err:
            raise Unavailable("cupy not installed") from err
        m = csp.csr_matrix(a.astype(dtype))
        return lambda x: m @ x, lambda x: cupy.asarray(x.astype(dtype)), lambda y: cupy.asnumpy(y).astype(float)
    if name == "torch-sla":
        m = _sla_tensor(a, device, dtype)
        return lambda x: m @ x, lambda x: torch.from_numpy(x).to(device, tdt), lambda y: y.double().cpu().numpy()
    raise ValueError(name)


class _NGSolveInverse:
    """NGSolve sparse direct solver with right-hand sides in parallel threads (ngsrng's CPU engine pattern)."""

    def __init__(self, a, inverse):
        import ngsolve as ngs

        from ngsrng.backend import Scratch, ngs_sparse

        self.mat = ngs_sparse(a)
        free = ngs.BitArray(a.shape[0])
        free.Set()
        with ngs.TaskManager():
            self.inv = self.mat.Inverse(free, inverse)
        self.vecs = Scratch(lambda: (self.mat.CreateColVector(), self.mat.CreateColVector()))

    def solve(self, b):
        import ngsolve as ngs

        from ngsrng.backend import blas_threads, parallel_ranges

        out = np.empty((b.shape[1], b.shape[0]))

        def work(lo, hi):
            rhs, sol = self.vecs.get()
            for j in range(lo, hi):
                rhs.FV().NumPy()[:] = b[:, j]
                sol.data = self.inv * rhs
                out[j] = sol.FV().NumPy()

        if b.shape[1] == 1:
            with ngs.TaskManager():
                work(0, 1)
        else:
            with blas_threads(1):
                parallel_ranges(b.shape[1], work)
        return out.T


def solve_backend(name, a, device, dtype, k, rtol):
    """(setup() -> solver, solve(solver, R) -> X, to_input, to_numpy). setup is timed separately."""
    import torch

    tdt = _torch_dtype(dtype)
    to_t = lambda x: torch.from_numpy(x).to(device, tdt)  # noqa: E731
    from_t = lambda y: y.double().cpu().numpy()  # noqa: E731
    ident = lambda x: x  # noqa: E731
    if name.startswith("ngsolve-"):
        inverse = name.split("-", 1)[1]
        if inverse == "pardiso":
            try:
                import mkl  # noqa: F401
            except ImportError as err:
                raise Unavailable("PARDISO needs the mkl package") from err
        return lambda: _NGSolveInverse(a, inverse), lambda s, r: s.solve(r), ident, ident
    if name == "scipy-splu":
        import scipy.sparse.linalg as spla

        if a.shape[0] > DENSE_SOLVE_LIMIT:
            raise Unavailable(f"SuperLU skipped above {DENSE_SOLVE_LIMIT:.0e} dofs")
        return lambda: spla.splu(a.tocsc()), lambda s, r: s.solve(r), ident, ident
    if name == "ngsrng-cudss":
        from ngsrng.solvers import CudssDirect, cudss_available

        if not cudss_available():
            raise Unavailable("nvmath/cuDSS not available")
        return lambda: CudssDirect(a, torch.device(device), tdt, width=k), lambda s, r: s.apply(r), to_t, from_t
    if name == "ngsrng-torch-pcg":
        from ngsrng.solvers import TorchPCG

        return lambda: TorchPCG(a, torch.device(device), tdt, tol=rtol), lambda s, r: s.apply(r), to_t, from_t
    if name.startswith("torch-sla-"):
        ts = _torch_sla()
        if name == "torch-sla-scipy-lu":
            return lambda: _sla_tensor(a, device, dtype).lu(), lambda s, r: s.solve(r), to_t, from_t
        backend, method = {"torch-sla-pytorch-cg": ("pytorch", "cg"), "torch-sla-cudss-cholesky": ("cudss", "cholesky"),
                           "torch-sla-pyamg": ("pyamg", "sa"), "torch-sla-amgx": ("amgx", "pcg")}[name]
        if backend not in ts.get_available_backends():
            raise Unavailable(f"torch-sla backend {backend} not available")
        kw = {} if method in ("cholesky",) else {"rtol": rtol, "atol": 0.0}
        if backend == "pytorch":
            kw["preconditioner"] = "jacobi"

        def solve(s, r):
            # through SolverConfig: torch-sla 0.4.0's solve(b, backend=, method=) path fails for n x k right-hand
            # sides with the Jacobi preconditioner (broadcasting), the SolverConfig path does not
            with ts.SolverConfig(backend=backend, method=method, **kw):
                return s.solve(r)

        return lambda: _sla_tensor(a, device, dtype), solve, to_t, from_t
    raise ValueError(name)


def _measure(fn, sync, repeats):
    times = []
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        times.append(time.perf_counter() - t)
    return float(np.median(times)), float(np.min(times)), out


def _peak_gpu(device):
    if device != "cuda":
        return None
    import torch

    return torch.cuda.max_memory_allocated() / 2**30


def run_group(args):
    """Worker: one (case, kind, backend, device, dtype) group; prints one 'REC {json}' line per measurement."""
    import torch

    warnings.simplefilter("ignore")
    sys.path.insert(0, ROOT)
    mats = load_matrices(args.matrices)
    base = {"case": os.path.basename(args.matrices)[:-4], "suite": args.suite, "backend": args.backend,
            "device": args.device, "dtype": args.dtype}
    sync = lambda: _sync(args.device)  # noqa: E731
    rng = np.random.default_rng(0)

    def emit(rec):
        rec["host_peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        print("REC " + json.dumps({**base, **rec}, default=float), flush=True)

    rtol = 1e-8 if args.dtype == "float64" else 1e-5
    if args.suite == "product":
        for mname in ("K", "M", "H", "Ht", "B", "G", "Gt"):
            if mname not in mats:
                continue
            a = mats[mname]
            for k in args.ks:
                rec = {"matrix": mname, "k": k, "rows": a.shape[0], "cols": a.shape[1], "nnz": int(a.nnz)}
                try:
                    if args.device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    x = rng.standard_normal((a.shape[1], k))
                    t = time.perf_counter()
                    apply, to_input, to_numpy = product_backend(args.backend, a, args.device, args.dtype)
                    xi = to_input(x)
                    sync()
                    rec["setup_s"] = time.perf_counter() - t
                    apply(xi)  # warm-up (kernel selection, workspaces)
                    med, best, y = _measure(lambda: apply(xi), sync, args.repeats)
                    ref = a @ x
                    rec.update(status="ok", time_s=med, best_s=best, per_column_s=med / k,
                               gbytes_per_s=(12 * a.nnz + 8 * k * (a.shape[0] + a.shape[1])) / best / 1e9,
                               rel_err=float(np.abs(to_numpy(y) - ref).max() / max(np.abs(ref).max(), 1e-300)),
                               gpu_peak_gb=_peak_gpu(args.device))
                except Unavailable as err:
                    rec.update(status="n/a", error=str(err))
                    emit(rec)
                    return
                except Exception as err:  # noqa: BLE001
                    rec.update(status="error", error=f"{type(err).__name__}: {err}"[:300],
                               trace=traceback.format_exc()[-800:])
                emit(rec)
    else:
        a = mats["K"]
        for k in args.solve_ks:
            rec = {"matrix": "K", "k": k, "rows": a.shape[0], "nnz": int(a.nnz), "rtol": rtol}
            try:
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                r = rng.standard_normal((a.shape[0], k))
                setup, solve, to_input, to_numpy = solve_backend(args.backend, a, args.device, args.dtype, k, rtol)
                ri = to_input(r)
                sync()
                t = time.perf_counter()
                solver = setup()
                sync()
                rec["setup_s"] = time.perf_counter() - t
                t = time.perf_counter()
                x = solve(solver, ri)
                sync()
                rec["first_solve_s"] = time.perf_counter() - t
                med, best, x = _measure(lambda: solve(solver, ri), sync, args.repeats)
                xn = to_numpy(x).reshape(a.shape[0], k)
                res = np.linalg.norm(a @ xn - r, axis=0) / np.linalg.norm(r, axis=0)
                rec.update(status="ok", solve_s=med, best_s=best, per_rhs_s=med / k,
                           repeat_vs_first=med / max(rec["first_solve_s"], 1e-12),
                           residual=float(res.max()), gpu_peak_gb=_peak_gpu(args.device))
                its = getattr(solver, "iterations", None)
                if its:
                    rec["iterations"] = float(np.mean(its[-args.repeats:]))
            except Unavailable as err:
                rec.update(status="n/a", error=str(err))
                emit(rec)
                return
            except Exception as err:  # noqa: BLE001
                rec.update(status="error", error=f"{type(err).__name__}: {err}"[:300],
                           trace=traceback.format_exc()[-800:])
            emit(rec)


# ------------------------------------------------------------------ driver


def environment(devices):
    env = {"host": socket.gethostname(), "time": datetime.now().isoformat(timespec="seconds"),
           "platform": platform.platform(), "python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    try:
        env["git"] = subprocess.run(["git", "-C", ROOT, "describe", "--always", "--dirty"], capture_output=True,
                                    text=True).stdout.strip()
    except OSError:
        pass
    versions = {}
    for mod in ("numpy", "scipy", "torch", "ngsolve", "torch_sla", "cupy", "nvmath", "pyamg"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = None
    env["versions"] = versions
    if "cuda" in devices:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["cuda"] = {"device": p.name, "mem_total_gb": p.total_memory / 2**30, "torch_cuda": torch.version.cuda}
    return env


def groups(meta, cfg, devices, backends):
    for suite, names in (("product", PRODUCT_BACKENDS), ("solve", SOLVE_BACKENDS)):
        for b in names:
            if backends and b not in backends:
                continue
            for dev in devices:
                if (dev == "cpu" and b in CUDA_ONLY) or (dev == "cuda" and b in CPU_ONLY):
                    continue
                for dt in ("float64",) if b in FLOAT64_ONLY else ("float64", "float32"):
                    yield suite, b, dev, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default="standard", choices=list(PROFILES))
    ap.add_argument("--devices", default="cpu,cuda")
    ap.add_argument("--backends", default="", help="comma-separated subset (default: all)")
    ap.add_argument("--out", default=None, help="run directory (default: results/runs/<host>_<date>_sparse)")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None, help="seconds per group (default: by profile)")
    ap.add_argument("--report", default=None, help="only rebuild report.md of this run directory")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--matrices", help=argparse.SUPPRESS)
    ap.add_argument("--suite", help=argparse.SUPPRESS)
    ap.add_argument("--backend", help=argparse.SUPPRESS)
    ap.add_argument("--device", help=argparse.SUPPRESS)
    ap.add_argument("--dtype", help=argparse.SUPPRESS)
    ap.add_argument("--ks", type=lambda s: [int(v) for v in s.split(",")], help=argparse.SUPPRESS)
    ap.add_argument("--solve-ks", type=lambda s: [int(v) for v in s.split(",")], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return run_group(args)
    if args.report:
        return report(args.report)
    cfg = PROFILES[args.profile]
    devices = [d for d in args.devices.split(",") if d]
    if "cuda" in devices:
        import torch

        if not torch.cuda.is_available():
            print("no CUDA device: running the CPU part only", flush=True)
            devices = [d for d in devices if d != "cuda"]
    backends = [b for b in args.backends.split(",") if b]
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = args.out or os.path.join(ROOT, "results", "runs", f"{socket.gethostname()}_{stamp}_sparse")
    os.makedirs(os.path.join(run_dir, "matrices"), exist_ok=True)
    json.dump({**environment(devices), "profile": args.profile, "devices": devices},
              open(os.path.join(run_dir, "environment.json"), "w"), indent=1)
    out_path = os.path.join(run_dir, "results.jsonl")
    done = set()
    if os.path.exists(out_path):  # resume: skip finished groups
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["case"], r["suite"], r["backend"], r["device"], r["dtype"]))
    repeats = args.repeats or cfg["repeats"]
    timeout = args.timeout or cfg["timeout"]
    for case in cfg["cases"]:
        print(f"building {case}", flush=True)
        try:
            meta = build_case(case, os.path.join(run_dir, "matrices"))
        except Exception as err:  # noqa: BLE001
            print(f"  build failed: {err}", flush=True)
            continue
        print(f"  {meta['case']}: nfree={meta['nfree']}, "
              + ", ".join(f"{k} {v['shape'][0]}x{v['shape'][1]} nnz={v['nnz']}" for k, v in meta["matrices"].items()),
              flush=True)
        mpath = os.path.join(run_dir, "matrices", meta["case"] + ".npz")
        for suite, b, dev, dt in groups(meta, cfg, devices, backends):
            key = (meta["case"], suite, b, dev, dt)
            if key in done:
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--matrices", mpath, "--suite", suite,
                   "--backend", b, "--device", dev, "--dtype", dt, "--repeats", str(repeats),
                   "--ks", ",".join(map(str, cfg["ks"])), "--solve-ks", ",".join(map(str, cfg["solve_ks"]))]
            t = time.perf_counter()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                lines = [ln[4:] for ln in proc.stdout.splitlines() if ln.startswith("REC ")]
                recs = [json.loads(ln) for ln in lines]
                if proc.returncode != 0:  # crash (segfault, OOM kill): record what is missing
                    recs.append({"case": meta["case"], "suite": suite, "backend": b, "device": dev, "dtype": dt,
                                 "status": "crash", "returncode": proc.returncode, "error": proc.stderr[-500:]})
            except subprocess.TimeoutExpired as exc:
                out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                recs = [json.loads(ln[4:]) for ln in out.splitlines() if ln.startswith("REC ")]
                recs.append({"case": meta["case"], "suite": suite, "backend": b, "device": dev, "dtype": dt,
                             "status": "timeout", "timeout_s": timeout})
            with open(out_path, "a") as fh:
                for r in recs:
                    fh.write(json.dumps({**r, "nfree": meta["nfree"]}, default=float) + "\n")
            states = {r.get("status") for r in recs}
            print(f"  {suite:7s} {b:26s} {dev:4s} {dt:7s} {time.perf_counter() - t:7.1f}s {sorted(states)}",
                  flush=True)
    report(run_dir)


# ------------------------------------------------------------------ report


def _fmt(t):
    return "-" if t is None or not np.isfinite(t) else (f"{t * 1e6:.0f}us" if t < 1e-3 else
                                                        f"{t * 1e3:.2f}ms" if t < 1 else f"{t:.2f}s")


def report(run_dir):
    recs = [json.loads(line) for line in open(os.path.join(run_dir, "results.jsonl"))]
    env = json.load(open(os.path.join(run_dir, "environment.json")))
    out = [f"# Sparse backends: {os.path.basename(os.path.abspath(run_dir))}\n",
           f"- host {env['host']}, {env.get('cpu_count')} CPUs, GPU: {env.get('cuda', {}).get('device', 'none')}",
           f"- git {env.get('git')}, versions {env['versions']}\n"]
    ok = [r for r in recs if r.get("status") == "ok"]
    out.append("## Products Y = A X (median time; rel_err against SciPy float64)\n")
    out.append("| case | A | k | backend | device | dtype | time | per column | GB/s | rel_err |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted((r for r in ok if r["suite"] == "product"),
                    key=lambda r: (r["case"], r["matrix"], r["k"], r["time_s"])):
        out.append(f"| {r['case']} | {r['matrix']} | {r['k']} | {r['backend']} | {r['device']} | {r['dtype']} | "
                   f"{_fmt(r['time_s'])} | {_fmt(r['per_column_s'])} | {r['gbytes_per_s']:.1f} | {r['rel_err']:.1e} |")
    out.append("\n## SPD solves K X = R (set-up: conversion + factorisation; repeat/first < 1: factorisation "
               "reused)\n")
    out.append("| case | k | backend | device | dtype | set-up | first | repeat | per rhs | repeat/first | residual | "
               "its |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted((r for r in ok if r["suite"] == "solve"), key=lambda r: (r["case"], r["k"], r["per_rhs_s"])):
        out.append(f"| {r['case']} | {r['k']} | {r['backend']} | {r['device']} | {r['dtype']} | {_fmt(r['setup_s'])} | "
                   f"{_fmt(r['first_solve_s'])} | {_fmt(r['solve_s'])} | {_fmt(r['per_rhs_s'])} | "
                   f"{r['repeat_vs_first']:.2f} | {r['residual']:.1e} | {r.get('iterations', '-')} |")
    bad = [r for r in recs if r.get("status") not in ("ok",)]
    out.append("\n## Not measured (n/a, errors, crashes, timeouts)\n")
    seen = set()
    for r in bad:
        key = (r["suite"], r["backend"], r["device"], r["dtype"], r.get("status"), (r.get("error") or "")[:120])
        if key not in seen:
            seen.add(key)
            out.append(f"- {r['suite']} {r['backend']} {r['device']} {r['dtype']}: {r.get('status')} "
                       f"{(r.get('error') or '').strip()[:160]}")
    path = os.path.join(run_dir, "report.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
