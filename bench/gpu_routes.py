"""Device routes against host routes: everything that can process a GPU Gaussian's samples on the device or on the host.

    python bench/gpu_routes.py                       # CUDA: 2D P2 and 3D P1 at 1e4 and 1e5 dofs
    python bench/gpu_routes.py --quick               # 2D P2 at 1e4 dofs (a few minutes)
    python bench/gpu_routes.py --targets 1e5,1e6     # other sizes
    python bench/gpu_routes.py --only projection,kl  # some cases (substrings of their names)

Every case runs its device route (``compute="auto"`` on a GPU: the samples never leave it) and its host route
(``compute="host"``: NumPy and NGSolve on the CPU, the route before the device paths existed), in interleaved
repetitions (median times, so that load on a shared GPU affects both alike), and records

- the time of each route, their ratio and a verdict (``gain``: the device route at least 10% faster, ``loss``: at
  least 10% slower, ``tie``),
- the relative difference of their results on the same random input (they compute the same thing),
- the host<->device bytes per call of each route (torch.profiler; ``--no-traffic`` skips this).

Cases (fix numbers of the GPU audit): the in-place fill against the allocating fallback (1); pointwise families at
quadrature points, in the L2 projection, in the Bernstein form, with ``bnd=`` (2); KL set-up and sampling,
LowRankGaussian sampling, Matheron set-up and sampling (3); posterior factorisations, pointwise variances (4); and
apply_precision on GPU blocks, ``transformed`` with NumPy's and torch's own functions, stationary Matérn fields
(FFT and torus) on the GPU against the CPU (5).

Results go to results/runs/<host>_<date>_gpuroutes/ (results.jsonl, report.md). A loss is a reason to change the
default route (``compute`` / the KL, posterior and covariance set-ups default to the device on a GPU).
"""

import argparse
import gc
import json
import os
import socket
import statistics
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import torch  # before ngsolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ngsrng  # noqa: E402
from ngsrng.calibration.common import make_space  # noqa: E402
from ngsrng.inplace import chunks  # noqa: E402
from ngsrng.quadrature import QuadratureRule  # noqa: E402
from ngsrng.seeding import one_draw  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIE = 1.10  # within 10%: a tie


def sync():
    torch.cuda.synchronize()


def rel(a, b):
    a = a.detach().double().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, dtype=float)
    b = b.detach().double().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b, dtype=float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def interleaved(fa, fb, reps, min_time=0.5):
    """Median times of fa and fb, measured alternately (after one warm-up call each)."""
    fa(), fb()
    sync()
    ta, tb, start = [], [], time.perf_counter()
    while len(ta) < reps or (time.perf_counter() - start < min_time and len(ta) < 5 * reps):
        for f, out in ((fa, ta), (fb, tb)):
            t0 = time.perf_counter()
            f()
            sync()
            out.append(time.perf_counter() - t0)
    return statistics.median(ta), statistics.median(tb)


def traffic(fn):
    """(bytes host->device, bytes device->host) of one call."""
    from torch.profiler import ProfilerActivity, profile

    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        sync()
    path = os.path.join(tempfile.mkdtemp(), "trace.json")
    prof.export_chrome_trace(path)
    with open(path) as fh:
        events = json.load(fh)["traceEvents"]
    out = {"HtoD": 0, "DtoH": 0}
    for e in events:
        for k in out:
            if e.get("name", "").startswith("Memcpy") and k in e["name"]:
                out[k] += int(e.get("args", {}).get("bytes", 0))
    os.remove(path)
    return out["HtoD"], out["DtoH"]


def seeded(fn, seed=0):
    """fn() with torch's global generator reset: draw-time randomness makes both routes draw the same Gaussians."""
    def run():
        torch.manual_seed(seed)
        return fn()
    return run


# ------------------------------------------------------------------ cases
# each returns {"device": (call, result_for_comparison or None), "host": (...)} and an optional note; set-up cases
# time the set-up itself (a fresh object per call)


def cases(fes, g, args):
    space = g.space
    k = args.batch
    pts = np.random.default_rng(0).uniform(0.15, 0.85, (args.obs, fes.mesh.dim))

    lazy = {}  # objects are made on first use: --only skips the set-ups of other cases

    def once(key, make):
        if key not in lazy:
            lazy[key] = make()
        return lazy[key]

    def rng():
        return ngsrng.default_rng(fes, seed=7, correlation=g, device="cuda", dtype="float64")

    def family(name, *a, **opts):
        def dist(c):
            key = ("family", name, a, tuple(sorted(opts.items())), c)
            return once(key, lambda: rng().distribution(name, *a, compute=c, **opts))
        return {"device": seeded(lambda: dist("auto").sample(k)), "host": seeded(lambda: dist("host").sample(k)),
                "dist": dist}

    buf = torch.empty((k, space.ndof), device="cuda", dtype=g.dtype)

    def old_fill():
        """sample_into before the device fix: per chunk, the pre-fix sample() -- the mean and the free-dof indices
        uploaded from the host on every call, a full-size intermediate -- and a copy; as one draw (as sample_into
        is), so that both routes draw the same numbers."""
        with one_draw():
            for k0, k1 in chunks(k, g._per_sample(), g.work_values, g._min_chunk()):
                n = k1 - k0
                eta = torch.randn(g.nrand, n, generator=g.generator, device=g.device, dtype=g._work_dtype)
                c = g.apply(eta).to(g.dtype)
                full = torch.from_numpy(g._mean).to(g.device, g.dtype)[:, None].repeat(1, n)
                full[torch.from_numpy(g.space.free).to(g.device)] += c
                buf[k0:k1].copy_(full.T)
        return buf.clone()

    out = {
        "1 in-place fill: direct vs pre-fix fill": {
            "device": seeded(lambda: g.sample_into(buf).clone()),
            "host": seeded(old_fill),
        },
        "2 uniform, quadrature points": family("uniform"),
        "2 gamma(2), quadrature points (table)": family("gamma", 2.0, method="table"),
        "2 uniform, L2 projection": family("uniform", representation="projection"),
        "2 uniform, Bernstein": family("uniform", representation="bernstein"),
        "2 dirichlet(1,2,3), Bernstein": family("dirichlet", (1.0, 2.0, 3.0), representation="bernstein"),
        "2 uniform, bnd='bottom' (Bernstein)": family("uniform", bnd="bottom"),
    }
    def low_rank(device, seed):  # the same model on the GPU and on the CPU (NumPy), same seed: the same draws
        kl = once("kl", lambda: ngsrng.karhunen_loeve(g, args.rank))
        return ngsrng.LowRankGaussian(fes, kl.b, kl.k, seed=seed, device=device)

    out["3 KL set-up (rank %d)" % args.rank] = {
        "device": lambda: ngsrng.karhunen_loeve(g, args.rank, compute="auto").eigenvalues,
        "host": lambda: ngsrng.karhunen_loeve(g, args.rank, compute="host").eigenvalues,
        "setup": True,
    }
    # host routes deliver their samples on the GPU too: the comparison is about GPU workflows
    out["3 LowRankGaussian sampling (rank %d)" % args.rank] = {
        "device": lambda: once("lr-d", lambda: low_rank("cuda", 3)).sample(k),
        "host": lambda: once("lr-h", lambda: low_rank("cpu", 3)).sample(k).to("cuda"),
        "compare": (lambda: low_rank("cuda", 4).sample(k), lambda: low_rank("cpu", 4).sample(k)),
    }
    kl_family = family("karhunen_loeve", rank=args.rank)

    probes = np.random.default_rng(0).standard_normal((space.nfree, 8))

    def kl_covariance(c):  # the modes depend on the eigensolver (signs; rotations within degenerate eigenspaces):
        w = kl_family["dist"](c).field.scaled_modes  # compare the covariance W W^T through probes, O(nfree rank)
        return w @ (w.T @ probes)
    out["3 KL family sampling (rank %d)" % args.rank] = {
        **kl_family, "compare": (lambda: kl_covariance("auto"), lambda: kl_covariance("host"))}

    def obs():
        return once("obs", lambda: ngsrng.LinearObservations(g, points=pts, noise=1e-2, factorize=False))

    def matheron(c):
        return once(("mat", c), lambda: ngsrng.MatheronPosterior(obs(), data=np.ones(args.obs), compute=c))

    out[f"3 Matheron set-up ({args.obs} points)"] = {
        "device": lambda: ngsrng.MatheronPosterior(obs(), data=np.ones(args.obs), compute="auto").mean,
        "host": lambda: ngsrng.MatheronPosterior(obs(), data=np.ones(args.obs), compute="host").mean,
        "setup": True,
    }
    def eta():  # the same noise for both routes (made with the case, released with it)
        return once("eta", lambda: torch.randn(g.nrand + args.obs, k, device="cuda", dtype=torch.float64))
    out[f"3 Matheron sampling ({args.obs} points)"] = {
        "device": lambda: matheron("auto").sample(k), "host": lambda: matheron("host").sample(k).to("cuda"),
        "compare": (lambda: matheron("auto").apply(eta()), lambda: matheron("host").apply(eta())),
    }
    for rank in (None, 32):
        label = "exact" if rank is None else f"rank {rank}"

        def fact(c, rank=rank):
            o = ngsrng.LinearObservations(g, points=pts, noise=1e-2, rank=rank, compute=c)
            o._get_lq()  # L Q: part of the FE-space set-up
            return o.singular_values
        out[f"4 posterior factorisation ({label}, {args.obs} points)"] = {
            "device": lambda f=fact: f("auto"), "host": lambda f=fact: f("host"), "setup": True}
    e = QuadratureRule(space).evaluation  # values at 2000 of the points: the routes differ in the set-up only
    e = e[np.sort(np.random.default_rng(1).choice(e.shape[0], min(2000, e.shape[0]), replace=False))]
    for method, opts in (("lowrank", {"rank": 200, "samples": 64}), ("mc", {"samples": 400})):
        def lc(c, method=method, opts=opts):
            g.__dict__.pop("_ngsrng_covariance_cache", None)  # no cached result: time the set-up
            return ngsrng.local_covariance(g, method, compute=c, **opts).values(e)
        out[f"4 pointwise variance ({method})"] = {"device": lambda f=lc: f("auto"), "host": lambda f=lc: f("host"),
                                                   "setup": True}
    def block():
        return once("c", lambda: torch.randn(space.nfree, k, device="cuda", dtype=torch.float64))
    # the host route: the block to the CPU, SciPy and NGSolve, back
    out[f"5 apply_precision (power {g.spec.power:g})"] = {
        "device": lambda: g.apply_precision(block()), "host": lambda: g.apply_precision(block().cpu()).to("cuda")}
    out["5 transformed(np.tanh)"] = family("transformed", np.tanh)
    out["5 transformed(torch.tanh)"] = family("transformed", torch.tanh)

    def stationary(construction, device):
        return once(("stationary", construction, device), lambda: ngsrng.StationaryMatern(
            fes, 0.3, 1.0, construction=construction, method="fft" if construction == "restrict" else "auto",
            seed=1, device=device))
    for construction in ("restrict", "torus"):
        label = "FFT" if construction == "restrict" else "torus"
        out[f"5 stationary Matérn sampling ({label})"] = {
            "device": lambda c_=construction: stationary(c_, "cuda").sample(k),
            "host": lambda c_=construction: stationary(c_, "cpu").sample(k).to("cuda"),
            "compare": (lambda c_=construction: stationary(c_, "cuda").apply(stationary_eta(c_)),
                        lambda c_=construction: stationary(c_, "cpu").apply(stationary_eta(c_))),
        }

    def stationary_eta(construction):  # the same noise for both routes
        return once(("eta", construction), lambda: torch.randn(stationary(construction, "cpu").nrand, 4,
                                                              dtype=torch.float64))
    out["_lazy"] = lazy  # the objects the cases made: released after every case (see main)
    return out


def run_case(name, case, args):
    rec = {"case": name}
    fd, fh = case["device"], case["host"]
    reps = 2 if case.get("setup") else args.reps
    td, th = interleaved(fd, fh, reps, min_time=0.0 if case.get("setup") else 0.5)
    rec.update(device_s=td, host_s=th, ratio=th / td)
    rec["verdict"] = "gain" if th > TIE * td else ("loss" if td > TIE * th else "tie")
    ca, cb = case.get("compare", (fd, fh))
    rec["rel_diff"] = rel(ca(), cb())
    if args.traffic:
        rec["device_h2d"], rec["device_d2h"] = traffic(fd)
        rec["host_h2d"], rec["host_d2h"] = traffic(fh)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--targets", default=None, help="dof targets, e.g. 1e4,1e5")
    ap.add_argument("--spaces", default=None, help="e.g. 2-H1-2,3-H1-1 (dim-kind-order)")
    ap.add_argument("--only", default=None, help="comma-separated substrings of case names")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--obs", type=int, default=100)
    ap.add_argument("--rank", type=int, default=50)
    ap.add_argument("--no-traffic", dest="traffic", action="store_false")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not ngsrng.backend.cuda_usable():
        sys.exit("gpu_routes needs a usable CUDA device")
    targets = [float(t) for t in (args.targets or ("1e4" if args.quick else "1e4,1e5")).split(",")]
    spaces = [tuple(int(x) if i != 1 else x for i, x in enumerate(s.split("-")))
              for s in (args.spaces or ("2-H1-2" if args.quick else "2-H1-2,3-H1-1")).split(",")]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or os.path.join(ROOT, "results", "runs", f"{socket.gethostname()}_{stamp}_gpuroutes")
    os.makedirs(out_dir, exist_ok=True)
    recs = []
    warnings.simplefilter("ignore")
    with open(os.path.join(out_dir, "results.jsonl"), "w") as fh:
        for dim, kind, order in spaces:
            for target in targets:
                fes = make_space(dim, kind, order, target)
                g = ngsrng.Gaussian(fes, ngsrng.matern_correlation(0.3, nu=1 if dim == 2 else 1.5).bind(fes),
                                    device="cuda", dtype="float64", seed=None)
                print(f"== {dim}D {kind} P{order}: {fes.ndof} dofs ({g.plan.get('solver')})", flush=True)
                all_cases = cases(fes, g, args)
                lazy = all_cases.pop("_lazy")
                for name, case in all_cases.items():
                    if args.only and not any(s in name for s in args.only.split(",")):
                        continue
                    lazy.clear()  # one case's objects at a time: a GPU holds the set-ups of the largest ones only
                    g.__dict__.pop("_ngsrng_covariance_cache", None)
                    gc.collect()
                    torch.cuda.empty_cache()
                    base = {"dim": dim, "kind": kind, "order": order, "ndof": fes.ndof, "batch": args.batch}
                    try:
                        rec = {**base, **run_case(name, case, args)}
                        print(f"  {name:<52} device {rec['device_s'] * 1e3:9.1f} ms | host {rec['host_s'] * 1e3:9.1f} "
                              f"ms | {rec['verdict']:<4} x{rec['ratio']:.2f} | diff {rec['rel_diff']:.1e}", flush=True)
                    except Exception as err:  # noqa: BLE001 - a failure is a result
                        rec = {**base, "case": name, "error": f"{type(err).__name__}: {err}"[:300],
                               "traceback": traceback.format_exc()[-1200:]}
                        print(f"  {name:<52} ERROR {rec['error']}", flush=True)
                    recs.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                del g
                torch.cuda.empty_cache()
    write_report(out_dir, recs)
    print(f"\nreport: {os.path.join(out_dir, 'report.md')}")


def write_report(out_dir, recs):
    lines = ["# Device routes vs host routes\n",
             f"{len(recs)} measurements; device route = compute='auto' on the GPU, host route = compute='host'. "
             "Times are medians of interleaved repetitions (set-ups: of fresh objects); ratio = host / device "
             f"(> 1: the device route is faster); a tie is within {TIE - 1:.0%}. diff: relative difference of the "
             "two routes' results on the same random input.\n",
             "| space | ndof | case | device | host | ratio | verdict | diff | device H2D/D2H | host H2D/D2H |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def mb(r, a, b):
        return f"{r[a] / 2**20:.2f} / {r[b] / 2**20:.2f} MB" if a in r else ""

    for r in recs:
        space = f"{r['dim']}D {r['kind']} P{r['order']}"
        if "error" in r:
            lines.append(f"| {space} | {r['ndof']} | {r['case']} | error: {r['error']} | | | | | | |")
            continue
        lines.append(f"| {space} | {r['ndof']} | {r['case']} | {r['device_s'] * 1e3:.1f} ms "
                     f"| {r['host_s'] * 1e3:.1f} ms | {r['ratio']:.2f} | {r['verdict']} | {r['rel_diff']:.1e} "
                     f"| {mb(r, 'device_h2d', 'device_d2h')} | {mb(r, 'host_h2d', 'host_d2h')} |")
    losses = [r for r in recs if r.get("verdict") == "loss"]
    verdict = (f"{len(losses)} loss(es): " + "; ".join(f"{r['case']} at {r['ndof']} dofs" for r in losses)
               if losses else "No losses: the device routes are at least as fast as the host routes here.")
    errors = [r for r in recs if "error" in r]
    if errors:
        verdict += f" {len(errors)} case(s) failed: " + "; ".join(f"{r['case']} at {r['ndof']} dofs" for r in errors)
    lines.append("\n" + verdict)
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(verdict)


if __name__ == "__main__":
    main()
