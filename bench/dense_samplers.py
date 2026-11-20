"""Dense samplers (ngsrng.dense): set-up, throughput and per-call cost, against the route they replace.

    python bench/dense_samplers.py                        # every case on CUDA (and the trace case on the CPU)
    python bench/dense_samplers.py --only trace,lognormal # some cases (substrings of their names)
    python bench/dense_samplers.py --device cpu           # all on the CPU

Cases:

- trace: boundary data, two uniform(-1, 20) fields correlated 0.8 (Bernstein, order-4 H1, ``bnd="left"``, stationary
  Matern torus priors conditioned at the two corners of the edge): A = the trace DoFs (2 x 57), B = 65 nodes of the
  edge. Target: >= 1e6 draws per second on an A100.
- subdomain: a Gaussian field on a disc (normal, stationary Matern torus on ``Compress(H1(order 2, definedon=disc,
  dirichlet=disc edge))``): A = all DoFs, B = grid nodes inside the disc. Target: >= 1e5 draws per second.
- square: a Gaussian field (Matern PDE precision), order-2 H1: B = a 65 x 65 grid.
- lognormal: a lognormal field (Bernstein, order-1 H1): B = a 91 x 91 grid (a sparse output map).

Recorded per case and sampler: the dense set-up (seconds), latent values and rank (and whether the output map was
folded into the factor), draws per second at a large batch (float32 and float64, device and host noise), the cost of
a small call (n = 20), and the draws per second of the route it replaces (``d.sample``, then the
observation). Results go to results/runs/<host>_<date>_dense/ (results.jsonl, report.md).
"""

import argparse
import json
import os
import socket
import statistics
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import torch  # before ngsolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ngsolve as ngs  # noqa: E402
from netgen.geom2d import SplineGeometry  # noqa: E402

import ngsrng  # noqa: E402
import ngsrng.resolution  # noqa: E402

ngsrng.resolution.RESOLUTION_TOL = None  # timings at fixed mesh sizes, resolved or not: no resolution check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, reps=3):
    """Median seconds of ``fn()`` over ``reps`` calls (after one warm-up call)."""
    fn()
    sync(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ------------------------------------------------------------------ cases


def trace(device):
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (2, 1), bcs=["bottom", "right", "top", "left"])
    fes = ngs.H1(ngs.Mesh(geo.GenerateMesh(maxh=1 / 14)), order=4)
    corners = np.array([[0.0, 0.0], [0.0, 1.0]])
    dists = []
    t0 = time.perf_counter()
    for k in range(2):
        prior = ngsrng.StationaryMatern(fes, 0.3, nu=2, construction="torus", seed=k, device=device)
        post = prior.condition(points=corners, noise=1e-10, data=[1.0, 2.0])
        rng = ngsrng.default_rng(fes, seed=k, correlation=post, device=device, dtype="float32")
        dists.append(rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left"))
    pair = ngsrng.correlated(dists, 0.8)
    pair.sample(2)  # the route's lazy set-up (boundary sampler)
    route_setup = time.perf_counter() - t0
    edge = ngsrng.observation_operator(fes, points=np.column_stack([np.zeros(65), np.linspace(0, 1, 65)]))
    samplers = {"A: trace DoFs": lambda: pair.dense(), "B: 65 edge nodes": lambda: (edge @ pair).dense()}

    def route(n):  # the route these replace: draws of the whole fields, restricted (A)
        return pair.sample(n, output="torch")

    return fes, samplers, route_setup, route, 100


def _disc_space(maxh=0.022):
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), leftdomain=1, rightdomain=0, bc="outer")
    geo.AddCircle((0.5, 0.5), 0.35, leftdomain=2, rightdomain=1, bc="discedge")
    geo.SetMaterial(1, "outer")
    geo.SetMaterial(2, "disc")
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=maxh))
    return ngs.Compress(ngs.H1(mesh, order=2, definedon="disc", dirichlet="discedge"))


def _grid(n, inside=None):
    x = np.linspace(0, 1, n)
    pts = np.array([(a, b) for b in x for a in x])
    return pts if inside is None else pts[inside(pts)]


def subdomain(device):
    fes = _disc_space()
    t0 = time.perf_counter()
    prior = ngsrng.StationaryMatern(fes, 0.1, nu=5, construction="torus", seed=0, device=device)
    rng = ngsrng.default_rng(fes, seed=0, correlation=prior, device=device, dtype="float32")
    d = rng.distribution("normal")
    route_setup = time.perf_counter() - t0
    pts = _grid(73, lambda p: np.linalg.norm(p - 0.5, axis=1) < 0.35 - 1e-3)
    e = ngsrng.observation_operator(fes, points=pts)
    samplers = {"A: all DoFs": lambda: d.dense(), f"B: {len(pts)} grid nodes": lambda: (e @ d).dense()}
    et = {}

    def route(n):  # draws of the field, then the observation (B)
        x = d.sample(n, output="torch")
        if x.device not in et:
            et[x.device] = torch.sparse_csr_tensor(*(torch.from_numpy(a) for a in (e.indptr, e.indices, e.data)),
                                                   size=e.shape).to(x.device, x.dtype)
        return (et[x.device] @ x.T).T

    return fes, samplers, route_setup, route, 200


def square(device):
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.05)), order=2)
    t0 = time.perf_counter()
    rng = ngsrng.default_rng(fes, seed=0, correlation=ngsrng.matern_correlation(0.2, nu=1), device=device,
                             dtype="float32")
    d = rng.distribution("normal", 0.0, 1.0)
    d.sample(2)
    route_setup = time.perf_counter() - t0
    e = ngsrng.observation_operator(fes, points=_grid(65))
    samplers = {"B: 65 x 65 grid": lambda: (e @ d).dense()}
    return fes, samplers, route_setup, _observed_route(d, e), 200


def lognormal(device):
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.05)), order=1)
    t0 = time.perf_counter()
    rng = ngsrng.default_rng(fes, seed=0, correlation=ngsrng.matern_correlation(0.2, nu=1), device=device,
                             dtype="float32")
    d = rng.distribution("lognormal", 0.0, 1.0, representation="bernstein")
    d.sample(2)
    route_setup = time.perf_counter() - t0
    e = ngsrng.observation_operator(fes, points=_grid(91))
    samplers = {"B: 91 x 91 grid": lambda: (e @ d).dense()}
    return fes, samplers, route_setup, _observed_route(d, e), 200


def _observed_route(d, e):
    et = {}

    def route(n):
        x = d.sample(n, output="torch")
        if x.device not in et:
            et[x.device] = torch.sparse_csr_tensor(*(torch.from_numpy(a) for a in (e.indptr, e.indices, e.data)),
                                                   size=e.shape).to(x.device, x.dtype)
        return (et[x.device] @ x.T).T

    return route


CASES = {"trace": trace, "subdomain": subdomain, "square": square, "lognormal": lognormal}


# ------------------------------------------------------------------ measurements


def batch_for(sampler, budget=2**29):
    """A large batch whose outputs take at most ``budget`` bytes in float64 (and at most 10^6 draws)."""
    per = 8 * sampler.components * (sampler.nout if isinstance(sampler.nout, int) else max(sampler.nout))
    return int(min(10**6, max(1000, budget // per)))


def measure(name, sampler_fn, device, route, route_n, args):
    sync(device)
    t0 = time.perf_counter()
    base = sampler_fn()
    sync(device)
    setup = time.perf_counter() - t0
    rec = {"sampler": name, "setup_s": setup, "latent": base.latent_points, "rank": base.rank,
           "outputs": base.nout, "components": base.components, "folded": bool(base.info.get("folded")),
           "latent_kind": base.info.get("latent"), "covariance": base.info.get("covariance")}
    rates = {}
    for dtype in ("float32", "float64"):
        for noise in ("device", "host") if torch.device(device).type == "cuda" else ("host",):
            s = base.to(device=device, dtype=dtype, noise=noise, seed=1)
            n = batch_for(s)
            t = timed(lambda s=s, n=n: s.sample(n, output="torch"), device, args.reps)
            rates[f"{dtype}/{noise}"] = n / t
    rec["draws_per_s"] = rates
    small = base.to(device=device, dtype="float32", seed=1)
    calls = 200
    t = timed(lambda: [small.sample(20, output="torch") for _ in range(calls)], device, 1)
    rec["call_us_n20"] = 1e6 * t / calls
    rec["draws_per_s_n20"] = 20 * calls / t
    tr = timed(lambda: route(route_n), device, 1)
    rec["route_draws_per_s"] = route_n / tr
    rec["speedup"] = rates["float32/" + ("device" if torch.device(device).type == "cuda" else "host")] / \
        rec["route_draws_per_s"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="comma-separated substrings of case names")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-cpu-trace", dest="cpu_trace", action="store_false",
                    help="skip the trace case on the CPU (run besides the device's)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    os.environ.setdefault("NGSRNG_QUIET", "1")
    if args.device == "cuda" and not ngsrng.backend.cuda_usable():
        sys.exit("no usable CUDA device: pass --device cpu")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or os.path.join(ROOT, "results", "runs", f"{socket.gethostname()}_{stamp}_dense")
    os.makedirs(out_dir, exist_ok=True)
    runs = [(c, args.device) for c in CASES if not args.only or any(s in c for s in args.only.split(","))]
    if args.cpu_trace and args.device != "cpu" and ("trace", args.device) in runs:
        runs.append(("trace", "cpu"))
    recs = []
    with open(os.path.join(out_dir, "results.jsonl"), "w") as fh:
        for case, device in runs:
            fes, samplers, route_setup, route, route_n = CASES[case](device)
            print(f"== {case} on {device}: {fes.ndof} dofs, route set-up {route_setup:.1f} s", flush=True)
            for name, fn in samplers.items():
                rec = {"case": case, "device": device, "ndof": fes.ndof, "route_setup_s": route_setup,
                       **measure(name, fn, device, route, route_n, args)}
                rates = ", ".join(f"{k} {v:.3g}" for k, v in rec["draws_per_s"].items())
                print(f"  {name:<22} set-up {rec['setup_s']:6.2f} s | latent {rec['latent']:5d} rank {rec['rank']:5d}"
                      f"{' folded' if rec['folded'] else ''} | draws/s {rates} | n=20: {rec['call_us_n20']:.0f} us/call"
                      f" | route {rec['route_draws_per_s']:.3g}/s (x{rec['speedup']:.3g})", flush=True)
                recs.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            del fes, samplers, route
            if device == "cuda":
                torch.cuda.empty_cache()
    write_report(out_dir, recs)
    print(f"\nreport: {os.path.join(out_dir, 'report.md')}")


def write_report(out_dir, recs):
    lines = ["# Dense samplers\n",
             "draws/s: dtype/noise at a large batch; n=20: the cost of one call drawing 20 samples; "
             "route: the draws/s of d.sample (then the observation) it replaces, and the speedup over it "
             "(float32, device noise on a GPU).\n",
             "| case | device | sampler | set-up s | latent | rank | f32/device | f64/device | f32/host | f64/host | "
             "n=20 us/call | route /s | speedup |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        d = r["draws_per_s"]
        lines.append(f"| {r['case']} ({r['ndof']} dofs) | {r['device']} | {r['sampler']} | {r['setup_s']:.2f} | "
                     f"{r['latent']} | {r['rank']}{' (folded)' if r['folded'] else ''} | "
                     + " | ".join(f"{d[k]:.3g}" if k in d else "" for k in
                                  ("float32/device", "float64/device", "float32/host", "float64/host"))
                     + f" | {r['call_us_n20']:.0f} | {r['route_draws_per_s']:.3g} | {r['speedup']:.3g} |")
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
