"""Stationary Matérn priors (``ngsrng.StationaryMatern``) against Matérn priors with Neumann and Robin conditions:
accuracy first, then cost.

    python bench/stationary.py                     # standard profile (about half an hour on a laptop)
    python bench/stationary.py --profile quick     # a few minutes
    python bench/stationary.py --out-dir DIR

For every case (domain, mesh size, element order, nu; correlation length 0.3) and prior it measures:

- the exact pointwise variance of the discretised field at every mesh vertex (|L^T e_x|^2): min, max and the rms of
  log(variance) over all vertices and over the boundary vertices (the target is 1 everywhere);
- the correlation error against the Matérn correlation of R^d: max and rms of |rho_h(x, y) - rho_M(|x - y|)| over
  random vertex pairs closer than two correlation lengths (all pairs, and pairs with a point within one length of
  the boundary);
- set-up seconds (construction and first draw) and milliseconds per sample (in-place batch of 16);
- method details: torus cells, dofs and padding; FFT grid and embedding; dense rank; Robin coefficient.

Priors: "neumann" and "robin" (``matern_correlation``, ``beta=None``: the estimated coefficient), "torus"
(construction="torus", tol=1e-3), "restrict-fft", "restrict-dense" (small cases only) and "torus+dirichlet" (the
boundary update on a space with dirichlet="left"; its statistics leave out the vertices within one length of the
Dirichlet edge, whose variance is meant to drop to zero). Each prior runs in its own process with a time limit.
Results: results/runs/<host>_<date>_stationary/ (results.jsonl, report.md) or --out-dir.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LENGTH = 0.3

PROFILES = {
    "quick": [("square", 0.1, 2, 1.0), ("square", 0.1, 2, 2.0), ("lshape", 0.1, 2, 1.0),
              ("square", 0.1, 1, 2.0)],  # P1: few enough quadrature points for the exact dense kernel
    "standard": [("square", 0.05, 2, 1.0), ("square", 0.05, 2, 2.0), ("square", 0.05, 2, 5.0),
                 ("square", 0.08, 1, 2.0),
                 ("square", 0.025, 2, 1.0), ("lshape", 0.05, 2, 1.0), ("holes", 0.05, 2, 2.0),
                 ("cube", 0.15, 1, 1.0)],
}
PRIORS = ["neumann", "robin", "torus", "restrict-fft", "restrict-dense", "torus+dirichlet"]


def make_mesh(domain, h):
    import ngsolve as ngs

    if domain == "square":
        return ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=h))
    if domain == "cube":
        from netgen.csg import unit_cube
        return ngs.Mesh(unit_cube.GenerateMesh(maxh=h))
    from netgen.occ import Circle, MoveTo, OCCGeometry, Rectangle

    if domain == "lshape":
        shape = Rectangle(1, 1).Face() - MoveTo(0.5, 0.5).Rectangle(0.5, 0.5).Face()
    elif domain == "holes":  # a disk with two holes: curved boundaries, three boundary components
        shape = Circle((0.5, 0.5), 0.5).Face() - Circle((0.35, 0.5), 0.1).Face() - Circle((0.7, 0.55), 0.08).Face()
    else:
        raise ValueError(domain)
    mesh = ngs.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=h))
    mesh.Curve(3)
    return mesh


def run_one(domain, h, order, nu, prior, seed=0):
    """One prior on one case, in this process: a dict of results."""
    sys.path.insert(0, ROOT)
    import ngsrng  # noqa: F401, I001  (torch before NGSolve: known issue 1)
    import ngsolve as ngs
    import numpy as np
    import torch
    from ngsrng.stationary import MAX_DENSE, basis_at, matern_function

    mesh = make_mesh(domain, h)
    dirichlet = prior == "torus+dirichlet"
    if dirichlet and len(set(mesh.GetBoundaries())) < 2:
        return {"domain": domain, "h": h, "order": order, "nu": nu, "prior": prior,
                "skipped": "one boundary name: no Dirichlet edge separate from the rest"}
    bnd_name = mesh.GetBoundaries()[0] if dirichlet else None
    fes = ngs.H1(mesh, order=order, dirichlet=bnd_name) if dirichlet else ngs.H1(mesh, order=order)
    d = mesh.dim
    out = {"domain": domain, "h": h, "order": order, "nu": nu, "prior": prior, "ndof": fes.ndof}
    t0 = time.perf_counter()
    if prior in ("neumann", "robin"):
        kw = {"boundary": "robin", "beta": None} if prior == "robin" else {}
        corr = ngsrng.matern_correlation(LENGTH, nu, **kw)
        g = ngsrng.Gaussian(fes, corr.bind(fes), dtype="float64", seed=seed)
        if prior == "robin":
            out["beta_over_kappa"] = corr.robin_beta(fes) / corr.kappa()
    else:
        cons, meth = {"torus": ("torus", "auto"), "torus+dirichlet": ("torus", "auto"),
                      "restrict-fft": ("restrict", "fft"), "restrict-dense": ("restrict", "dense")}[prior]
        if meth == "dense":
            from ngsrng.quadrature import QuadratureRule
            from ngsrng.space import SpaceInfo
            if QuadratureRule.count(SpaceInfo(fes)) > MAX_DENSE:
                return {**out, "skipped": f"more than {MAX_DENSE} quadrature points"}
        g = ngsrng.StationaryMatern(fes, LENGTH, nu, construction=cons, method=meth, seed=seed)
        out["source"] = g.plan["source"]
        out["conditioning"] = g.plan["conditioning"]
    buf = np.empty((fes.ndof, 16))
    g.sample(16, out=buf)
    out["setup_s"] = time.perf_counter() - t0
    reps = 2
    t1 = time.perf_counter()
    for _ in range(reps):
        g.sample(16, out=buf)
    out["ms_per_sample"] = 1e3 * (time.perf_counter() - t1) / (16 * reps)
    out["nrand"] = int(g.nrand)

    # exact variances and correlations at the vertices
    xyz = np.array([v.point for v in mesh.vertices], dtype=float)[:, :d]
    centre = xyz.mean(0)
    pts = xyz + 1e-9 * (centre - xyz)
    bnd = np.zeros(len(xyz), bool)
    for el in mesh.Elements(ngs.BND):
        bnd[[v.nr for v in el.vertices]] = True
    from scipy.spatial import cKDTree

    near = cKDTree(xyz[bnd]).query(xyz)[0] < LENGTH
    keep = np.ones(len(xyz), bool)
    if dirichlet:
        dpts = xyz[[v.nr for el in mesh.Elements(ngs.BND) if el.mat == bnd_name for v in el.vertices]]
        keep = cKDTree(dpts).query(xyz)[0] >= LENGTH
    E = basis_at(fes, pts)[:, g.space.free].tocsc()
    lt = np.empty((g.nrand, len(pts)))
    for j in range(0, len(pts), 256):
        lt[:, j:j + 256] = g.apply_t(torch.from_numpy(E[j:j + 256].T.toarray())).numpy()
    var = (lt**2).sum(0)

    def stats(mask):
        v = var[mask & keep]
        if v.size == 0:  # e.g. the whole boundary carries the Dirichlet condition
            return None
        return {"min": float(v.min()), "max": float(v.max()), "rms_log": float(np.sqrt(np.mean(np.log(v) ** 2)))}

    out["variance"] = stats(np.ones(len(xyz), bool))
    out["variance_boundary"] = stats(bnd)
    rng = np.random.default_rng(1)
    i = rng.integers(0, len(xyz), 20000)
    j = rng.integers(0, len(xyz), 20000)
    dist = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    sel = (dist > 0.05 * LENGTH) & (dist < 2 * LENGTH) & keep[i] & keep[j]
    i, j, dist = i[sel][:3000], j[sel][:3000], dist[sel][:3000]
    rho_h = (lt[:, i] * lt[:, j]).sum(0) / np.sqrt(var[i] * var[j])
    err = np.abs(rho_h - matern_function(dist, LENGTH, nu))
    if err.size == 0:
        out["corr_error"] = {"max": float("nan"), "rms": float("nan"), "max_near_boundary": None, "pairs": 0}
        return out
    bpair = near[i] | near[j]
    out["corr_error"] = {"max": float(err.max()), "rms": float(np.sqrt(np.mean(err**2))),
                         "max_near_boundary": float(err[bpair].max()) if bpair.any() else None,
                         "pairs": int(len(err))}
    return out


def fmt(r):
    if "skipped" in r:
        return f"| {r['prior']} | skipped: {r['skipped']} | | | | | |"
    if "error" in r:
        return f"| {r['prior']} | failed: {r['error'][:80]} | | | | | |"
    v, vb, c = r["variance"], r["variance_boundary"], r["corr_error"]
    extra = ""
    if "beta_over_kappa" in r:
        extra = f"beta = {r['beta_over_kappa']:.3f} kappa"
    elif "source" in r:
        s = r["source"]
        extra = (f"torus {s['cells']} cells, {s['ndof']} dofs, padding {s['padding']:.2f}" if "padding" in s else
                 f"grid {s['grid']}, embedding {s['embedding']}" if "grid" in s else f"rank {s.get('rank')}")
    def cell(x):
        return "-" if x is None else f"[{x['min']:.3f}, {x['max']:.3f}] {x['rms_log']:.3f}"

    return (f"| {r['prior']} | {cell(v)} | {cell(vb)} | {c['max']:.3f} / {c['rms']:.3f} | "
            f"{r['setup_s']:.1f} | {r['ms_per_sample']:.2f} | {extra} |")


def report(results, path):
    lines = ["# Stationary Matérn priors: accuracy and cost", "",
             f"Correlation length {LENGTH}. Variance: [min, max] and rms of log(variance) at the mesh vertices "
             "(target 1); correlation error: max / rms of |rho_h - rho_Matern| over vertex pairs closer than two "
             "lengths. torus+dirichlet: statistics away from the Dirichlet edge.", ""]
    cases = []
    for r in results:
        key = (r["domain"], r["h"], r["order"], r["nu"])
        if key not in cases:
            cases.append(key)
    for key in cases:
        rs = [r for r in results if (r["domain"], r["h"], r["order"], r["nu"]) == key]
        ndof = next((r["ndof"] for r in rs if "ndof" in r), "?")
        lines += [f"## {key[0]}, h = {key[1]}, P{key[2]}, nu = {key[3]:g} ({ndof} dofs)", "",
                  "| prior | variance (all) | variance (boundary) | corr. error | set-up s | ms/sample | details |",
                  "|---|---|---|---|---|---|---|"]
        lines += [fmt(r) for r in rs] + [""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    ap.add_argument("--out-dir")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds per prior and case")
    ap.add_argument("--one", help=argparse.SUPPRESS)  # internal: run one json-encoded job and print its result
    args = ap.parse_args()
    if args.one:
        job = json.loads(args.one)
        try:
            res = run_one(*job)
        except Exception as err:  # reported, not raised: the series continues
            res = {"domain": job[0], "h": job[1], "order": job[2], "nu": job[3], "prior": job[4],
                   "error": f"{type(err).__name__}: {err}"}
        print("RESULT " + json.dumps(res))
        return
    out = args.out_dir or os.path.join(ROOT, "results", "runs",
                                       f"{socket.gethostname()}_{datetime.now():%Y%m%d-%H%M}_stationary")
    os.makedirs(out, exist_ok=True)
    results = []
    jsonl = os.path.join(out, "results.jsonl")
    for case in PROFILES[args.profile]:
        for prior in PRIORS:
            job = [*case, prior]
            env = {**os.environ, "NGSRNG_QUIET": "1", "PYTHONUNBUFFERED": "1"}
            t = time.perf_counter()
            try:
                proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", json.dumps(job)],
                                      cwd=ROOT, env=env, capture_output=True, text=True, timeout=args.timeout)
                line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
                res = json.loads(line[7:]) if line else {
                    "domain": case[0], "h": case[1], "order": case[2], "nu": case[3], "prior": prior,
                    "error": f"exit {proc.returncode}: {proc.stderr.strip().splitlines()[-1] if proc.stderr else ''}"}
            except subprocess.TimeoutExpired:
                res = {"domain": case[0], "h": case[1], "order": case[2], "nu": case[3], "prior": prior,
                       "error": f"timeout after {args.timeout:.0f} s"}
            res["wall_s"] = time.perf_counter() - t
            results.append(res)
            with open(jsonl, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(f"{case} {prior}: " + (fmt(res) if "variance" in res or "skipped" in res or "error" in res
                                         else str(res)), flush=True)
            report(results, os.path.join(out, "report.md"))
    print(f"report: {os.path.join(out, 'report.md')}")


if __name__ == "__main__":
    main()
