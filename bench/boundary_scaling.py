"""Scaling of boundary sampling (``bnd=``): set-up and per-sample cost of restricted and conditioned draws.

    python bench/boundary_scaling.py                      # standard profile, CPU and CUDA if available
    python bench/boundary_scaling.py --profile quick      # small sizes, a few minutes
    python bench/boundary_scaling.py --devices cpu --profile quick

For structured meshes of the unit square (P1, P2) and cube (P1), Gamma = "left", and the precisions
A^power with A = -Laplace + kappa^2 (natural boundary conditions), it measures per (space, size, power, device):

- milliseconds per sample of a plain draw, a restricted draw (``bnd="left"``) and a conditioned draw
  (``bnd_values=g``), each filled in place into a preallocated batch of coefficient vectors;
- the set-up time of each (first call): the plain draw builds the Gaussian (factorisation), which the generator
  caches, so the boundary set-up is the conditioned draw's minus the restricted draw's;
- the conditioning path taken: "sparse" (power 1: the splitting through K_II, no dense matrix), "dense-lifting"
  (small problems: the lifting stored, |F| n_B <= DENSE_LIFTING_VALUES) or "matrix-free" (Sigma_BB from n_B
  applications of Sigma, then one application of Sigma per batch);
- the error of the imposed datum, max |u_B - g| (exact up to rounding), and the peak resident (and device) memory.

Power 1 should cost a few plain draws per conditioned sample at every size; power 2 has an n_B x n_B dense block
and n_B applications of Sigma at set-up, which is the cost the documentation names for fine 3D boundaries.
Every group runs in its own process with a time limit; a group that fails or times out ends its series.
Results: results/runs/<host>_<date>_boundary/ (results.jsonl, report.md).
"""

import argparse
import json
import os
import resource
import socket
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KAPPA2 = 25.0
# cells per direction; P1 2D has (n+1)^2 dofs and n+1 trace dofs, P2 2D (2n+1)^2 and 2n+1, P1 3D (n+1)^3 and (n+1)^2
PROFILES = {
    "quick": {"sizes": {"2D-P1": [32, 64], "2D-P2": [16], "3D-P1": [6, 10]}, "batch": 50, "repeats": 2,
              "timeout": 600},
    # bench/calibrate.py: the sizes at which conditioning cost 7.5x (CPU) and 22x (GPU) a plain draw on 2026-11-02
    "calibrate": {"sizes": {"2D-P1": [128, 512], "2D-P2": [256], "3D-P1": [16, 32]}, "batch": 100, "repeats": 2,
                  "timeout": 900},
    "standard": {"sizes": {"2D-P1": [64, 128, 256, 512], "2D-P2": [32, 64, 128, 256], "3D-P1": [8, 16, 24, 32]},
                 "batch": 200, "repeats": 3, "timeout": 1800},
    "full": {"sizes": {"2D-P1": [64, 128, 256, 512, 1024], "2D-P2": [32, 64, 128, 256, 512],
                       "3D-P1": [8, 16, 24, 32, 48, 64]}, "batch": 200, "repeats": 3, "timeout": 3600},
}
POWERS = (1, 2)
BUFFER_BYTES = 2e9


# ------------------------------------------------------------------ worker


def run_case(case):
    import ngsolve as ngs
    import numpy as np
    import torch
    from ngsolve.meshes import MakeStructured2DMesh, MakeStructured3DMesh

    sys.path.insert(0, ROOT)
    import ngsrng
    from ngsrng.boundary import boundary_dofs

    dim, order = (2, 1) if case["space"] == "2D-P1" else (2, 2) if case["space"] == "2D-P2" else (3, 1)
    n = case["n"]
    mesh = (MakeStructured2DMesh(quads=False, nx=n, ny=n) if dim == 2
            else MakeStructured3DMesh(hexes=False, nx=n, ny=n, nz=n))
    fes = ngs.H1(mesh, order=order)
    form = lambda u, v: (ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) + KAPPA2 * u * v) * ngs.dx  # noqa: E731
    dev = torch.device(case["device"])
    dtype = torch.float64
    rng = ngsrng.default_rng(fes, seed=0, precision=ngsrng.PDEPrecision(form, power=case["power"]), device=dev,
                             dtype="float64", method={"engine": "torch"} if dev.type == "cuda" else None)
    dofs = boundary_dofs(fes, "left")
    g = np.sin(np.arange(dofs.size, dtype=float))
    rec = {"ndof": fes.ndof, "n_bnd": int(dofs.size)}
    batch = max(1, min(case["batch"], int(BUFFER_BYTES // (8 * fes.ndof))))
    rec["batch"] = batch

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    modes = (("plain", {}), ("restrict", {"bnd": "left"}), ("condition", {"bnd": "left", "bnd_values": g}))
    for mode, kw in modes:
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sync()
        t0 = time.perf_counter()
        d = rng.distribution("normal", **kw)  # no representation=: that would be the pointwise-standardised copula
        d.sample(2, output="torch")
        sync()
        rec[f"{mode}_setup_s"] = time.perf_counter() - t0
        buf = torch.empty((batch, fes.ndof), device=dev, dtype=dtype)
        times = []
        for _ in range(case["repeats"]):
            sync()
            t0 = time.perf_counter()
            d.sample(out=buf)
            sync()
            times.append(time.perf_counter() - t0)
        rec[f"{mode}_ms"] = 1e3 * min(times) / batch
        if dev.type == "cuda":
            rec[f"{mode}_gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        if mode == "restrict":
            outside = torch.as_tensor(np.setdiff1d(np.arange(fes.ndof), dofs), device=dev)
            rec["restrict_outside_max"] = float(buf[:, outside].abs().max())
        if mode == "condition":
            rec["datum_err"] = float(np.abs(buf[:, torch.as_tensor(dofs, device=dev)].cpu().numpy() - g).max())
            s = d.field.bnd["sampler"]  # path taken by the correction (private attributes: a benchmark probe)
            rec["path"] = ("sparse" if s._sparse_form is not None else
                           "dense-lifting" if s._dense is not None else "matrix-free")
        del d, buf
    rec["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return rec


# ------------------------------------------------------------------ driver


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="standard", choices=list(PROFILES))
    ap.add_argument("--devices", nargs="+", default=None, help="subset of: cpu cuda (default: all available)")
    ap.add_argument("--powers", type=lambda s: [int(v) for v in s.split(",")], default=list(POWERS))
    ap.add_argument("--out-dir", default=None, help="run directory (default: results/runs/<host>_<date>_boundary)")
    ap.add_argument("--report", default=None, help="only rebuild report.md of this run directory")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        rec = run_case(json.loads(args.worker))
        print("RESULT " + json.dumps(rec), flush=True)
        return
    if args.report:
        report(args.report)
        return
    import torch

    prof = PROFILES[args.profile]
    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = args.out_dir or os.path.join(ROOT, "results", "runs", f"{socket.gethostname()}_{stamp}_boundary")
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "results.jsonl")
    done = set()
    if os.path.exists(out_path):  # resume: skip finished groups
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["space"], r["n"], r["power"], r["device"]))
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump({"profile": args.profile, "devices": devices, "powers": args.powers, **prof}, fh, indent=1)
    for space, sizes in prof["sizes"].items():
        for power in args.powers:
            for device in devices:
                for n in sizes:
                    key = (space, n, power, device)
                    if key in done:
                        continue
                    case = {"space": space, "n": n, "power": power, "device": device, "batch": prof["batch"],
                            "repeats": prof["repeats"]}
                    rec = {**case, "time": datetime.now().isoformat(timespec="seconds")}
                    t0 = time.perf_counter()
                    try:
                        proc = subprocess.run([sys.executable, __file__, "--worker", json.dumps(case)],
                                              capture_output=True, text=True, timeout=prof["timeout"])
                        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
                        if proc.returncode == 0 and lines:
                            rec.update(json.loads(lines[-1][7:]), status="ok")
                        else:
                            err = (proc.stderr.strip().splitlines() or ["no output"])[-1]
                            rec.update(status="memory" if "MemoryError" in proc.stderr else "error",
                                       returncode=proc.returncode, error=err[:300])
                    except subprocess.TimeoutExpired:
                        rec.update(status="timeout")
                    rec["wall_s"] = time.perf_counter() - t0
                    with open(out_path, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    msg = (f"plain {rec['plain_ms']:.3f}, restrict {rec['restrict_ms']:.3f}, condition "
                           f"{rec['condition_ms']:.3f} ms/sample [{rec['path']}], err {rec['datum_err']:.1e}"
                           if rec["status"] == "ok" else f"{rec['status']}: {rec.get('error', '')}")
                    print(f"[{datetime.now():%H:%M:%S}] {space} n={n} power={power} {device}: {msg}", flush=True)
                    if rec["status"] != "ok":
                        break  # larger sizes of this series would fail too
    report(run_dir)


def report(run_dir):
    recs = [json.loads(line) for line in open(os.path.join(run_dir, "results.jsonl"))]
    head = ("| space | n | ndof | n_B | power | device | path | plain ms | restrict ms | condition ms | "
            "condition / plain | Gaussian set-up s | boundary set-up s | datum error | peak RSS GB | GPU peak GB |")
    lines = [f"# Boundary sampling scaling: {os.path.basename(os.path.abspath(run_dir))}\n",
             "Milliseconds per sample, filled in place. Gaussian set-up = first plain draw (factorisation, cached by "
             "the generator); boundary set-up = first conditioned draw minus first restricted draw. Power 1 takes the "
             "sparse path at every size.\n", head,
             "|" + "---|" * (head.count("|") - 1)]
    for r in sorted(recs, key=lambda r: (r["space"], r["power"], r["device"], r["n"])):
        if r["status"] != "ok":
            lines.append(f"| {r['space']} | {r['n']} | | | {r['power']} | {r['device']} | {r['status']} |"
                         + " |" * 9)
            continue
        gpu = max((r.get(f"{m}_gpu_peak_gb", 0.0) for m in ("plain", "restrict", "condition")), default=0.0)
        lines.append(
            f"| {r['space']} | {r['n']} | {r['ndof']} | {r['n_bnd']} | {r['power']} | {r['device']} | {r['path']} | "
            f"{r['plain_ms']:.3f} | {r['restrict_ms']:.3f} | {r['condition_ms']:.3f} | "
            f"{r['condition_ms'] / r['plain_ms']:.2f} | {r['plain_setup_s']:.2f} | "
            f"{r['condition_setup_s'] - r['restrict_setup_s']:.2f} | "
            f"{r['datum_err']:.1e} | {r['peak_rss_gb']:.2f} | {gpu:.2f} |")
    path = os.path.join(run_dir, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"report: {path}")


if __name__ == "__main__":
    main()
