"""Host synchronisations in the torch solvers: do fewer of them pay off? (Benchmark only: the variants live here.)

    python bench/solver_syncs.py                      # GPU: 2D and 3D P1 at 1e4 and 1e5 dofs
    python bench/solver_syncs.py --quick              # 2D P1 at 1e4 dofs
    python bench/solver_syncs.py --targets 1e5,1e6    # other sizes

The torch engine's iterative solvers (Jacobi PCG, AMG-preconditioned PCG -- the solver of AMD GPUs --, and the
multi-shift PCG of apply-only precisions) test convergence after every iteration, ``if res <= tol``: a host
synchronisation that empties the GPU's queue, so the next iteration's kernel launches cannot overlap the current
iteration's work. Where an iteration is cheap (small and medium problems, the many small kernels of an AMG V-cycle),
launch latency may dominate. Variants, against the production solver (``every``):

- ``every-4``, ``every-16``: test convergence every k iterations (up to k - 1 extra iterations, which only tighten
  the result);
- ``lagged``: after enqueuing iteration i, read iteration i-1's residual from pinned memory (an event, not a stream
  synchronisation: the GPU keeps working meanwhile; one extra iteration);
- ``graph-8``: 8 iterations captured once as a CUDA graph and replayed (no launch overhead; a test every 8
  iterations). ``n/a`` where capture fails (e.g. a sparse kernel that cannot be captured).

Also the type-G fields' gamma sampler (Marsaglia-Tsang rejection, ``while todo.any()``: two synchronisations per
round) against drawing 3 proposals per value at once and looping only over the rare leftovers; its law is checked
(KS test against scipy).

For every case: median time per solve (interleaved repetitions), iterations, relative residual, difference from the
production solution, ratio and verdict (gain / loss / tie within 10%). Results go to
results/runs/<host>_<date>_syncs/ (results.jsonl, report.md). AMG cases need pyamg. Most informative on AMD GPUs,
where AMG-PCG is the sampler's solver; on NVIDIA GPUs cuDSS is, and these solvers serve mass solves, apply-only
precisions and the fallback.
"""

import argparse
import json
import os
import socket
import statistics
import sys
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import scipy.stats as st
import torch  # before ngsolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ngsrng  # noqa: E402
from ngsrng.blackbox import MultiShiftPCG  # noqa: E402
from ngsrng.calibration.common import make_space  # noqa: E402
from ngsrng.distributions.mixtures import torch_gamma  # noqa: E402
from ngsrng.solvers import TorchAMGPCG, TorchPCG, pyamg_available  # noqa: E402
from ngsrng.space import SpaceInfo  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIE = 1.10
F64 = torch.float64


def sync():
    torch.cuda.synchronize()


# ------------------------------------------------------------------ PCG variants (same algorithm as ngsrng.solvers)


def _init(prec, b):
    x = torch.zeros_like(b)
    r = b.clone()
    z = prec(r)
    return x, r, z.clone(), (r * z).sum(0), torch.linalg.vector_norm(b, dim=0).clamp_min(torch.finfo(b.dtype).tiny)


def _step(a, prec, x, r, p, rz, bnorm):
    """One PCG iteration in place (x, r, p, rz); returns the relative residual (a device scalar)."""
    ap = a @ p
    pap = (p * ap).sum(0)
    alpha = torch.where(pap > 0, rz / pap, torch.zeros_like(rz))
    x.add_(alpha * p)
    r.sub_(alpha * ap)
    res = (torch.linalg.vector_norm(r, dim=0) / bnorm).max()
    z = prec(r)
    rz_new = (r * z).sum(0)
    p.copy_(z + torch.where(rz > 0, rz_new / rz, torch.zeros_like(rz)) * p)
    rz.copy_(rz_new)
    return res


def pcg_every(a, prec, b, tol, maxiter, k):
    x, r, p, rz, bnorm = _init(prec, b)
    it = 0
    for it in range(1, maxiter + 1):
        res = _step(a, prec, x, r, p, rz, bnorm)
        if it % k == 0 and res <= tol:
            break
    return x, it


def pcg_lagged(a, prec, b, tol, maxiter):
    x, r, p, rz, bnorm = _init(prec, b)
    bufs = [torch.empty((), dtype=b.dtype, pin_memory=True) for _ in range(2)]
    events = [torch.cuda.Event(), torch.cuda.Event()]
    it = 0
    for it in range(1, maxiter + 1):
        res = _step(a, prec, x, r, p, rz, bnorm)
        bufs[it % 2].copy_(res, non_blocking=True)
        events[it % 2].record()
        if it > 1:  # iteration it is queued: wait only for it - 1's residual
            events[(it - 1) % 2].synchronize()
            if bufs[(it - 1) % 2].item() <= tol:
                break
    return x, it


class GraphPCG:
    """k PCG iterations captured once (per right-hand-side shape) as a CUDA graph, replayed until convergence."""

    def __init__(self, a, prec, shape, dtype, device, k=8):
        self.a, self.prec, self.k = a, prec, k
        self.b = torch.zeros(shape, dtype=dtype, device=device)
        self.x, self.r, self.p = (torch.zeros(shape, dtype=dtype, device=device) for _ in range(3))
        self.rz = torch.zeros(shape[1], dtype=dtype, device=device)
        self.bnorm = torch.ones(shape[1], dtype=dtype, device=device)
        self.res = torch.zeros((), dtype=dtype, device=device)
        self.b.normal_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up outside the capture (workspaces, kernels)
            self._reset()
            for _ in range(3):
                self._iterate()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for _ in range(k):
                self._iterate()

    def _reset(self):
        x, r, p, rz, bnorm = _init(self.prec, self.b)
        for dst, src in ((self.x, x), (self.r, r), (self.p, p), (self.rz, rz), (self.bnorm, bnorm)):
            dst.copy_(src)

    def _iterate(self):
        self.res.copy_(_step(self.a, self.prec, self.x, self.r, self.p, self.rz, self.bnorm))

    def solve(self, b, tol, maxiter):
        self.b.copy_(b)
        self._reset()
        it = 0
        while it < maxiter:
            self.graph.replay()
            it += self.k
            if self.res.item() <= tol:
                break
        return self.x.clone(), it


class MultiShiftEveryK(MultiShiftPCG):
    """The production multi-shift PCG with its convergence test every k iterations (the loop is ngsrng's, with that
    one change)."""

    def __init__(self, *args, k=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def solve(self, b):
        b = b[:, None] if b.ndim == 1 else b
        n, k = b.shape
        s = torch.as_tensor(self.shifts, dtype=b.dtype, device=b.device)[:, None]
        w = torch.as_tensor(self.weights, dtype=b.dtype, device=b.device)
        x = torch.zeros((len(self.shifts), n, k), dtype=b.dtype, device=b.device)
        r = b.clone()
        z = self.apply_minv(r)
        p_seed = z.clone()
        ps = z.unsqueeze(0).repeat(len(self.shifts), 1, 1)
        rz = (r * z).sum(0)
        bnorm = torch.sqrt(rz).clamp_min(torch.finfo(b.dtype).tiny)
        zeta_old = torch.ones((len(self.shifts), k), dtype=b.dtype, device=b.device)
        zeta = torch.ones_like(zeta_old)
        alpha_old = torch.ones(k, dtype=b.dtype, device=b.device)
        beta_old = torch.zeros(k, dtype=b.dtype, device=b.device)
        active = torch.ones((len(self.shifts), k), dtype=torch.bool, device=b.device)
        it = 0
        for it in range(1, self.maxiter + 1):
            qp = self.apply_q(p_seed)
            pqp = (p_seed * qp).sum(0)
            alpha = torch.where(pqp > 0, rz / pqp, torch.zeros_like(rz))
            denom = alpha * beta_old * (zeta_old - zeta) + zeta_old * alpha_old * (1 + s * alpha)
            zeta_new = torch.where(active, zeta * zeta_old * alpha_old / denom, zeta)
            ratio = torch.where(active, zeta_new / zeta, torch.zeros_like(zeta))
            x += (alpha * ratio).unsqueeze(1) * ps
            r = r - alpha * qp
            z = self.apply_minv(r)
            rz_new = (r * z).sum(0)
            beta = torch.where(rz > 0, rz_new / rz, torch.zeros_like(rz))
            ps = torch.where(active.unsqueeze(1),
                             zeta_new.unsqueeze(1) * z.unsqueeze(0) + (beta * ratio**2).unsqueeze(1) * ps, ps)
            p_seed = z + beta * p_seed
            shifted_res = torch.sqrt(rz_new.clamp_min(0)) * zeta_new.abs() / bnorm
            active = active & (shifted_res > 1e-3 * self.tol)
            zeta_old, zeta, alpha_old, beta_old, rz = zeta, zeta_new, alpha, beta, rz_new
            if it % self.k == 0 and shifted_res.max() <= self.tol:
                break
        self.iterations.append(it)
        return torch.einsum("s,snk->nk", w, x)


# ------------------------------------------------------------------ gamma sampler variant


def torch_gamma_proposals(shape, gen, proposals=3):
    """Gamma(shape, 1) like ngsrng's torch_gamma (Marsaglia & Tsang, U^(1/a) boost for a < 1), but ``proposals``
    candidates per value at once: the first accepted one is kept, and the production loop serves the rare values
    with none (one synchronisation instead of two per round)."""
    boost = shape < 1
    a = torch.where(boost, shape + 1, shape)
    d = a - 1.0 / 3.0
    c = 1.0 / torch.sqrt(9.0 * d)
    x = torch.randn((proposals, *a.shape), generator=gen, device=a.device, dtype=a.dtype)
    u = torch.rand((proposals, *a.shape), generator=gen, device=a.device, dtype=a.dtype)
    v = (1 + c * x) ** 3
    ok = (v > 0) & (torch.log(u) < 0.5 * x * x + d - d * v + d * torch.log(v.clamp_min(1e-300)))
    first = ok.to(torch.uint8).argmax(0)
    out = d * torch.take_along_dim(v, first[None], 0)[0]
    left = ~ok.any(0)
    if left.any():
        idx = left.nonzero(as_tuple=True)
        out[idx] = torch_gamma(a[idx], gen)  # a >= 1 there: no boost inside
    u2 = torch.rand(shape.shape, generator=gen, device=a.device, dtype=a.dtype)
    return torch.where(boost, out * u2 ** (1.0 / shape), out)


# ------------------------------------------------------------------ measurements


def interleaved(fns, reps):
    """Median time of each function, called in turn (one warm-up call each)."""
    for f in fns.values():
        f()
    sync()
    times = {k: [] for k in fns}
    for _ in range(reps):
        for k, f in fns.items():
            t0 = time.perf_counter()
            f()
            sync()
            times[k].append(time.perf_counter() - t0)
    return {k: statistics.median(v) for k, v in times.items()}


def solver_cases(fes, args, dev):
    """(label, production solver apply, matrix, preconditioner, tol) per solver available here."""
    s = SpaceInfo(fes)
    a_free = s.assemble_free(ngsrng.matern_correlation(0.2, nu=1).bind(fes).form)
    tol = 1e-10
    out = []
    jac = TorchPCG(a_free, dev, F64, tol=tol)
    out.append(("Jacobi PCG", jac, jac.a, lambda r, s_=jac: s_.dinv[:, None] * r, tol))
    if pyamg_available():
        amg = TorchAMGPCG(a_free, dev, F64, tol=tol)
        out.append(("AMG-PCG", amg, amg.a, lambda r, s_=amg: s_._vcycle(0, r), tol))
    mass = TorchPCG(s.mass_free, dev, F64, tol=1e-13)
    out.append(("mass (Jacobi PCG)", mass, mass.a, lambda r, s_=mass: s_.dinv[:, None] * r, 1e-13))
    return out, a_free, s


def run_solvers(fes, args, dev, fh, recs):
    cases, a_free, s = solver_cases(fes, args, dev)
    for label, solver, a, prec, tol in cases:
        for width in args.widths:
            b = torch.randn(a.shape[0], width, dtype=F64, device=dev)
            ref = solver.apply(b)
            variants = {"every": lambda: (solver.apply(b), solver.iterations[-1])}
            for k in (4, 16):
                variants[f"every-{k}"] = lambda k=k: pcg_every(a, prec, b, tol, solver.maxiter, k)
            variants["lagged"] = lambda: pcg_lagged(a, prec, b, tol, solver.maxiter)
            try:
                g = GraphPCG(a, prec, tuple(b.shape), F64, dev, k=8)
                variants["graph-8"] = lambda: g.solve(b, tol, solver.maxiter)
                graph_error = None
            except Exception as err:  # noqa: BLE001 - capture is the experiment
                graph_error = f"{type(err).__name__}: {err}"[:200]
            times = interleaved({k: (lambda f=f: f()) for k, f in variants.items()}, args.reps)
            for name, f in variants.items():
                x, its = f()
                resid = float((torch.linalg.vector_norm(b - a @ x, dim=0) / torch.linalg.vector_norm(b, dim=0)).max())
                rec = {"case": label, "variant": name, "ndof": s.ndof, "width": width, "seconds": times[name],
                       "iterations": int(its), "rel_residual": resid,
                       "diff_vs_every": float((x - ref).abs().max() / ref.abs().max())}
                _finish(rec, times["every"], fh, recs)
            if graph_error:
                _finish({"case": label, "variant": "graph-8", "ndof": s.ndof, "width": width, "error": graph_error},
                        None, fh, recs)
    # multi-shift PCG (apply-only precisions): 12 shifts over the spectrum's range, M^-1 by Jacobi PCG
    mass = TorchPCG(s.mass_free, dev, F64, tol=1e-13)
    q = cases[0][2]
    shifts, weights = np.geomspace(1e-2, 1e4, 12), np.ones(12)
    for width in args.widths:
        b = torch.randn(q.shape[0], width, dtype=F64, device=dev)
        base = MultiShiftPCG(lambda v: q @ v, mass.apply, shifts, weights, tol=1e-8)
        variants = {"every": base}
        variants.update({f"every-{k}": MultiShiftEveryK(lambda v: q @ v, mass.apply, shifts, weights, tol=1e-8, k=k)
                         for k in (4, 16)})
        ref = base.solve(b)
        times = interleaved({k: (lambda m=m: m.solve(b)) for k, m in variants.items()}, args.reps)
        for name, m in variants.items():
            x = m.solve(b)
            rec = {"case": "multi-shift PCG (12 shifts)", "variant": name, "ndof": s.ndof, "width": width,
                   "seconds": times[name], "iterations": int(m.iterations[-1]),
                   "diff_vs_every": float((x - ref).abs().max() / ref.abs().max())}
            _finish(rec, times["every"], fh, recs)


def run_gamma(dev, args, fh, recs):
    gen = torch.Generator(device=dev).manual_seed(0)
    for n in args.gamma_sizes:
        for shape_value in (0.05, 0.5, 2.0):
            shape = torch.full((int(n),), shape_value, dtype=F64, device=dev)
            fns = {"production": lambda: torch_gamma(shape, gen),
                   "3 proposals": lambda: torch_gamma_proposals(shape, gen)}
            times = interleaved(fns, args.reps)
            for name, f in fns.items():
                x = f().cpu().numpy()
                rec = {"case": f"gamma sampler (shape {shape_value})", "variant": name, "ndof": int(n), "width": 1,
                       "seconds": times[name],
                       "ks_pvalue": float(st.kstest(x[:20000], st.gamma(shape_value).cdf).pvalue)}
                _finish(rec, times["production"], fh, recs)


def _finish(rec, baseline, fh, recs):
    if baseline is not None and "seconds" in rec:
        rec["ratio"] = baseline / rec["seconds"]  # > 1: faster than the production code
        rec["verdict"] = "gain" if rec["ratio"] > TIE else ("loss" if rec["ratio"] < 1 / TIE else "tie")
    recs.append(rec)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    if "error" in rec:
        print(f"  {rec['case']:<30} {rec['variant']:<12} w={rec['width']:<3} n/a: {rec['error']}", flush=True)
        return
    if "iterations" in rec:
        extra = f"its {rec['iterations']:5d} " + (f"resid {rec['rel_residual']:.1e} " if "rel_residual" in rec else "")
    else:
        extra = f"KS p {rec['ks_pvalue']:.2f} "
    print(f"  {rec['case']:<30} {rec['variant']:<12} w={rec['width']:<3} {rec['seconds'] * 1e3:9.2f} ms  {extra}"
          f"{rec.get('verdict', ''):<4} x{rec.get('ratio', 1):.2f}", flush=True)


def write_report(out_dir, recs):
    lines = ["# Host synchronisations in the torch solvers: variants against the production code\n",
             "ratio = production time / variant time (> 1: the variant is faster); a tie is within "
             f"{TIE - 1:.0%}. diff: largest difference from the production solution, relative to its largest entry.\n",
             "| case | ndof | width | variant | time | iterations | residual / KS p | diff | ratio | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        if "error" in r:
            lines.append(f"| {r['case']} | {r['ndof']} | {r['width']} | {r['variant']} | n/a: {r['error']} "
                         "| | | | | |")
            continue
        acc = f"{r['rel_residual']:.1e}" if "rel_residual" in r else (f"{r['ks_pvalue']:.2f}" if "ks_pvalue" in r
                                                                        else "")
        lines.append(f"| {r['case']} | {r['ndof']} | {r['width']} | {r['variant']} | {r['seconds'] * 1e3:.2f} ms "
                     f"| {r.get('iterations', '')} | {acc} | {r.get('diff_vs_every', float('nan')):.1e} "
                     f"| {r.get('ratio', 1):.2f} | {r.get('verdict', '')} |")
    gains = [r for r in recs if r.get("verdict") == "gain"]
    verdict = (f"{len(gains)} gain(s): " + "; ".join(f"{r['variant']} for {r['case']} at {r['ndof']} (w={r['width']})"
                                                     for r in gains)
               if gains else "No gains: the production solvers' per-iteration test costs nothing measurable here.")
    lines.append("\n" + verdict)
    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--targets", default=None, help="dof targets, e.g. 1e4,1e5")
    ap.add_argument("--dims", default=None, help="e.g. 2,3")
    ap.add_argument("--widths", default="1,32", help="right-hand sides per solve")
    ap.add_argument("--gamma-sizes", default=None, help="values per gamma draw, e.g. 1e5,1e7")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not ngsrng.backend.cuda_usable():
        sys.exit("solver_syncs needs a usable GPU")
    args.widths = [int(w) for w in args.widths.split(",")]
    targets = [float(t) for t in (args.targets or ("1e4" if args.quick else "1e4,1e5")).split(",")]
    dims = [int(d) for d in (args.dims or ("2" if args.quick else "2,3")).split(",")]
    args.gamma_sizes = [float(n) for n in (args.gamma_sizes or ("1e5" if args.quick else "1e5,1e7")).split(",")]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or os.path.join(ROOT, "results", "runs", f"{socket.gethostname()}_{stamp}_syncs")
    os.makedirs(out_dir, exist_ok=True)
    dev = ngsrng.backend.resolve_device("cuda")
    warnings.simplefilter("ignore")
    recs = []
    with open(os.path.join(out_dir, "results.jsonl"), "w") as fh:
        for dim in dims:
            for target in targets:
                fes = make_space(dim, "H1", 1, target)
                print(f"== {dim}D H1 P1: {fes.ndof} dofs ({ngsrng.tuning.gpu_backend()}, "
                      f"{torch.cuda.get_device_name(dev)})", flush=True)
                try:
                    run_solvers(fes, args, dev, fh, recs)
                except Exception as err:  # noqa: BLE001
                    print(f"  solvers failed: {type(err).__name__}: {err}\n{traceback.format_exc()[-800:]}")
        print("== gamma sampler", flush=True)
        run_gamma(dev, args, fh, recs)
    write_report(out_dir, recs)
    print(f"report: {os.path.join(out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
