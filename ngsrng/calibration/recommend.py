"""The recommendations of a calibration run, and a short report: plain Python, no pandas or matplotlib (the full
report in ``report.py`` needs them; the calibration itself must not, since ngsrng may be installed as a dependency)."""

import json
import os

from .cases import CPU_SOLVERS, GPU_SOLVERS, accurate, choose

SOLVER_OPTIONS = {**CPU_SOLVERS, **GPU_SOLVERS}


def platform(rec) -> str:
    """Tuning table of a record: cpu-ngsolve, cpu-torch, or <backend>-f64 (the device name is added on install)."""
    if rec["engine"] == "ngsolve":
        return "cpu-ngsolve"
    if rec["device"] == "cpu":
        return "cpu-torch"
    return f"{rec.get('backend', 'cuda')}-{rec['dtype'].replace('float', 'f')}"


def load(run_dir) -> dict:
    """The noise and prior records of a run (the last record of every task)."""
    out = {}
    for suite in ("noise", "prior"):
        path = os.path.join(run_dir, f"{suite}.jsonl")
        if os.path.exists(path):
            with open(path) as fh:
                rows = {}
                for line in fh:
                    rec = json.loads(line)
                    rows[rec["task_id"]] = rec
            out[suite] = list(rows.values())
    return out


def recommendations(run_dir) -> dict:
    """Per platform and calibrated case, the accurate candidate to use (``cases.choose``)."""
    data = load(run_dir)
    env_path = os.path.join(run_dir, "environment.json")
    env = json.load(open(env_path)) if os.path.exists(env_path) else None
    out = {"run": os.path.basename(os.path.abspath(run_dir)), "environment": env, "noise": {}, "prior": {}}
    groups = {}
    for suite in ("noise", "prior"):
        for r in data.get(suite, []):
            if r.get("status") == "ok" and r.get("exact", True) and accurate(r):
                key = (suite, platform(r), r["dim"], r["kind"], r["order"], r.get("power"), r["target"])
                groups.setdefault(key, []).append(r)
    for (suite, plat, d, k, o, power, t), recs in sorted(groups.items(), key=str):
        b = choose(recs)
        if suite == "noise":
            out["noise"].setdefault(plat, {})[f"{d}|{k}|{o}|{int(t)}"] = {
                "method": b["method"], "per_sample_s": float(b["per_sample_s"])}
        else:
            out["prior"].setdefault(plat, {})[f"{d}|{k}|{o}|{power:g}|{int(t)}"] = {
                "solver": b["solver"],
                "method": {**SOLVER_OPTIONS.get(b["solver"], {}), "engine": b["engine"]},
                "noise": b.get("noise"),
                "device": b["device"],
                "dtype": b["dtype"],
                "per_sample_s": float(b["per_sample_s"]),
                "per_sample_batch_s": float(b.get("per_sample_batch_s") or "nan"),
            }
    return out


def _ms(v):
    return "" if v is None or v != v else f"{1e3 * v:.3g} ms"


def short_report(run_dir, recs) -> str:
    """Markdown: the recommended choice per case and platform, and the rejected candidates."""
    lines = [f"# ngsrng calibration: {recs['run']}\n",
             "Recommended choices (times: single sample / per sample in a batch). The full report with every "
             "measurement and plots needs pandas, tabulate and matplotlib: python -m ngsrng.calibration.report "
             f"{run_dir}\n"]
    for suite in ("prior", "noise"):
        for plat, table in recs.get(suite, {}).items():
            fields = "dim kind order power ndof" if suite == "prior" else "dim kind order ndof"
            lines += [f"## {suite}: {plat}\n", f"| case ({fields}) | choice | single | batch |", "|---|---|---|---|"]
            for case, v in table.items():
                choice = v["solver"] if suite == "prior" else v["method"]
                lines.append(f"| {case.replace('|', ' ')} | {choice} | {_ms(v.get('per_sample_s'))} | "
                             f"{_ms(v.get('per_sample_batch_s'))} |")
            lines.append("")
    path = os.path.join(run_dir, "rejected.json")
    if os.path.exists(path):
        with open(path) as fh:
            rejected = json.load(fh)
        lines.append(f"## Rejected candidates ({len(rejected)})\n")
        lines += [f"- `{k}`: {v}" for k, v in sorted(rejected.items())]
    return "\n".join(lines) + "\n"
