"""Calibrate a machine for ngsrng in one command: tests, validation, solver/noise calibration, with logs and a summary.

    python bench/calibrate.py                     # tests, validation, calibration (~30 min on a GPU box)
    python bench/calibrate.py --quick             # a few minutes: core tests, quick validation, smoke calibration
    python bench/calibrate.py --large             # calibration up to 1e6 dofs (large CPU/GPU boxes, an hour or more)
    python bench/calibrate.py --boundary --throughput    # also the boundary-sampling and GPU-throughput benchmarks
    python bench/calibrate.py --resume results/calibrate/<run>   # rerun only the steps that did not finish

Steps, in order:

  env          environment snapshot: CPU, GPU, driver, CUDA, torch, cuDSS, NGSolve, report dependencies, git state
  tests        the pytest suite, CUDA variants included when a device is usable (junit.xml)
  calibrate    bench/run_all.py --install: measures the white-noise samplers and solvers that can win on this machine,
               rejects the inaccurate, failing and far slower ones, and installs the winners as the active tuning
               (python -m ngsrng.tuning show)
  boundary     (--boundary) bench/boundary_scaling.py: restricted and conditioned boundary draws
  throughput   (--throughput, GPU) bench/gpu_batch.py --quick: device checks and in-place batch throughput

Each step runs as a subprocess with a time limit (enforced also while it prints nothing). Its output goes to the
console and to logs/<step>.log, its status and duration to manifest.json, so an interrupted or failed run is continued
with --resume (the calibration resumes inside the step). SUMMARY.md lists every step with its log and report. The exit
code is nonzero if a step failed.
"""

import argparse
import json
import os
import platform
import queue
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
MIN = 60

ENV_SCRIPT = r"""
import importlib, json, os, platform, subprocess, sys
sys.path.insert(0, os.getcwd())
out = {"python": platform.python_version(), "executable": sys.executable, "platform": platform.platform(),
       "cpu_count": os.cpu_count()}
import torch, numpy, scipy, ngsolve
out.update(torch=torch.__version__, torch_cuda=torch.version.cuda, numpy=numpy.__version__, scipy=scipy.__version__,
           ngsolve=ngsolve.__version__, cuda_available=torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    out.update(gpu=torch.cuda.get_device_name(), gpu_total_gb=total / 1e9, gpu_free_gb=free / 1e9,
               capability=".".join(map(str, torch.cuda.get_device_capability())))
import ngsrng
from ngsrng.backend import cuda_usable
from ngsrng.solvers import cudss_available
out.update(cuda_usable=bool(cuda_usable()), cudss=bool(cudss_available()))
for mod in ("pandas", "tabulate", "matplotlib"):  # needed by the reports
    try:
        out[mod] = getattr(importlib.import_module(mod), "__version__", "installed")
    except Exception:
        out[mod] = None
try:
    out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    out["nvidia_smi"] = None
print(json.dumps(out, indent=1))
"""


def steps(cfg, run):
    """(name, command, timeout in seconds, report path relative to the run directory) for the chosen options."""
    out = os.path.join(run, "out")
    cal_dir = os.path.join(out, "calibrate")
    profile = "smoke" if cfg["quick"] else "large" if cfg["large"] else "calibrate"
    calibrate = [PY, "bench/run_all.py", "--profile", profile, "--install"]
    calibrate += (["--resume", cal_dir] if os.path.exists(os.path.join(cal_dir, "config.json"))
                  else ["--out", cal_dir])
    tests = ["tests/test_gaussian.py", "tests/test_boundary.py", "tests/test_inplace.py"] if cfg["quick"] \
        else ["tests"]
    plan = [
        ("env", [PY, "-c", ENV_SCRIPT], 10 * MIN, None),
        ("tests", [PY, "-m", "pytest", "-q", "-rsfE", "-p", "no:cacheprovider", *tests,
                   f"--junitxml={os.path.join(out, 'junit.xml')}"], (20 if cfg["quick"] else 60) * MIN,
         "out/junit.xml"),
        ("calibrate", calibrate, (30 if cfg["quick"] else 360 if cfg["large"] else 120) * MIN,
         "out/calibrate/report.md"),
    ]
    if cfg["boundary"]:
        plan.append(("boundary", [PY, "bench/boundary_scaling.py", "--profile", "quick" if cfg["quick"] else "standard",
                                  "--out-dir", os.path.join(out, "boundary")], 60 * MIN, "out/boundary/report.md"))
    if cfg["throughput"]:
        plan.append(("throughput", [PY, "bench/gpu_batch.py", "--quick", "--out-dir", os.path.join(out, "throughput")],
                     60 * MIN, "out/throughput/report.md"))
    return plan


def stream(cmd, log, timeout):
    """Run cmd from the repository root, echoing its output to the console and the step log; return the exit code,
    or "timeout" (checked every second, also while the command prints nothing)."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    lines = queue.Queue()

    def pump():
        for item in proc.stdout:
            lines.put(item)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    t0 = time.time()
    while True:
        try:
            line = lines.get(timeout=1.0)
        except queue.Empty:
            line = ""
        if line is None:
            return proc.wait()
        if line:
            log(line.rstrip("\n"))
        if time.time() - t0 > timeout:
            proc.kill()
            proc.wait()
            log(f"[calibrate] timeout after {timeout} s")
            return "timeout"


def git_state():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "--short", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len(git("status", "--porcelain").splitlines())}


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def summary(run, manifest):
    cfg = manifest["options"]
    opts = ", ".join(k for k, v in cfg.items() if v) or "default"
    lines = [f"# Calibration run `{os.path.basename(run)}`\n",
             f"- host: {manifest['host']}, started {manifest['started']}, git {manifest['git']}",
             f"- options: {opts}\n",
             "| step | status | minutes | log | report |", "|---|---|---|---|---|"]
    for name, rec in manifest["steps"].items():
        rep = rec.get("report")
        rep = f"[{rep}]({rep})" if rep and os.path.exists(os.path.join(run, rep)) else ""
        minutes = rec.get("seconds", 0) / 60
        lines.append(f"| {name} | {rec['status']} | {minutes:.1f} | [log](logs/{name}.log) | {rep} |")
    env = manifest.get("env")
    if env:
        lines.append("\n## Environment\n\n```\n" + json.dumps(env, indent=1) + "\n```")
    for name, pattern in (("tests", ("passed", "failed", "error")), ("validate", ("ALL CHECKS", "FAILED:")),
                          ("calibrate", ("finished in", "installed tuning")), ("throughput", ("checks:",))):
        path = os.path.join(run, "logs", f"{name}.log")
        if os.path.exists(path):
            hits = [ln for ln in open(path) if any(p in ln for p in pattern)][-3:]
            if hits:
                lines.append(f"\n## {name}\n\n```\n" + "".join(hits).rstrip() + "\n```")
    rejected = os.path.join(run, "out", "calibrate", "rejected.json")
    if os.path.exists(rejected):
        with open(rejected) as fh:
            rej = json.load(fh)
        lines.append(f"\n## Rejected candidates ({len(rej)})\n")
        lines += [f"- `{k}`: {v}" for k, v in sorted(rej.items())]
    with open(os.path.join(run, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="core tests, quick validation, smoke calibration")
    ap.add_argument("--large", action="store_true", help="calibrate up to 1e6 dofs")
    ap.add_argument("--boundary", action="store_true", help="also run bench/boundary_scaling.py")
    ap.add_argument("--throughput", action="store_true", help="also run bench/gpu_batch.py --quick (GPU)")
    ap.add_argument("--steps", default=None, help="comma-separated subset of the steps")
    ap.add_argument("--skip", default="", help="comma-separated steps to leave out")
    ap.add_argument("--resume", default=None, help="run directory to continue: steps with status ok are skipped")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    args = ap.parse_args()

    if args.resume:
        run = os.path.abspath(args.resume)
        manifest = json.load(open(os.path.join(run, "manifest.json")))
    else:
        cfg = {k: getattr(args, k) for k in ("quick", "large", "boundary", "throughput")}
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        tag = f"{socket.gethostname()}_{stamp}{'_quick' if args.quick else '_large' if args.large else ''}"
        run = os.path.join(ROOT, "results", "calibrate", tag)
        manifest = {"host": socket.gethostname(), "started": datetime.now().isoformat(timespec="seconds"),
                    "options": cfg, "python": platform.python_version(), "git": git_state(), "steps": {}}
    plan = steps(manifest["options"], run)
    names = [n for n, *_ in plan]
    if args.list:
        print("\n".join(names))
        return
    chosen = args.steps.split(",") if args.steps else names
    skip = set(filter(None, args.skip.split(",")))
    unknown = (set(chosen) | skip) - set(names)
    if unknown:
        sys.exit(f"unknown steps {sorted(unknown)}; available: {names}")
    chosen = [n for n in chosen if n not in skip]
    os.makedirs(os.path.join(run, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run, "out"), exist_ok=True)
    mpath = os.path.join(run, "manifest.json")
    print(f"[calibrate] run directory: {run}\n[calibrate] steps: {', '.join(chosen)}", flush=True)
    if manifest["git"]["dirty_files"]:
        print(f"[calibrate] note: the working tree has {manifest['git']['dirty_files']} modified files", flush=True)

    for name, cmd, timeout, report in plan:
        if name not in chosen:
            continue
        if manifest["steps"].get(name, {}).get("status") == "ok":
            print(f"[calibrate] {name}: already ok, skipped", flush=True)
            continue
        captured = []
        with open(os.path.join(run, "logs", f"{name}.log"), "a") as fh:
            def log(line, fh=fh, name=name):
                print(f"[{name}] {line}", flush=True)
                fh.write(line + "\n")
                fh.flush()
                captured.append(line)

            log(f"[calibrate] {datetime.now():%Y-%m-%d %H:%M:%S} start: " + (name if name == "env" else " ".join(cmd)))
            t0 = time.time()
            rc = stream(cmd, log, timeout)
        status = "ok" if rc == 0 else "timeout" if rc == "timeout" else "failed"
        manifest["steps"][name] = {"status": status, "exit": rc, "seconds": time.time() - t0,
                                   "finished": datetime.now().isoformat(timespec="seconds"), "report": report}
        if name == "env" and rc == 0:
            text = "\n".join(captured)
            try:
                manifest["env"] = json.loads(text[text.index("{"):text.rindex("}") + 1])
            except ValueError:
                pass
            env = manifest.get("env", {})
            if not env.get("cuda_usable"):
                print("[calibrate] no usable CUDA device: the CPU engine is calibrated", flush=True)
            if not all(env.get(m) for m in ("pandas", "tabulate", "matplotlib")):
                save(mpath, manifest)
                sys.exit("[calibrate] the reports need pandas, tabulate and matplotlib: pip install -e .[bench]")
        save(mpath, manifest)
        summary(run, manifest)
        print(f"[calibrate] {name}: {status} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    summary(run, manifest)
    bad = [n for n in chosen if manifest["steps"].get(n, {}).get("status") not in ("ok", None)]
    print(f"[calibrate] summary: {os.path.join(run, 'SUMMARY.md')}" + (f"; not ok: {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
