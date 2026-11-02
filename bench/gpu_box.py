"""Everything pending on the GPU box, in one command: tests, validation and all benchmarks, with logs and a summary.

    python bench/gpu_box.py --install-extras      # first run: also pip-install the optional sparse backends
    python bench/gpu_box.py                       # all steps, standard profiles (several hours)
    python bench/gpu_box.py --quick               # quick profiles of every step (about an hour)
    python bench/gpu_box.py --resume results/gpu_box/<run>   # rerun only the steps that did not finish
    python bench/gpu_box.py --steps tests,boundary          # a subset (--list shows the steps)

Steps, in order:

  env              environment snapshot: GPU, driver, CUDA, torch, cuDSS, NGSolve, optional backends, git state
  extras           (only with --install-extras) torch-sla without extras, cupy for torch's CUDA major, pyamg,
                   and the report dependencies pandas, tabulate, matplotlib (the "bench" extra)
  tests            the full pytest suite, CUDA variants included when a device is usable (junit.xml)
  gpu_batch        bench/gpu_batch.py: device checks (normal CDF tails, variances, KS, out= vs returned) and
                   in-place batch throughput of normal / lognormal / uniform, cuDSS widths, PCG, float32/64
  gpu_sweep        bench/gpu_batch.py --quick with work_values 2^22, 2^24 (default), 2^26 and peak device memory
  boundary         bench/boundary_scaling.py: restricted and conditioned draws, set-up, path, datum error, memory
  sparse_quick     bench/sparse_backends.py --profile quick: products and SPD solves by backend
  sparse_standard  bench/sparse_backends.py --profile standard (not in --quick)
  run_all          bench/run_all.py --install: every sampler, engine, device and size; installs the tuning

Each step runs as a subprocess with a time limit. Its output goes to the console and to logs/<step>.log, and its
status, exit code and duration go to manifest.json, so an interrupted or failed run is continued with --resume
(boundary_scaling and run_all also resume inside the step). SUMMARY.md lists every step with its log and report.
The exit code is nonzero if a step failed.
"""

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
HOUR = 3600

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
for mod in ("torch_sla", "cupy", "pyamg", "pandas", "tabulate", "matplotlib"):
    try:
        out[mod] = getattr(importlib.import_module(mod), "__version__", "installed")
    except Exception as err:
        out[mod] = None
try:
    out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except FileNotFoundError:
    out["nvidia_smi"] = None
print(json.dumps(out, indent=1))
"""


def steps(args, run):
    """(name, command or callable, timeout in seconds, report path relative to the run directory, in --quick)."""
    q = args.quick
    out = os.path.join(run, "out")
    run_all_dir = os.path.join(out, "run_all")
    run_all = [PY, "bench/run_all.py", "--profile", "smoke" if q else "standard", "--install"]
    run_all += (["--resume", run_all_dir] if os.path.exists(os.path.join(run_all_dir, "config.json"))
                else ["--out", run_all_dir])
    tests = ["tests"] if not q else ["tests/test_inplace.py", "tests/test_boundary.py", "tests/test_gaussian.py",
                                     "tests/test_posterior.py", "tests/test_generator.py"]
    return [
        ("env", [PY, "-c", ENV_SCRIPT], 600, None),
        ("extras", install_extras, HOUR, None),
        ("tests", [PY, "-m", "pytest", "-q", "-rsfE", "-p", "no:cacheprovider", *tests,
                   f"--junitxml={os.path.join(out, 'junit.xml')}"], (1 if q else 4) * HOUR, "out/junit.xml"),
        ("gpu_batch", [PY, "bench/gpu_batch.py", *(["--quick"] if q else []), "--out-dir",
                       os.path.join(out, "gpu_batch")], (1 if q else 3) * HOUR, "out/gpu_batch/report.md"),
        ("gpu_sweep", [PY, "bench/gpu_batch.py", "--quick", "--work-values", "2**22,2**24,2**26", "--out-dir",
                       os.path.join(out, "gpu_sweep")], 2 * HOUR, "out/gpu_sweep/report.md"),
        ("boundary", [PY, "bench/boundary_scaling.py", "--profile", "quick" if q else "standard", "--out-dir",
                      os.path.join(out, "boundary")], (1 if q else 6) * HOUR, "out/boundary/report.md"),
        ("sparse_quick", [PY, "bench/sparse_backends.py", "--profile", "quick", "--out",
                          os.path.join(out, "sparse_quick")], 2 * HOUR, "out/sparse_quick/report.md"),
        ("sparse_standard", None if q else [PY, "bench/sparse_backends.py", "--profile", "standard", "--out",
                                             os.path.join(out, "sparse_standard")], 8 * HOUR,
         "out/sparse_standard/report.md"),
        ("run_all", run_all, (2 if q else 24) * HOUR, "out/run_all/report.md"),
    ]


def install_extras(log):
    """Optional sparse backends: torch-sla without its [cudss] extra, which pins
    nvmath-python[cu12]; cupy for torch's CUDA major version; pyamg; and the report dependencies of the benchmarks.
    Failures are logged and make the step fail, but the following steps still run."""
    import torch

    major = (torch.version.cuda or "13").split(".")[0]
    pip = [PY, "-m", "pip", "install"]
    if subprocess.run([PY, "-m", "pip", "--version"], capture_output=True).returncode != 0 and shutil.which("uv"):
        pip = ["uv", "pip", "install", "--python", PY]
    failed = 0
    for pkg in ("torch-sla", f"cupy-cuda{major}x", "pyamg", "pandas", "tabulate", "matplotlib"):
        rc = stream(pip + [pkg], log, HOUR)
        log(f"[gpu_box] install {pkg}: {'ok' if rc == 0 else f'failed ({rc})'}")
        failed += rc != 0
    return 0 if failed == 0 else 1


def stream(cmd, log, timeout):
    """Run cmd from the repository root, echoing its output to the console and the step log; return the exit code."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    t0 = time.time()
    try:
        for line in proc.stdout:
            log(line.rstrip("\n"))
            if time.time() - t0 > timeout:
                raise subprocess.TimeoutExpired(cmd, timeout)
        return proc.wait(timeout=max(1.0, timeout - (time.time() - t0)))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        log(f"[gpu_box] timeout after {timeout} s")
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
    lines = [f"# GPU box run `{os.path.basename(run)}`\n",
             f"- host: {manifest['host']}, started {manifest['started']}, git {manifest['git']}",
             f"- profile: {'quick' if manifest['quick'] else 'standard'}\n",
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
                          ("gpu_batch", ("checks:",)), ("gpu_sweep", ("checks:",))):
        path = os.path.join(run, "logs", f"{name}.log")
        if os.path.exists(path):
            hits = [ln for ln in open(path) if any(p in ln for p in pattern)][-3:]
            if hits:
                lines.append(f"\n## {name}\n\n```\n" + "".join(hits).rstrip() + "\n```")
    with open(os.path.join(run, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="quick profiles of every step")
    ap.add_argument("--install-extras", action="store_true", help="pip-install torch-sla, cupy and pyamg first")
    ap.add_argument("--steps", default=None, help="comma-separated subset of the steps")
    ap.add_argument("--skip", default="", help="comma-separated steps to leave out")
    ap.add_argument("--resume", default=None, help="run directory to continue: steps with status ok are skipped")
    ap.add_argument("--allow-cpu", action="store_true", help="run even if no CUDA device is usable (dry run)")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    args = ap.parse_args()

    if args.resume:
        run = os.path.abspath(args.resume)
        manifest = json.load(open(os.path.join(run, "manifest.json")))
        args.quick = manifest["quick"]
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        tag = f"{socket.gethostname()}_{stamp}{'_quick' if args.quick else ''}"
        run = os.path.join(ROOT, "results", "gpu_box", tag)
        manifest = {"host": socket.gethostname(), "started": datetime.now().isoformat(timespec="seconds"),
                    "quick": args.quick, "python": platform.python_version(), "git": git_state(), "steps": {}}
    plan = steps(args, run)
    names = [n for n, cmd, _, _ in plan if cmd is not None]
    if args.list:
        print("\n".join(names))
        return
    chosen = args.steps.split(",") if args.steps else names
    unknown = set(chosen) - set(names) | (set(filter(None, args.skip.split(","))) - set(names))
    if unknown:
        sys.exit(f"unknown steps {sorted(unknown)}; available: {names}")
    chosen = [n for n in chosen if n not in args.skip.split(",")]
    if not args.install_extras and "extras" in chosen and not args.steps:
        chosen.remove("extras")
    os.makedirs(os.path.join(run, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run, "out"), exist_ok=True)
    mpath = os.path.join(run, "manifest.json")
    print(f"[gpu_box] run directory: {run}\n[gpu_box] steps: {', '.join(chosen)}", flush=True)
    if manifest["git"]["dirty_files"]:
        print(f"[gpu_box] note: the working tree has {manifest['git']['dirty_files']} modified files", flush=True)

    for name, cmd, timeout, report in plan:
        if name not in chosen:
            continue
        rec = manifest["steps"].get(name, {})
        if rec.get("status") == "ok":
            print(f"[gpu_box] {name}: already ok, skipped", flush=True)
            continue
        logpath = os.path.join(run, "logs", f"{name}.log")
        captured = []
        with open(logpath, "a") as fh:
            def log(line, fh=fh, name=name):
                print(f"[{name}] {line}", flush=True)
                fh.write(line + "\n")
                fh.flush()
                captured.append(line)

            log(f"[gpu_box] {datetime.now():%Y-%m-%d %H:%M:%S} start: "
                + (" ".join(cmd) if isinstance(cmd, list) and name != "env" else name))
            t0 = time.time()
            rc = cmd(log) if callable(cmd) else stream(cmd, log, timeout)
        status = "ok" if rc == 0 else "timeout" if rc == "timeout" else "failed"
        manifest["steps"][name] = {"status": status, "exit": rc, "seconds": time.time() - t0,
                                   "finished": datetime.now().isoformat(timespec="seconds"), "report": report}
        if name == "env" and rc == 0:
            text = "\n".join(captured)
            try:
                manifest["env"] = json.loads(text[text.index("{"):text.rindex("}") + 1])
            except ValueError:
                pass
            if not manifest.get("env", {}).get("cuda_usable") and not args.allow_cpu:
                save(mpath, manifest)
                sys.exit("[gpu_box] no usable CUDA device; rerun with --allow-cpu for a dry run")
        save(mpath, manifest)
        summary(run, manifest)
        print(f"[gpu_box] {name}: {status} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    summary(run, manifest)
    bad = [n for n in chosen if manifest["steps"].get(n, {}).get("status") not in ("ok", None)]
    print(f"[gpu_box] summary: {os.path.join(run, 'SUMMARY.md')}" + (f"; not ok: {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
