"""Calibrate this machine: measure the candidate white-noise samplers and solvers, reject the bad ones, and write
the tuning that ngsrng.Gaussian uses for its automatic choices.

    python bench/run_all.py --install            # "calibrate" (default): 2D/3D, 1e4 and 1e5 dofs, all devices
    python bench/run_all.py --profile smoke      # a few minutes: validates the harness end to end
    python bench/run_all.py --profile large --install    # calibrate up to 1e6 dofs
    python bench/run_all.py --profile research   # every suite (posteriors, distributions, ...): hours, not tuning
    python bench/run_all.py --resume results/runs/<run>   # continue an interrupted run

Each (suite, space, size) group runs in its own worker process, so crashes, out-of-memory kills and hangs are
recorded as results instead of ending the run. A candidate is rejected -- not measured at larger sizes -- once it
fails, exceeds its time budget, misses the accuracy threshold (cases.ACCURACY), or is RETIRE_RATIO times slower than
the best accurate candidate on its platform. Results go to results/runs/<host>_<date>_<profile>/ as JSON lines,
followed by report.md, recommendations.json, rejected.json and plots (see report.py).
"""

import argparse
import datetime
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from cases import ACCURACY, ITERATIVE, PROFILES, RETIRE_RATIO, build_groups  # noqa: E402

FAILED = {"error", "timeout", "crash", "memlimit"}


def accurate(rec) -> bool:
    """Whether an ok record meets the accuracy threshold of its suite (records of other suites always do)."""
    if rec.get("suite") == "noise":
        return rec.get("cov_rel_err", 0.0) <= ACCURACY["noise"]
    if rec.get("suite") == "prior":
        return all(not (rec.get(k) is not None and rec[k] > ACCURACY["prior"])
                   for k in ("pathwise_rel_err", "law_rel_fro"))
    return True


def best_time(rec) -> float:
    times = [rec.get(k) for k in ("per_sample_s", "per_sample_batch_s")]
    return min(t for t in times if t is not None) if any(t is not None for t in times) else float("inf")


def environment() -> dict:
    import ngsolve
    import numpy
    import scipy
    import torch

    env = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "mem_total_gb": _meminfo("MemTotal") / 2**20,
        "mem_available_gb": _meminfo("MemAvailable") / 2**20,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "versions": {
            "ngsolve": ngsolve.__version__,
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "cuda": None,
        "git": _git(),
    }
    if _cuda_usable():
        free, total = torch.cuda.mem_get_info()
        env["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "mem_total_gb": total / 2**30,
            "mem_free_gb": free / 2**30,
            "cuda": torch.version.cuda,
        }
    return env


def _cuda_usable():
    import torch

    try:
        return torch.cuda.is_available() and torch.empty(1, device="cuda") is not None
    except Exception:  # noqa: BLE001
        return False


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _meminfo(key):
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(key):
                    return float(line.split()[1])
    except OSError:
        pass
    return float("nan")


def _git():
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        )
        return f"{rev}{'+dirty' if dirty else ''}"
    except OSError:
        return None


class Runner:
    def __init__(self, run_dir, profile, mem_limit_gb, log):
        self.run_dir, self.cfg, self.mem_limit_gb, self.log = run_dir, PROFILES[profile], mem_limit_gb, log
        self.records = {}
        self.retired = set()
        self.rejected = {}  # lineage -> reason, written to rejected.json
        for suite in ("noise", "prior", "posterior", "distributions", "lowrank", "covariance", "continuum"):
            path = self.out_path(suite)
            if os.path.exists(path):
                with open(path) as fh:
                    for line in fh:
                        rec = json.loads(line)
                        self.records[rec["task_id"]] = rec
                        self.update_retired(rec)
        groups = {}
        for rec in self.records.values():
            groups.setdefault((rec["suite"], rec["dim"], rec["kind"], rec["order"], rec["target"]), []).append(rec)
        for recs in groups.values():
            self.retire_slow(recs)

    def out_path(self, suite):
        return os.path.join(self.run_dir, f"{suite}.jsonl")

    def update_retired(self, rec):
        if rec.get("suite") == "continuum":
            return
        too_slow = (
            rec.get("per_sample_s", 0) > self.cfg["sample_budget"] or rec.get("setup_s", 0) > self.cfg["setup_budget"]
        )
        reason = None
        if rec["status"] in FAILED:
            reason = f"{rec['status']} at {int(rec['target'])} dofs"
        elif too_slow:
            reason = f"over the time budget at {int(rec['target'])} dofs"
        elif rec["status"] == "ok" and not accurate(rec):
            err = rec.get("cov_rel_err", rec.get("pathwise_rel_err"))
            reason = f"inaccurate at {int(rec['target'])} dofs (error {err:.1e})"
        if reason:
            self.retired.add(rec["lineage"])
            self.rejected.setdefault(rec["lineage"], reason)
        if rec["status"] in ("timeout", "crash", "memlimit"):
            self.retired.add(sibling(rec))

    def retire_slow(self, recs):
        """Reject the candidates of one group that are RETIRE_RATIO x slower than the best accurate one on their
        platform (iterative solvers in 3D excepted: their cost grows more slowly than a factorisation's)."""
        ok = [r for r in recs if r.get("status") == "ok" and r["suite"] in ("noise", "prior") and accurate(r)]
        best = {}
        for r in ok:
            key = (r["suite"], r.get("power"), r["engine"], r["device"])
            best[key] = min(best.get(key, float("inf")), best_time(r))
        for r in ok:
            ratio = best_time(r) / best[(r["suite"], r.get("power"), r["engine"], r["device"])]
            if ratio > RETIRE_RATIO and not (r["dim"] == 3 and r.get("solver") in ITERATIVE):
                self.retired.add(r["lineage"])
                self.rejected.setdefault(r["lineage"], f"{ratio:.0f}x slower than the best at {int(r['target'])} dofs")

    def is_retired(self, t):
        return lineage(t) in self.retired or sibling(t) in self.retired

    def write_failure(self, task, status, detail):
        rec = {**task, "task_id": task_id(task), "lineage": lineage(task), "status": status, "error": detail}
        rec.pop("options", None)
        with open(self.out_path(task["suite"]), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.records[rec["task_id"]] = rec
        self.update_retired(rec)

    def run_group(self, group):
        pending = [t for t in group.to_dict()["tasks"] if task_id(t) not in self.records and not self.is_retired(t)]
        max_timeout = self.cfg["method_timeout"] * (4 if group.suite == "continuum" else 1)
        floor = self.cfg.get("timeout_floor", 60)
        fastest = None
        setup_failures = 0
        while pending:
            gfile = os.path.join(self.run_dir, "current_group.json")
            with open(gfile, "w") as fh:
                json.dump({**group.to_dict(), "tasks": pending}, fh)
            cmd = [
                sys.executable,
                "-u",
                os.path.join(HERE, "worker.py"),
                "--group",
                gfile,
                "--out",
                self.out_path(group.suite),
                "--mem-limit-gb",
                str(self.mem_limit_gb),
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
            lines = queue.Queue()

            def pump(stream=proc.stdout, q=lines):
                for item in stream:
                    q.put(item)
                q.put(None)

            threading.Thread(target=pump, daemon=True).start()
            current, started, finished, memlimit, killed, tail = None, time.time(), set(), False, False, []
            while True:
                try:
                    line = lines.get(timeout=1.0)
                except queue.Empty:
                    line = ""
                if line is None:
                    break
                if line:
                    line = line.rstrip()
                    if line.startswith("START "):
                        current, started = line[6:], time.time()
                    elif line.startswith("DONE "):
                        finished.add(line[5:])
                        took = time.time() - started
                        fastest = took if fastest is None else min(fastest, took)
                        current, started = None, time.time()
                    elif line.startswith("MEMLIMIT"):
                        memlimit = True
                    else:
                        tail = (tail + [line])[-20:]
                # Adaptive: a task may take at most 200x the fastest task of its group (at least ``floor`` seconds),
                # so a hanging method does not cost the full profile timeout when fast alternatives exist.
                timeout = max_timeout if fastest is None else min(max_timeout, max(floor, 200 * fastest))
                limit = timeout if current else self.cfg["setup_budget"] * 2
                if time.time() - started > limit:
                    proc.kill()
                    killed = True
                    self.log(f"    timeout after {limit:.0f}s in {current or 'group setup'}")
                    break
            code = proc.wait()
            self.reload(group.suite)
            by_id = {task_id(t): t for t in pending}
            if current is not None and current not in finished and current in by_id:
                status = "timeout" if killed else ("memlimit" if memlimit else "crash")
                self.write_failure(by_id[current], status, f"exit code {code}; " + " | ".join(tail[-3:]))
                self.log(f"    {status}: {current} (exit {code})")
            elif current is None and code != 0 and not finished:
                setup_failures += 1
                if setup_failures >= 2:
                    for t in pending:
                        if task_id(t) not in self.records:
                            self.write_failure(
                                t,
                                "memlimit" if memlimit else "crash",
                                f"group setup failed, exit code {code}; " + " | ".join(tail[-3:]),
                            )
                    self.log(f"    group setup failed twice (exit {code}): {' | '.join(tail[-2:])}")
                    return
            pending = [t for t in pending if task_id(t) not in self.records and not self.is_retired(t)]
        self.retire_slow([self.records[task_id(t)] for t in group.to_dict()["tasks"] if task_id(t) in self.records])

    def reload(self, suite):
        path = self.out_path(suite)
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec["task_id"] not in self.records:
                        self.records[rec["task_id"]] = rec
                        self.update_retired(rec)


def task_id(t):
    return "|".join(
        str(v)
        for v in (
            t["suite"],
            t["dim"],
            t["kind"],
            t["order"],
            int(t["target"]),
            t["engine"],
            t["device"],
            t["dtype"],
            t["method"],
            t["solver"],
            t["power"],
        )
    )


def lineage(t):
    return "|".join(
        str(v)
        for v in (
            t["suite"],
            t["dim"],
            t["kind"],
            t["order"],
            t["engine"],
            t["device"],
            t["dtype"],
            t["method"],
            t["solver"],
            t["power"],
        )
    )


def sibling(t):
    """Lineage across powers: a crash or hang of a solver configuration is not retried for other powers."""
    return lineage({**t, "power": "*"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="calibrate", choices=list(PROFILES))
    ap.add_argument("--suites", nargs="+", default=None,
                    help="default: the profile's (noise and prior, except for the research profile)")
    ap.add_argument("--devices", nargs="+", default=None, help="subset of: cpu cuda (default: all available)")
    ap.add_argument("--max-ndof", type=float, default=None)
    ap.add_argument(
        "--mem-limit-gb",
        type=float,
        default=None,
        help="per-worker resident memory limit (default: 80%% of currently available memory)",
    )
    ap.add_argument("--resume", default=None, help="run directory to continue")
    ap.add_argument("--out", default=None,
                    help="run directory of a new run (default: results/runs/<host>_<date>_<profile>)")
    ap.add_argument("--dry-run", action="store_true", help="list groups and task counts only")
    ap.add_argument(
        "--install",
        action="store_true",
        help="make this run's recommendations the active tuning for ngsrng.Gaussian (see ngsrng/tuning.py)",
    )
    args = ap.parse_args()

    env = environment()
    args.suites = args.suites or PROFILES[args.profile]["suites"]
    devices = args.devices or (["cpu", "cuda"] if env["cuda"] else ["cpu"])
    if "cuda" in devices and not env["cuda"]:
        sys.exit("CUDA requested but not usable")
    groups = build_groups(args.profile, args.suites, devices, args.max_ndof)
    ntasks = sum(len(g.tasks) for g in groups)
    if args.dry_run:
        for g in groups:
            print(f"{g.id:40s} {len(g.tasks):4d} tasks")
        print(f"{len(groups)} groups, {ntasks} tasks")
        return
    if args.resume:
        run_dir = args.resume
        with open(os.path.join(run_dir, "config.json")) as fh:
            saved = json.load(fh)
        args.profile, devices, args.suites = saved["profile"], saved["devices"], saved["suites"]
        groups = build_groups(args.profile, args.suites, devices, saved.get("max_ndof"))
        ntasks = sum(len(g.tasks) for g in groups)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        run_dir = args.out or os.path.join(ROOT, "results", "runs", f"{env['host']}_{stamp}_{args.profile}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as fh:
            json.dump(
                {
                    "profile": args.profile,
                    "suites": args.suites,
                    "devices": devices,
                    "max_ndof": args.max_ndof,
                    "profile_settings": PROFILES[args.profile],
                },
                fh,
                indent=1,
            )
    with open(os.path.join(run_dir, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=1)
    mem_limit = args.mem_limit_gb or 0.8 * env["mem_available_gb"]
    logfile = open(os.path.join(run_dir, "run.log"), "a")

    def log(msg):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        logfile.write(line + "\n")
        logfile.flush()

    log(
        f"run dir {run_dir}: profile {args.profile}, suites {args.suites}, devices {devices}, "
        f"{len(groups)} groups / {ntasks} tasks, worker memory limit {mem_limit:.1f} GB"
    )
    runner = Runner(run_dir, args.profile, mem_limit, log)
    t0 = time.time()
    for i, g in enumerate(groups):
        before = set(runner.records)
        tg = time.time()
        runner.run_group(g)
        new = [runner.records[k] for k in set(runner.records) - before]
        nfail = sum(r["status"] in FAILED for r in new)
        skipped = sum(runner.is_retired(t.to_dict()) and t.id not in runner.records for t in g.tasks)
        log(
            f"[{i + 1}/{len(groups)}] {g.id}: {len(new)} new, {nfail} failed"
            f"{f', {skipped} retired' if skipped else ''}, {time.time() - tg:.0f}s "
            f"(total {(time.time() - t0) / 60:.1f} min)"
        )
    with open(os.path.join(run_dir, "rejected.json"), "w") as fh:
        json.dump(runner.rejected, fh, indent=1)
    log(f"benchmarks finished in {(time.time() - t0) / 60:.1f} min; {len(runner.rejected)} candidates rejected; "
        "writing report")
    import report

    report.write(run_dir)
    log(f"report: {os.path.join(run_dir, 'report.md')}")
    if args.install:
        from ngsrng import tuning

        log(f"installed tuning: {tuning.install(run_dir)}")


if __name__ == "__main__":
    main()
