"""The thread limit (NGSRNG_NUM_THREADS, OMP_NUM_THREADS, ngsrng.set_num_threads): ngsrng's pools and the libraries it
calls stay within it, and the results do not depend on it. Each case runs in a fresh process (the limit is read when
ngsrng is imported)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

# a Burgers-like inflow sampler (stationary Matern torus priors conditioned at two corners, Bernstein uniform fields
# restricted to the inflow edge, correlated, and their dense sampler): its set-up and draws, timed, with the peak
# number of the process's threads during them
BUILD = r"""
import json, os, sys, threading, time
import ngsrng
import ngsrng.resolution
ngsrng.resolution.RESOLUTION_TOL = None  # a coarse mesh on purpose, as in conftest
import ngsolve as ngs
import numpy as np
from netgen.geom2d import SplineGeometry

peak, stop = [0], threading.Event()

def watch():
    while not stop.is_set():
        peak[0] = max(peak[0], len(os.listdir("/proc/self/task")))
        time.sleep(0.001)

before = len(os.listdir("/proc/self/task"))
geo = SplineGeometry()
geo.AddRectangle((0, 0), (2, 1), bcs=["bottom", "right", "top", "left"])
fes = ngs.H1(ngs.Mesh(geo.GenerateMesh(maxh=0.25)), order=4)
monitor = threading.Thread(target=watch, daemon=True)
monitor.start()
w0, c0 = time.perf_counter(), time.process_time()
dists = []
for k in range(2):
    prior = ngsrng.StationaryMatern(fes, 0.3, nu=2, construction="torus", seed=k)
    post = prior.condition(points=np.array([[0.0, 0.0], [0.0, 1.0]]), noise=1e-10, data=[1.0, 2.0])
    rng = ngsrng.default_rng(fes, seed=10 + k, correlation=post, dtype="float64")
    dists.append(rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left"))
pair = ngsrng.correlated(dists, 0.8)
draws = pair.sample(100)
dense = pair.dense(seed=3).sample(200)
wall, cpu = time.perf_counter() - w0, time.process_time() - c0
stop.set()
monitor.join()
from ngsrng.backend import _POOL
np.savez(sys.argv[1], a=draws[0], b=draws[1], dense=dense)
print(json.dumps({"limit": ngsrng.get_num_threads(), "cpu_per_wall": cpu / wall, "before": before,
                  "peak": peak[0] - 1, "pool": max((len(p._threads) for p in _POOL), default=0)}))
"""


def _run(code, *args, **env):
    clean = {k: v for k, v in os.environ.items() if k not in ("NGSRNG_NUM_THREADS", "OMP_NUM_THREADS",
                                                                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    run = subprocess.run([sys.executable, "-W", "ignore", "-c", code, *args], capture_output=True, text=True,
                         timeout=900, env={**clean, "NGSRNG_QUIET": "1", **env})
    assert run.returncode == 0, run.stderr[-3000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_one_thread_and_the_same_results(tmp_path):
    """NGSRNG_NUM_THREADS=1: the set-up and the draws run on one thread (CPU time about the wall time, at most one pool
    worker, the process gains no other thread), and give the results of an unlimited run up to the rounding of the
    set-ups."""
    one = _run(BUILD, str(tmp_path / "one.npz"), NGSRNG_NUM_THREADS="1")
    assert one["limit"] == 1 and one["pool"] <= 1
    assert one["peak"] - one["before"] <= 1, one
    assert one["cpu_per_wall"] < 1.3, one
    _run(BUILD, str(tmp_path / "all.npz"))
    a, b = np.load(tmp_path / "one.npz"), np.load(tmp_path / "all.npz")
    for key in ("a", "b", "dense"):
        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-8 * np.abs(b[key]).max(), err_msg=key)


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 2, reason="needs two CPUs")
def test_settings_and_their_reach():
    """OMP_NUM_THREADS is honoured; set_num_threads(n) lowers torch, BLAS/OpenMP and NGSolve's task manager to n (never
    raises them) and resizes ngsrng's pools; None returns to the environment's limit."""
    code = r"""
import json
import ngsrng, torch, ngsolve as ngs
from threadpoolctl import threadpool_info
from ngsrng.backend import task_manager, parallel_ranges, _POOL
out = {"from_omp": ngsrng.get_num_threads()}
ngsrng.set_num_threads(1)
parallel_ranges(8, lambda lo, hi: None)
out["pool_1"] = max((len(p._threads) for p in _POOL), default=0) <= 1
with task_manager():
    out["ngsolve_1"] = ngs.GetNumThreads()
out["torch_1"], out["blas_1"] = torch.get_num_threads(), max(i["num_threads"] for i in threadpool_info())
ngsrng.set_num_threads(None)
out["back"] = ngsrng.get_num_threads()
try:
    ngsrng.set_num_threads(0)
except ValueError:
    out["zero_refused"] = True
print(json.dumps(out))
"""
    out = _run(code, OMP_NUM_THREADS="2")
    assert out == {"from_omp": 2, "pool_1": True, "ngsolve_1": 1, "torch_1": 1, "blas_1": 1, "back": 2,
                   "zero_refused": True}
