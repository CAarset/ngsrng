"""Thin torch helpers: every sample-time kernel runs on a torch device (CPU or CUDA)."""

import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp
import torch


def resolve_device(device) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if cuda_usable() else "cpu")
    return torch.device(device)


def cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.empty(1, device="cuda")
        return True
    except (RuntimeError, torch.AcceleratorError):
        return False


def resolve_dtype(dtype) -> torch.dtype:
    if dtype in (None, "float32", torch.float32, np.float32):
        return torch.float32
    if dtype in ("float64", torch.float64, np.float64):
        return torch.float64
    raise ValueError(f"unsupported dtype {dtype!r}")


def sparse_to_torch(mat: sp.spmatrix, device, dtype) -> torch.Tensor:
    mat = mat.tocsr()
    mat.sort_indices()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Ss]parse.*")
        return torch.sparse_csr_tensor(
            torch.from_numpy(mat.indptr.astype(np.int64)),
            torch.from_numpy(mat.indices.astype(np.int64)),
            torch.from_numpy(mat.data),
            size=mat.shape,
            check_invariants=False,
        ).to(device=device, dtype=dtype)


def spmm(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return (a @ x[:, None])[:, 0]
    return a @ x


def ngs_sparse(mat: sp.spmatrix, rows=None, nrows=None, cols=None, ncols=None):
    """scipy matrix -> NGSolve SparseMatrixd (multithreaded mat-vec), optionally re-indexing rows/columns."""
    from ngsolve.la import SparseMatrixd

    c = mat.tocoo()
    r = c.row if rows is None else np.asarray(rows)[c.row]
    k = c.col if cols is None else np.asarray(cols)[c.col]
    return SparseMatrixd.CreateFromCOO(
        r.astype(np.int64),
        k.astype(np.int64),
        c.data.astype(np.float64),
        mat.shape[0] if nrows is None else nrows,
        mat.shape[1] if ncols is None else ncols,
    )


class NormalStream:
    """Fast, reproducible N(0,1) generation into NumPy buffers (CPU engine).

    Chunks of 2^18 numbers each get their own SFC64 generator keyed by (seed, call counter, chunk index), so the
    stream does not depend on the number of threads. Threaded filling runs at several hundred M numbers/s;
    torch's CPU generator manages ~40 M/s.
    """

    chunk = 1 << 18
    _pool = None

    def __init__(self, seed=None):
        self.entropy = np.random.SeedSequence(seed).entropy
        self.counter = 0

    @classmethod
    def pool(cls):
        if cls._pool is None:
            import os
            from concurrent.futures import ThreadPoolExecutor

            cls._pool = ThreadPoolExecutor(os.cpu_count())
        return cls._pool

    def fill(self, out: np.ndarray):
        key, self.counter = self.counter, self.counter + 1
        nchunks = -(-len(out) // self.chunk)
        if nchunks == 1:
            self._job(out, key, 0)
        else:
            list(self.pool().map(lambda j: self._job(out, key, j), range(nchunks)))
        return out

    def reserve(self, n: int) -> int:
        """Reserve the keys of the next n calls of ``fill`` (for :meth:`fill_keyed`): returns the first."""
        key, self.counter = self.counter, self.counter + int(n)
        return key

    def fill_keyed(self, out: np.ndarray, key: int):
        """The numbers ``fill`` would produce as its call number ``key``, generated in the calling thread (for
        batches spread over threads: sample k of a batch gets key first + k, as in a loop of ``fill`` calls)."""
        for j in range(-(-len(out) // self.chunk)):
            self._job(out, key, j)
        return out

    def _job(self, out, key, j):
        ss = np.random.SeedSequence(self.entropy, spawn_key=(key, j))
        np.random.Generator(np.random.SFC64(ss)).standard_normal(out=out[j * self.chunk : (j + 1) * self.chunk])


def make_generator(seed, device) -> torch.Generator:
    gen = torch.Generator(device=device)
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(int(seed))
    return gen


@contextmanager
def numpy_blas_threads(n):
    """Limit only NumPy/SciPy's bundled OpenBLAS (prefix ``libscipy_openblas``), not NGSolve's BLAS.

    In the NGSolve engine, NumPy BLAS calls (GEMM, GEMV) sit between NGSolve kernels. OpenBLAS worker threads keep
    busy-waiting after a call and slow the following NGSolve solve by up to 2.3x (measured). ``n=None`` leaves the
    thread counts unchanged; without threadpoolctl this is a no-op.
    """
    sel = _numpy_blas_controller() if n is not None else None
    if sel is None:
        yield
        return
    with sel.limit(limits=int(n)):
        yield


_BLAS_SELECTION = []
_ALL_BLAS = []


@contextmanager
def blas_threads(n):
    """Limit every BLAS library in the process (NumPy's OpenBLAS, NGSolve's MKL and OpenBLAS) to n threads."""
    if not _ALL_BLAS:
        try:
            from threadpoolctl import ThreadpoolController

            _ALL_BLAS.append(ThreadpoolController().select(user_api="blas"))
        except ImportError:
            _ALL_BLAS.append(None)
    if _ALL_BLAS[0] is None:
        yield
        return
    with _ALL_BLAS[0].limit(limits=int(n)):
        yield


def _numpy_blas_controller():
    """Cached threadpoolctl selection of NumPy/SciPy's OpenBLAS (building it scans all libraries: ~50 ms;
    entering a limit on the cached selection costs ~10 us)."""
    if not _BLAS_SELECTION:
        try:
            from threadpoolctl import ThreadpoolController

            _BLAS_SELECTION.append(ThreadpoolController().select(prefix="libscipy_openblas"))
        except ImportError:
            _BLAS_SELECTION.append(None)
    return _BLAS_SELECTION[0]


# ------------------------------------------------------------------ threads (CPU engine)


def available_threads() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def available_memory() -> float | None:
    """Bytes of host memory available to new allocations (Linux ``MemAvailable``), or None where unknown."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return float(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


_POOL = []
_IN_WORKER = threading.local()


def parallel_ranges(n: int, work, threads: int | None = None):
    """Run ``work(lo, hi)`` on contiguous index ranges covering range(n), in a shared thread pool.

    NGSolve's kernels (sparse products, direct solves) release the GIL and are thread-safe on separate vectors, so
    independent samples or columns run in parallel; results do not depend on the number of threads. Calls from inside a
    worker run sequentially (no nested parallelism, no pool deadlock).
    """
    if getattr(_IN_WORKER, "active", False):
        work(0, n)
        return
    threads = min(available_threads() if threads is None else int(threads), n)

    def run(lo, hi):
        _IN_WORKER.active = True
        try:
            work(lo, hi)
        finally:
            _IN_WORKER.active = False

    # every BLAS (NumPy's OpenBLAS, NGSolve's MKL and OpenBLAS) single-threaded meanwhile: the parallelism is over
    # samples, nested BLAS threads would oversubscribe the cores, and per-call thread counts chosen at run time would
    # make the rounding depend on the schedule
    with blas_threads(1):
        if threads <= 1:
            run(0, n)
            return
        if not _POOL:
            _POOL.append(ThreadPoolExecutor(available_threads(), thread_name_prefix="ngsrng"))
        bounds = np.linspace(0, n, threads + 1).astype(int)
        futures = [_POOL[0].submit(run, lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]
        for f in futures:
            f.result()  # re-raises a worker's exception


class Scratch:
    """Per-thread work objects (e.g. NGSolve vectors), created by ``factory`` on first use in each thread: shared
    temporaries would make concurrent applications race."""

    def __init__(self, factory):
        self.factory = factory
        self._tls = threading.local()

    def get(self):
        obj = getattr(self._tls, "obj", None)
        if obj is None:
            obj = self._tls.obj = self.factory()
        return obj

    def __getstate__(self):  # copy.copy of an owner shares the factory, not other threads' objects
        return {"factory": self.factory}

    def __setstate__(self, state):
        self.factory = state["factory"]
        self._tls = threading.local()
