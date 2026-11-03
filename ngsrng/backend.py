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

    Chunks of 2^18 numbers each get their own SFC64 generator keyed by (stream, sample key, chunk index), so the
    stream does not depend on the number of threads. Threaded filling runs at several hundred M numbers/s;
    torch's CPU generator manages ~40 M/s. ``seed``: anything :class:`ngsrng.seeding.Source` accepts (None: torch's
    global generator, read at each draw).
    """

    chunk = 1 << 18
    _pool = None

    def __init__(self, seed=None):
        from .seeding import as_source

        self.source = as_source(seed)

    @classmethod
    def pool(cls):
        if cls._pool is None:
            import os
            from concurrent.futures import ThreadPoolExecutor

            cls._pool = ThreadPoolExecutor(os.cpu_count())
        return cls._pool

    def fill(self, out: np.ndarray):
        """One sample's numbers (one key), chunks filled in parallel threads."""
        ticket = self.reserve(1)
        nchunks = -(-len(out) // self.chunk)
        if nchunks == 1:
            self._job(out, ticket, 0, 0)
        else:
            list(self.pool().map(lambda j: self._job(out, ticket, 0, j), range(nchunks)))
        return out

    def reserve(self, n: int):
        """Reserve the keys of the next n samples (for :meth:`fill_keyed`), in the calling thread: an opaque ticket.
        Within one draw (:func:`ngsrng.seeding.one_draw`) consecutive reservations continue one another."""
        return self.source.streams().reserve(n)

    def fill_keyed(self, out: np.ndarray, ticket, k: int):
        """The numbers of sample k of a reservation, generated in the calling thread (batches spread over threads:
        every sample's numbers depend on its key only, not on the thread that makes them)."""
        for j in range(-(-len(out) // self.chunk)):
            self._job(out, ticket, k, j)
        return out

    def _job(self, out, ticket, k, j):
        seq, first = ticket
        ss = np.random.SeedSequence(seq.entropy, spawn_key=seq.spawn_key + (first + k, j))
        np.random.Generator(np.random.SFC64(ss)).standard_normal(out=out[j * self.chunk : (j + 1) * self.chunk])


def make_generator(seed, device) -> torch.Generator:
    gen = torch.Generator(device=device)
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(int(seed))
    return gen


@contextmanager
def fixed_numpy_global_rng(seed=0):
    """Run third-party code that draws from NumPy's global (legacy) random state -- pyamg's spectral-radius
    estimates start from ``np.random.rand`` -- with a fixed seed, and give the caller's state back afterwards: the
    result does not depend on the user's ``np.random.seed``, and the user's ``np.random`` stream does not advance.
    (Like any use of the global state, not isolated from other threads drawing from it at the same time.)"""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


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


def _forget_pools_after_fork():
    """A forked child inherits the thread pools' bookkeeping but none of their threads: a pool that counts idle
    threads starts no new ones, and the child's first threaded batch waits forever (e.g. in a DataLoader worker
    forked after the parent sampled). The child starts new pools on first use instead."""
    _POOL.clear()
    NormalStream._pool = None
    _IN_WORKER.__dict__.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_pools_after_fork)


def parallel_ranges(n: int, work, threads: int | None = None):
    """Run ``work(lo, hi)`` on contiguous index ranges covering range(n), in a shared thread pool.

    NGSolve's kernels (sparse products, direct solves) release the GIL and are thread-safe on separate vectors, so
    independent samples or columns run in parallel; results do not depend on the number of threads. Calls from inside a
    worker run sequentially (no nested parallelism, no pool deadlock).

    The work always runs on pool threads, a single range too, never on the caller's: NGSolve runs its kernels
    single-threaded there even inside the caller's ``with TaskManager():`` block (measured), whereas on the caller's
    thread they would use the task manager, whose threads add in a varying order: results would then vary at
    rounding level from call to call.
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
        if not _POOL:
            _POOL.append(ThreadPoolExecutor(available_threads(), thread_name_prefix="ngsrng"))
        bounds = np.linspace(0, n, threads + 1).astype(int)
        futures = [_POOL[0].submit(run, lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]
        for f in futures:
            f.result()  # re-raises a worker's exception


class Rebuildable:
    """Pickling for objects that hold what does not pickle: NGSolve sparse matrices, factorisations, preconditioners
    and integration rules, cuDSS handles. Every ``_ngs_*`` attribute and those named in ``_transient`` are left out
    when pickling, and ``_restore()`` rebuilds them after unpickling (so a worker process that receives a sampler
    assembles and factorises once). ``copy.copy`` stays a plain shallow copy that shares them.

    No lambda or nested function may be stored on such objects (they do not pickle): module-level functions, bound
    methods or small callable classes instead (``tests/test_pickling.py`` checks every sampler).
    """

    _transient = ()

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not (k.startswith("_ngs_") or k in self._transient)}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._restore()

    def _restore(self):
        """Rebuild what ``__getstate__`` left out (default: nothing; lazily built attributes need nothing)."""

    def __copy__(self):
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new


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
