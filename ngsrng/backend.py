"""Thin torch helpers: every sample-time kernel runs on a torch device (CPU or CUDA)."""

import warnings
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

        def job(j):
            ss = np.random.SeedSequence(self.entropy, spawn_key=(key, j))
            np.random.Generator(np.random.SFC64(ss)).standard_normal(out=out[j * self.chunk : (j + 1) * self.chunk])

        if nchunks == 1:
            job(0)
        else:
            list(self.pool().map(job, range(nchunks)))
        return out


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
