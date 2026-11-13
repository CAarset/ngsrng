"""Randomised low-rank factorisations of operators known only through their action and adjoint action.

Randomised subspace iteration (Halko, Martinsson & Tropp 2011, Algs. 4.4 and 5.1), in the form of Aarset (2024,
arXiv:2410.16590, Appendix A, Algorithm 2), for a possibly non-symmetric operator A: R^m -> Y:

    Q_0 = orth(A Omega)                                  Omega: m x l standard Gaussian, l = rank + oversampling
    for j = 1..q:  Qt_j = qr(A* Q_{j-1}),  Q_j = orth(A Qt_j)
    B = (A* Q_q)^T = Q_q* A  (l x m),  B = U S V^T        (SVD; small singular values may be truncated)
    A ~ (Q_q U) S V^T,   or in QR form  A ~ Q R  with  S V^T = Qh R,  Q = Q_q U Qh.

Re-orthonormalising after *every* application of A and of A* (instead of applying (A A*)^q to a block and
orthonormalising once, plain power iteration) keeps the singular values of the iterated block at their own scale, so
the small ones are not lost to rounding; the final SVD of B (instead of an eigendecomposition of B B^T) does not
square the condition number either. Cost: l (2q + 2) applications of A or A*.

The output space Y may carry a weighted inner product (e.g. the L2 inner product of an FE space, Gram matrix M): pass
its orthonormaliser as ``orth`` and the adjoint with respect to it as ``apply_adj`` (e.g. A* = A^T M). The input space
is Euclidean (white noise, whitened observations).

Every routine works on NumPy arrays or, with ``device`` (or torch inputs), on torch tensors on that device: then the
operators are applied, and all dense algebra done, on the device. The Gaussian test matrix Omega always comes from
NumPy with the given seed, so the result does not depend on the device beyond rounding.
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import torch


def _is_torch(x) -> bool:
    return isinstance(x, torch.Tensor)


def euclidean_orth(y):
    """Orthonormal basis of the columns of y (thin QR); NumPy or torch."""
    if _is_torch(y):
        return torch.linalg.qr(y).Q
    q, _ = np.linalg.qr(y)
    return q


def gram_orth(gram):
    """Orthonormaliser for the inner product <x, y> = x^T G y: Cholesky QR, applied twice for stability. A block that
    is numerically rank deficient (e.g. powered blocks of a fast-decaying spectrum) falls back to the eigenvectors of
    its Gram matrix, dropping the dependent directions, so the result may have fewer columns. ``gram`` is a SciPy
    (sparse) or NumPy matrix; torch blocks are orthonormalised on their device (G copied there once)."""
    on_device = {}

    def mul(y):
        if not _is_torch(y):
            return gram @ y
        key = (str(y.device), y.dtype)
        if key not in on_device:
            from .backend import sparse_to_torch

            on_device[key] = (sparse_to_torch(sp.csr_matrix(gram), y.device, y.dtype) if sp.issparse(gram)
                              else torch.as_tensor(np.asarray(gram), dtype=y.dtype, device=y.device))
        return on_device[key] @ y

    def orth_torch(y):
        for _ in range(2):
            c = y.T @ mul(y)
            c = 0.5 * (c + c.T)
            r, info = torch.linalg.cholesky_ex(c)
            if int(info) == 0:
                y = torch.linalg.solve_triangular(r, y.T, upper=False).T
            else:
                lam, v = torch.linalg.eigh(c)
                keep = lam > lam.max() * 1e-13
                y = (y @ v[:, keep]) / torch.sqrt(lam[keep])
        return y

    def orth(y):
        if _is_torch(y):
            return orth_torch(y)
        for _ in range(2):
            c = y.T @ (gram @ y)
            c = 0.5 * (c + c.T)
            try:
                r = np.linalg.cholesky(c)
                y = sla.solve_triangular(r, y.T, lower=True).T
            except np.linalg.LinAlgError:
                lam, v = np.linalg.eigh(c)
                keep = lam > lam.max() * 1e-13
                y = (y @ v[:, keep]) / np.sqrt(lam[keep])
        return y

    return orth


def gaussian_block(rows, cols, seed, device=None, dtype=torch.float64):
    """A standard Gaussian test matrix from NumPy (seeded), on ``device`` as a torch tensor if one is given."""
    omega = np.random.default_rng(seed).standard_normal((rows, cols))
    return omega if device is None else torch.from_numpy(omega).to(device, dtype)


def randomized_svd(apply_a, apply_adj, m, rank, *, power_iterations=1, oversampling=10, seed=0, orth=None,
                   rtol=0.0, method="subspace", device=None):
    """Truncated SVD A ~ U diag(s) V^T of an operator A: R^m -> Y given by ``apply_a`` (m x k -> n x k) and its
    adjoint ``apply_adj`` (n x k -> m x k). U is orthonormal in Y (``orth``, default Euclidean), V in R^m.

    ``method``: "subspace" (randomised subspace iteration, the default) or "power" (plain power iteration: (A A*)^q
    applied before a single orthonormalisation, kept for comparison). Singular values below ``rtol * s[0]`` are
    dropped. ``device``: a torch device, on which the operators then act on (and return) float64 tensors and the
    dense algebra runs; None for NumPy. Returns (U, s, V), NumPy arrays or tensors on ``device``.
    """
    if method not in ("subspace", "power"):
        raise ValueError("method must be 'subspace' or 'power'")
    orth = euclidean_orth if orth is None else orth
    ell = min(int(rank) + int(oversampling), int(m))
    y = apply_a(gaussian_block(m, ell, seed, device))
    if method == "subspace":
        q = orth(y)
        for _ in range(power_iterations):
            qt = euclidean_orth(apply_adj(q))
            q = orth(apply_a(qt))
    else:
        for _ in range(power_iterations):
            y = apply_a(apply_adj(y))
        q = orth(y)
    b = apply_adj(q).T  # Q* A (l x m)
    if _is_torch(b):
        ub, s, vt = torch.linalg.svd(b, full_matrices=False)
        keep = torch.arange(len(s), device=s.device) < rank
        if s.numel():
            keep &= s > rtol * s[0]
    else:
        ub, s, vt = np.linalg.svd(b, full_matrices=False)
        keep = np.arange(len(s)) < rank
        if s.size:
            keep &= s > rtol * s[0]
    return q @ ub[:, keep], s[keep], vt[keep].T


def randomized_qr(apply_a, apply_adj, m, rank, **kwargs):
    """Thin QR form A ~ Q R (Q: n x l orthonormal in Y, R: l x m upper trapezoidal), as in Aarset (2024,
    Algorithm 2). Same arguments as :func:`randomized_svd` (with ``device``: torch tensors on it)."""
    u, s, v = randomized_svd(apply_a, apply_adj, m, rank, **kwargs)
    if _is_torch(u):
        qh, r = torch.linalg.qr(s[:, None] * v.T)
    else:
        qh, r = np.linalg.qr(s[:, None] * v.T)
    return u @ qh, r
