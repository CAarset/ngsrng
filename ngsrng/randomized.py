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
"""

import numpy as np
import scipy.linalg as sla


def euclidean_orth(y: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the columns of y (thin QR)."""
    q, _ = np.linalg.qr(y)
    return q


def gram_orth(gram):
    """Orthonormaliser for the inner product <x, y> = x^T G y: Cholesky QR, applied twice for stability. A block that
    is numerically rank deficient (e.g. powered blocks of a fast-decaying spectrum) falls back to the eigenvectors of
    its Gram matrix, dropping the dependent directions, so the result may have fewer columns."""

    def orth(y: np.ndarray) -> np.ndarray:
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


def randomized_svd(apply_a, apply_adj, m, rank, *, power_iterations=1, oversampling=10, seed=0, orth=None,
                   rtol=0.0, method="subspace"):
    """Truncated SVD A ~ U diag(s) V^T of an operator A: R^m -> Y given by ``apply_a`` (m x k -> n x k) and its
    adjoint ``apply_adj`` (n x k -> m x k). U is orthonormal in Y (``orth``, default Euclidean), V in R^m.

    ``method``: "subspace" (randomised subspace iteration, the default) or "power" (plain power iteration: (A A*)^q
    applied before a single orthonormalisation, kept for comparison). Singular values below ``rtol * s[0]`` are
    dropped. Returns (U, s, V).
    """
    if method not in ("subspace", "power"):
        raise ValueError("method must be 'subspace' or 'power'")
    orth = euclidean_orth if orth is None else orth
    ell = min(int(rank) + int(oversampling), int(m))
    omega = np.random.default_rng(seed).standard_normal((m, ell))
    y = apply_a(omega)
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
    ub, s, vt = np.linalg.svd(b, full_matrices=False)
    keep = np.arange(len(s)) < rank
    if s.size:
        keep &= s > rtol * s[0]
    return q @ ub[:, keep], s[keep], vt[keep].T


def randomized_qr(apply_a, apply_adj, m, rank, **kwargs):
    """Thin QR form A ~ Q R (Q: n x l orthonormal in Y, R: l x m upper trapezoidal), as in Aarset (2024,
    Algorithm 2). Same arguments as :func:`randomized_svd`."""
    u, s, v = randomized_svd(apply_a, apply_adj, m, rank, **kwargs)
    qh, r = np.linalg.qr(s[:, None] * v.T)
    return u @ qh, r
