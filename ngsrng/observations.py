"""Observation operators: any finite set of bounded linear functionals on a FE space, as sparse G (m x ndof)."""

import re

import ngsolve as ngs
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

CHUNK = 256


def observation_operator(fes, matrix=None, functionals=None, points=None, averages=None, adjoint=None, m=None):
    """Observation matrix G (m x ndof) acting on coefficient vectors: any finite set of bounded linear functionals.

    Rows are stacked from any combination of:

    - ``matrix``: array, sparse matrix, or ``scipy.sparse.linalg.LinearOperator`` (m x ndof). A LinearOperator is
      applied through its adjoint, so it may wrap a PDE solve, e.g. G = O A^-1;
    - ``adjoint`` (with ``m``): callable Y (m x k) -> G^T Y (ndof x k) returning dual (load) vectors; m applications
      of it are the irreducible setup cost of a matrix-free observation map;
    - ``functionals``: LinearForms on ``fes`` (assembled or not; any measure, e.g. boundary fluxes with ``ds`` or
      derivative terms) or CoefficientFunctions psi, meaning f -> int f psi dx;
    - ``averages``: ``(centers, radius)`` or ``(centers, radius, kernel)`` with kernel "ball" (default) or "gaussian":
      normalised local averages f -> int f psi_i / int psi_i around each centre;
    - ``points``: (m x dim) point evaluations f -> f(x_i). Only meaningful for continuous samples (point evaluation is
      not bounded on L2); local averages are the robust alternative.
    """
    blocks = []
    if matrix is not None:
        if hasattr(matrix, "rmatmat") or (hasattr(matrix, "T") and hasattr(matrix, "matvec")):
            blocks.append(sp.csr_matrix(np.asarray(matrix.T @ np.eye(matrix.shape[0])).T))
        else:
            blocks.append(sp.csr_matrix(matrix))
    if adjoint is not None:
        if m is None:
            raise ValueError("adjoint needs the number of observations m")
        cols = [np.asarray(adjoint(np.eye(m)[:, i : i + CHUNK])) for i in range(0, m, CHUNK)]
        blocks.append(sp.csr_matrix(np.hstack(cols).T))
    if functionals:
        rows = []
        for fn in functionals:
            if isinstance(fn, ngs.LinearForm):
                if fn.vec.size != fes.ndof:
                    raise ValueError("LinearForm is defined on a different space")
                if not np.any(fn.vec.FV().NumPy()):
                    fn.Assemble()
                rows.append(fn.vec.FV().NumPy().copy())
            else:
                rows.append(ngs.LinearForm(fn * fes.TestFunction() * ngs.dx).Assemble().vec.FV().NumPy().copy())
        blocks.append(sp.csr_matrix(np.array(rows)))
    if averages is not None:
        blocks.append(_average_rows(fes, *averages))
    if points is not None:
        blocks.append(_point_rows(fes, np.atleast_2d(np.asarray(points, dtype=float))))
    if not blocks:
        raise ValueError("no observations given")
    g = sp.vstack(blocks).tocsr()
    if g.shape[1] != fes.ndof:
        raise ValueError(f"observation matrix has {g.shape[1]} columns, space has {fes.ndof} dofs")
    g.eliminate_zeros()
    return g


def _average_rows(fes, centers, radius, kernel="ball", bonus_intorder=4):
    """Rows of f -> int f psi / int psi with psi a ball indicator or a Gaussian of width ``radius``.

    Each row is normalised by its own value on the (exactly interpolated) constant function, so averages of constants
    are exact whatever quadrature resolves the kernel.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    coords = [ngs.x, ngs.y, ngs.z][: fes.mesh.dim]
    dx = ngs.dx(bonus_intorder=bonus_intorder)
    v = fes.TestFunction()
    one = ngs.GridFunction(fes)
    one.Set(1.0)
    ones = one.vec.FV().NumPy()
    rows = []
    for c in centers:
        r2 = sum((xi - ci) ** 2 for xi, ci in zip(coords, c))
        psi = ngs.IfPos(radius**2 - r2, 1.0, 0.0) if kernel == "ball" else ngs.exp(-r2 / (2 * radius**2))
        row = ngs.LinearForm(psi * v * dx).Assemble().vec.FV().NumPy().copy()
        vol = row @ ones
        if vol <= 0:
            raise ValueError(f"averaging region around {c} does not intersect the mesh")
        rows.append(row / vol)
    return sp.csr_matrix(np.array(rows))


def _point_rows(fes, points):
    """Exact point-evaluation rows from NGSolve's PointEvaluationFunctional (sparse; values read back exactly)."""
    v = fes.TestFunction()
    probe = ngs.la.BaseVector(fes.ndof)
    pv = probe.FV().NumPy()
    pv[:] = 0.0
    rows, cols, vals = [], [], []
    for i, p in enumerate(points):
        sv = v(*p).Assemble()
        idx = [int(t) for t in re.findall(r"(\d+):", str(sv))]
        for j in idx:
            pv[j] = 1.0
            rows.append(i)
            cols.append(j)
            vals.append(sv.InnerProduct(probe.FV()))
            pv[j] = 0.0
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(points), fes.ndof))


def whitener(noise, m):
    """Gamma^-1/2 as (kind, data): 'diag' with a vector, or 'dense' with C^-1 for Gamma = C C^T."""
    arr = np.asarray(noise, dtype=float)
    if arr.ndim == 0:
        return "diag", np.full(m, 1.0 / np.sqrt(float(arr)))
    if arr.ndim == 1:
        if arr.shape != (m,):
            raise ValueError("noise variance vector has the wrong length")
        return "diag", 1.0 / np.sqrt(arr)
    if arr.shape != (m, m):
        raise ValueError("noise covariance has the wrong shape")
    return "dense", sla.solve_triangular(np.linalg.cholesky(arr), np.eye(m), lower=True)


def whiten(kind, data, y):
    return data[:, None] * y if (kind == "diag" and y.ndim == 2) else (data * y if kind == "diag" else data @ y)
