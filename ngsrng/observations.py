"""Observation operators: any finite set of bounded linear functionals on a FE space, as sparse G (m x ndof)."""

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

    Every observation must see the field: a row that is identically zero (e.g. a ``ds`` region name that matches no
    boundary) raises ValueError, as does a point outside the mesh.
    """
    # Rows are kept sparse as they come (G is m x ndof: dense intermediates would cost m ndof values); matrix-free maps
    # are applied CHUNK observations at a time
    blocks = []
    if matrix is not None:
        if hasattr(matrix, "rmatmat") or (hasattr(matrix, "T") and hasattr(matrix, "matvec")):
            blocks.append(_from_adjoint(lambda e: matrix.T @ e, matrix.shape[0]))
        else:
            blocks.append(sp.csr_matrix(matrix))
    if adjoint is not None:
        if m is None:
            raise ValueError("adjoint needs the number of observations m")
        blocks.append(_from_adjoint(adjoint, m))
    if functionals:
        rows = []
        for fn in functionals:
            if isinstance(fn, ngs.LinearForm):
                if fn.vec.size != fes.ndof:
                    raise ValueError("LinearForm is defined on a different space")
                if not np.any(fn.vec.FV().NumPy()):
                    fn.Assemble()
                rows.append(sp.csr_matrix(fn.vec.FV().NumPy()))
            else:
                rows.append(sp.csr_matrix(ngs.LinearForm(fn * fes.TestFunction() * ngs.dx).Assemble().vec.FV().NumPy()))
        blocks.append(sp.vstack(rows))
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
    empty = np.flatnonzero(np.diff(g.indptr) == 0)
    if empty.size:
        raise ValueError(f"observation(s) {empty[:10].tolist()} are identically zero (rows in the order: matrix, "
                         "adjoint, functionals, averages, points), e.g. a boundary name that matches nothing; an "
                         "observation must depend on the field")
    return g


def _from_adjoint(adjoint, m):
    """G (m x ndof, sparse) from Y -> G^T Y, applied to CHUNK unit columns at a time."""
    blocks = []
    for i in range(0, m, CHUNK):
        e = np.zeros((m, min(CHUNK, m - i)))
        e[np.arange(i, i + e.shape[1]), np.arange(e.shape[1])] = 1.0
        blocks.append(sp.csr_matrix(np.asarray(adjoint(e)).T))
    return sp.vstack(blocks)


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
        rows.append(sp.csr_matrix(row / vol))
    return sp.vstack(rows)


def _point_rows(fes, points):
    """Exact point-evaluation rows from NGSolve's PointEvaluationFunctional (sparse; values read back exactly).

    The functional only exposes inner products, so its entries are read with unit vectors on the dofs of the element
    containing the point (the element NGSolve evaluates in). Each row must reproduce constants (row . 1_h = 1), which
    checks that no entry was missed.
    """
    mesh = fes.mesh
    if points.shape[1] != mesh.dim:
        raise ValueError(f"points have {points.shape[1]} coordinates, the mesh is {mesh.dim}D")
    v = fes.TestFunction()
    probe = ngs.la.BaseVector(fes.ndof)
    pv = probe.FV().NumPy()
    pv[:] = 0.0
    one = ngs.GridFunction(fes)
    one.Set(1.0)
    ones = one.vec.FV().NumPy()
    rows, cols, vals = [], [], []
    for i, p in enumerate(points):
        el = mesh(*p).nr
        if el < 0:  # NGSolve 6.2.2608 segfaults assembling a point evaluation outside the mesh (known issue 15)
            raise ValueError(f"point {tuple(float(c) for c in p)} lies outside the mesh")
        sv = v(*p).Assemble()
        row = {}
        for j in fes.GetDofNrs(ngs.ElementId(ngs.VOL, el)):
            if j < 0 or j in row:
                continue
            pv[j] = 1.0
            row[j] = sv.InnerProduct(probe.FV())
            pv[j] = 0.0
        if abs(sum(val * ones[j] for j, val in row.items()) - 1.0) > 1e-8:
            raise RuntimeError(f"could not read the point evaluation at {tuple(float(c) for c in p)} from element {el}")
        rows += [i] * len(row)
        cols += list(row)
        vals += list(row.values())
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
