"""Does the mesh resolve the correlation length? An automatic check of the Matérn fields' discretisation.

The Matérn SPDE discretised with elements of size h and order p deviates from the continuous field when the
correlation length is not well resolved: the field is too smooth, its covariance and pointwise variance too small.
:func:`check_resolution` measures this where nothing else interferes: the SPDE on a periodic box with a structured mesh
of the same element size and order, whose continuous pointwise variance is known exactly (the Matérn correlation
summed over the box's periodic images, :func:`periodised_variance`). The discrete field's exact variance at the points
of one cell (a vertex, the midpoints of its edges, the centroids of its simplices) is compared with it; a relative
error above ``tol`` (default ``RESOLUTION_TOL``, 1%) raises a ValueError that says how far off the field is and how
fine the mesh would have to be.

The box is two correlation lengths wide (images are accounted for exactly, so any width is valid), with at least four
cells per side and at most as many DoFs as the space the field lives on: the check never costs more than one field.
It runs where Matérn fields are made (``ngsrng.matern_correlation(...)`` bound to a space,
:class:`ngsrng.StationaryMatern`, the smallest length of a random one) and is cached per element size, order,
dimension, length and smoothness.
"""

import functools
import itertools
import math

import numpy as np

__all__ = ["RESOLUTION_TOL", "check_resolution", "resolution_error", "periodised_variance"]

RESOLUTION_TOL = 0.01  # the largest relative error of the discrete field's pointwise variance accepted by default


def periodised_variance(box, length, nu, dim, rtol=1e-12) -> float:
    """The pointwise variance of the unit Matérn field periodised over a cubic box of side ``box``: sum over the image
    lattice of the correlation at the images' distances (terms summed until they fall below rtol of the total)."""
    from .stationary import matern_function

    total, shell = 0.0, 0
    while True:
        pts = [n for n in itertools.product(range(-shell, shell + 1), repeat=dim) if max(map(abs, n)) == shell]
        part = float(matern_function(box * np.linalg.norm(np.asarray(pts, dtype=float), axis=1), length, nu).sum())
        total += part
        if shell > 0 and part <= rtol * total:
            return total
        shell += 1


@functools.lru_cache(maxsize=256)
def resolution_error(h, order, dim, length, nu, max_dofs) -> float:
    """The largest relative error of the exact pointwise variance of the discrete Matérn field (H1, element size h,
    ``order``) on a periodic structured box, at the points of one cell, against the continuous field's."""
    import ngsolve as ngs
    import torch
    from ngsolve.meshes import Make1DMesh, MakeStructured2DMesh, MakeStructured3DMesh

    from .gaussian import Gaussian
    from .specs import matern
    from .stationary import _Affine1, _Affine2, _Affine3, _variance_scale, basis_at

    per_cell = order**dim  # H1 DoFs per cell of a periodic structured mesh (order^dim)
    cells = max(4, int(math.ceil(2.0 * length / h)))
    cells = max(4, min(cells, int((max_dofs / per_cell) ** (1.0 / dim))))
    box = cells * h
    if dim == 1:
        mesh = Make1DMesh(cells, mapping=_Affine1(0.0, box), periodic=True)
    elif dim == 2:
        mesh = MakeStructured2DMesh(quads=False, nx=cells, ny=cells, periodic_x=True, periodic_y=True,
                                    mapping=_Affine2([0.0, 0.0], [box, box]))
    else:
        mesh = MakeStructured3DMesh(hexes=False, nx=cells, ny=cells, nz=cells, periodic_x=True, periodic_y=True,
                                    periodic_z=True, mapping=_Affine3([0.0] * 3, [box] * 3))
    fes = ngs.Periodic(ngs.H1(mesh, order=order))
    kappa = math.sqrt(8.0 * nu) / length
    g = Gaussian(fes, matern(kappa, nu + dim / 2, _variance_scale(dim, length, nu)), dtype="float64", seed=0)
    # points of one cell (in cell units): a vertex, the midpoints of its axis edges, its centre, and points inside
    # its simplices (whichever diagonal splits it)
    unit = [np.zeros(dim), *(0.5 * np.eye(dim)), np.full(dim, 0.5)]
    unit += [np.array(q) for q in sorted(set(itertools.permutations([1 / 3, 2 / 3, 0.5][:dim])))]
    pts = h * (1.0 + np.array(unit))  # the cell [h, 2h]^d
    e = basis_at(fes, pts)[:, g.space.free].toarray()
    lt = g.apply_t(torch.from_numpy(np.ascontiguousarray(e.T))).numpy()
    var = (lt * lt).sum(axis=0)
    exact = periodised_variance(box, length, nu, dim)
    return float(np.abs(var / exact - 1.0).max())


DEFAULT = "default"  # resolution_tol= not given: RESOLUTION_TOL


def check_resolution(fes, length, nu, h=None, tol=DEFAULT):
    """Raise a ValueError if the space's elements (size ``h``, default the median edge length) do not resolve a
    Matérn field of correlation length ``length`` and smoothness ``nu``: the relative error of its pointwise variance
    on a periodic box with these elements exceeds ``tol`` (default ``RESOLUTION_TOL``; None: no check). Returns the
    error (None without a check)."""
    tol = RESOLUTION_TOL if tol == DEFAULT else tol
    if tol is None:
        return None
    from .stationary import _median_edge

    h = _median_edge(fes.mesh) if h is None else float(h)
    order, dim = max(1, int(fes.globalorder)), fes.mesh.dim
    budget = max(int(fes.ndof), 64 * order**dim)
    err = resolution_error(round(h, 12), order, dim, float(length), float(nu), budget)
    if err > tol:
        fine = _passing(h, order, dim, length, nu, tol, budget)
        raise ValueError(
            f"the mesh does not resolve the correlation length {length:g} (nu = {nu:g}): with elements of size "
            f"{h:.3g} and order {order} the field's pointwise variance is off by {err:.1%} (tolerance {tol:.1%}). "
            + (f"Elements of size {fine:.3g} would pass; " if fine else "")
            + "refine the mesh, raise the order or the length, or pass resolution_tol= (None: no check)")
    return err


def _passing(h, order, dim, length, nu, tol, budget):
    """The first of h / 2^(k/2), k = 1..6, that passes the check (None if none does)."""
    for k in range(1, 7):
        hk = h * 2 ** (-k / 2)
        if resolution_error(round(hk, 12), order, dim, float(length), float(nu), budget) <= tol:
            return hk
    return None
