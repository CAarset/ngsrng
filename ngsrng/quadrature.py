"""Quadrature-point representation of FE functions: the bridge for white noise and pointwise (non-Gaussian) operations.

For a space V_h and a rule {x_q, w_q} (the points of an NGSolve IntegrationRuleSpace), B (ndof x nq) has entries
B_iq = w_q phi_i(x_q). Then:

- the values of f_h = sum c_i phi_i at the points are B^T c / w,
- int g phi_i dx ~ (B g)_i for point values g, so the L2 projection of point values into V_h is M^-1 B g,
- b = B W^-1/2 xi (xi iid standard normal) is FE white noise: Cov(b) = B W^-1 B^T = M whenever the rule integrates
  phi_i phi_j exactly.

``order`` is that of the IntegrationRuleSpace, whose rule is NGSolve's ``IntegrationRule(et, 2 * order)`` (exact for
polynomials of degree 2 order, on every element type). So order >= p already integrates phi_i phi_j exactly on affine
elements, and on curved elements NGSolve assembles M with the degree-2p rule, which order = p reproduces and order = 2p
does not: white noise (``noise.quadrature_noise``) uses order = p. The default here, order = 2p, is the point set of
the pointwise ("quadrature") representation of the distributions.
"""

from functools import cached_property

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
from ngsolve.comp import IntegrationRuleSpace

from .backend import Rebuildable
from .space import SpaceInfo, to_scipy


class QuadratureRule(Rebuildable):
    _transient = ("rule",)  # an NGSolve IntegrationRule: made again after unpickling

    def __init__(self, space: SpaceInfo, order: int | None = None, max_nnz: float = 4e8):
        self.space = space
        self.order = 2 * space.order if order is None else order
        mesh = space.mesh
        if not space.single_element_type:
            raise NotImplementedError("quadrature representations currently need a single element type")
        self.irs = IntegrationRuleSpace(mesh, order=self.order)
        self.rule = self._make_rule()
        if self.irs.ndof != len(self.rule.points) * mesh.ne:
            raise RuntimeError("could not match the IntegrationRuleSpace points")
        nnz = mesh.ne * len(self.rule.points) * (space.broken.ndof / mesh.ne)
        if nnz > max_nnz:
            raise MemoryError(f"quadrature operator would have ~{nnz:.2e} nonzeros (limit {max_nnz:.0e})")
        # simd_evaluate=False: SIMD assembly gives wrong results for large rules (343-point tet rule, NGSolve 6.2.2608).
        fes = space.plain
        b = ngs.BilinearForm(trialspace=self.irs, testspace=fes)
        b += ngs.SymbolicBFI(self.irs.TrialFunction() * fes.TestFunction(), intrule=self.rule, simd_evaluate=False)
        lf = ngs.LinearForm(self.irs)
        lf += ngs.SymbolicLFI(self.irs.TestFunction(), intrule=self.rule, simd_evaluate=False)
        with ngs.TaskManager():
            b.Assemble()
            lf.Assemble()
        self.weights = lf.vec.FV().NumPy().copy()
        if (self.weights <= 0).any():
            raise RuntimeError("non-positive quadrature weights")
        self.b = to_scipy(b.mat)  # ndof x nq

    def _make_rule(self):
        # IntegrationRuleSpace uses IntegrationRule(et, 2 * order) for every element type, but does not report its 1D
        # rule.
        et = next(iter(self.space.mesh.Elements(ngs.VOL))).type
        return ngs.IntegrationRule(et, 2 * self.order) if et == ngs.ET.SEGM else self.irs.GetIntegrationRules()[et]

    def _restore(self):
        self.rule = self._make_rule()

    @property
    def npoints(self) -> int:
        return len(self.weights)

    @staticmethod
    def count(space: SpaceInfo, order: int | None = None) -> int | None:
        """Number of points of ``QuadratureRule(space, order)`` without building it (None: not supported)."""
        if not space.single_element_type:
            return None
        et = next(iter(space.mesh.Elements(ngs.VOL))).type
        order = 2 * space.order if order is None else order
        return space.mesh.ne * len(ngs.IntegrationRule(et, 2 * order).points)

    @cached_property
    def evaluation_full(self) -> sp.csr_matrix:
        """E (nq x ndof): point values of a function from all its coefficients."""
        return (sp.diags(1.0 / self.weights) @ self.b.T).tocsr()

    @cached_property
    def evaluation(self) -> sp.csr_matrix:
        """E (nq x nfree): point values of a function from its free-dof coefficients (Dirichlet part excluded)."""
        return (sp.diags(1.0 / self.weights) @ self.b[self.space.free].T).tocsr()

    def values(self, coeffs_full: np.ndarray) -> np.ndarray:
        """Point values (nq x k) of full coefficient vectors (ndof x k)."""
        out = self.b.T @ coeffs_full
        return out / (self.weights[:, None] if out.ndim == 2 else self.weights)

    def to_gridfunction(self, values: np.ndarray) -> ngs.GridFunction:
        """A GridFunction on the IntegrationRuleSpace holding the point values (usable as a coefficient in forms)."""
        gf = ngs.GridFunction(self.irs)
        gf.vec.FV().NumPy()[:] = values
        return gf

    def load(self, values: np.ndarray) -> np.ndarray:
        """int g phi_i dx ~ B g for point values g (ndof x k): the load vector of a point-valued function."""
        return self.b @ values

    @cached_property
    def points(self) -> np.ndarray:
        """(nq x dim) physical coordinates of the points, in IntegrationRuleSpace dof order."""
        dim = self.space.mesh.dim
        coords = []
        for xi in [ngs.x, ngs.y, ngs.z][:dim]:
            gf = ngs.GridFunction(self.irs)
            gf.Set(xi)  # on an IntegrationRuleSpace, Set evaluates at the points
            coords.append(gf.vec.FV().NumPy().copy())
        return np.column_stack(coords)
