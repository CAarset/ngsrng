"""Static information about an NGSolve finite element space, shared by all samplers."""

from dataclasses import dataclass, field
from functools import cached_property

import ngsolve as ngs
import ngsolve.comp as comp
import numpy as np
import scipy.sparse as sp


def to_scipy(mat) -> sp.csr_matrix:
    vals, cols, ptr = mat.CSR()
    return sp.csr_matrix((np.array(vals), np.array(cols), np.array(ptr)), shape=(mat.height, mat.width))


def assemble(fes, form) -> sp.csr_matrix:
    """Assemble ``form(u, v)`` (a SumOfIntegrals, measures included) on ``fes`` into a scipy matrix."""
    u, v = fes.TnT()
    a = ngs.BilinearForm(fes)
    a += form(u, v)
    with ngs.TaskManager():
        a.Assemble()
    return to_scipy(a.mat)


def mass_form(u, v):
    return ngs.InnerProduct(u, v) * ngs.dx


@dataclass
class SpaceInfo:
    """Wraps an FE space. All sampler vectors live on the free dofs (length ``nfree``)."""

    fes: ngs.FESpace
    free: np.ndarray = field(init=False)

    def __post_init__(self):
        self.free = np.flatnonzero(np.array(self.fes.FreeDofs(), dtype=bool))

    @property
    def mesh(self):
        return self.fes.mesh

    @property
    def dim(self) -> int:
        return self.fes.mesh.dim

    @property
    def ndof(self) -> int:
        return self.fes.ndof

    @property
    def nfree(self) -> int:
        return len(self.free)

    @property
    def order(self) -> int:
        return self.fes.globalorder

    @property
    def kind(self) -> str:
        return type(self.fes).__name__

    @cached_property
    def mass(self) -> sp.csr_matrix:
        return assemble(self.fes, mass_form)

    @cached_property
    def mass_free(self) -> sp.csr_matrix:
        return restrict(self.mass, self.free)

    def assemble_free(self, form) -> sp.csr_matrix:
        return restrict(assemble(self.fes, form), self.free)

    @cached_property
    def mass_is_diagonal(self) -> bool:
        m = self.mass_free.tocoo()
        off = m.row != m.col
        if not off.any():
            return True
        return np.abs(m.data[off]).max() <= 1e-13 * np.abs(m.diagonal()).max()

    @cached_property
    def single_element_type(self) -> bool:
        return len({el.type for el in self.mesh.Elements(ngs.VOL)}) == 1

    @cached_property
    def plain(self) -> ngs.FESpace:
        """Same space without ``dgjumps`` (identical dof numbering).

        Mixed bilinear forms with a dgjumps test space segfault in NGSolve 6.2.2608.
        """
        flags = self.fes.flags.ToDict()
        if not flags.pop("dgjumps", False):
            return self.fes
        twin = ngs.FESpace(self.fes.type, self.mesh, **flags)
        if twin.ndof != self.ndof:
            raise RuntimeError("could not build a dgjumps-free twin of the space")
        return twin

    @cached_property
    def broken(self) -> ngs.FESpace:
        return self.fes if self.kind == "L2" else comp.Discontinuous(self.fes)

    @cached_property
    def embedding(self) -> sp.csr_matrix:
        """0/1 matrix E (broken dofs x dofs) with E^T M_broken E = M.

        Built from element dof lists: Discontinuous(fes) uses the same local shape functions, in the same order, as
        fes. (ConvertOperator gives the same map, but via a costly projection that leaves ~1e-11 residue.)
        """
        if self.broken is self.fes:
            return sp.identity(self.ndof, format="csr")
        glob = [list(el.dofs) for el in self.fes.Elements(ngs.VOL)]
        cols = np.concatenate(glob)
        if self.single_element_type and self.broken.ndof == len(cols):
            rows = np.arange(len(cols))
        else:
            brok = [list(el.dofs) for el in self.broken.Elements(ngs.VOL)]
            if any(len(g) != len(b) for g, b in zip(glob, brok)):
                raise RuntimeError("element dof counts differ between the space and its broken counterpart")
            rows = np.concatenate(brok)
        valid = cols >= 0
        return sp.csr_matrix((np.ones(valid.sum()), (rows[valid], cols[valid])), shape=(self.broken.ndof, self.ndof))

    def describe(self) -> str:
        return (
            f"{self.kind}(order={self.order}) on {self.dim}D mesh, "
            f"{self.mesh.ne} elements, ndof={self.ndof}, free={self.nfree}"
        )


def restrict(mat: sp.spmatrix, idx: np.ndarray) -> sp.csr_matrix:
    mat = mat.tocsr()
    if len(idx) == mat.shape[0]:
        return mat
    return mat[idx][:, idx].tocsr()


def coefficient_columns(b, ndof) -> np.ndarray:
    """(ndof x r) coefficient columns from an array, sparse matrix, or list of GridFunctions/vectors."""
    if isinstance(b, (list, tuple)):
        return np.column_stack([g.vec.FV().NumPy() if hasattr(g, "vec") else np.asarray(g) for g in b])
    arr = np.asarray(b.toarray() if sp.issparse(b) else b, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != ndof:
        raise ValueError(f"B has {arr.shape[0]} rows, the space has {ndof} dofs")
    return arr


def coefficient_vector(x, space) -> np.ndarray:
    """Full coefficient vector from a GridFunction, a CoefficientFunction (interpolated with Set) or an array."""
    if isinstance(x, ngs.GridFunction):
        return x.vec.FV().NumPy().copy()
    if isinstance(x, ngs.CoefficientFunction):
        gf = ngs.GridFunction(space.fes)
        gf.Set(x)
        return gf.vec.FV().NumPy().copy()
    return np.asarray(x, dtype=float).copy()
