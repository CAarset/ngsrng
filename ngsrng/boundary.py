"""Sampling random finite element functions *on a mesh boundary*.

Why restricting the coefficients is the right thing to do
--------------------------------------------------------
The trace of a finite element function on a boundary (or interface, or patch) ``Gamma`` is a **linear map on the
coefficient vector**. On a straight-sided mesh, for a Lagrange space of any order, it is exactly the restriction to
the DoFs whose basis function has a non-zero trace,

    ``coeffs(u_h|Gamma) = R u``,      ``R`` = row selection on :func:`boundary_dofs`.

Facet-interior, element-interior (bubble) and hierarchical interior DoFs drop out because their trace vanishes.
Two consequences drive this module.

*Sampling then restricting is exact, for every family.* A Gaussian pushes forward linearly,
``R u ~ N(R m, R Sigma_h R^T)``, and every non-Gaussian family in the library is built from a Gaussian by a
*pointwise* map (copula ``f = F^{-1}(Phi(z))``, scale mixtures, quadratic functionals, expansions) or by an affine
map, and pointwise maps commute with the trace. So restricting the DoFs of a sample gives an exact sample of the
*marginal law of the boundary DoFs* -- no approximation, no extra parameter, no mesh-dependent correction.

*The covariance you want is a principal block of the coefficient covariance.* With
``Sigma_h = M^{-1} K_C M^{-1}`` (see :mod:`ngsrng.covariance`), the boundary covariance is
``Sigma_GammaGamma = (Sigma_h)_{B,B}``. A boundary mass matrix is *not* a covariance: ``int_Gamma phi_i phi_j dS``
is the Gram matrix of the trace space, useful here only to identify the trace DoFs and to measure the spectrum of
the trace operator (see :meth:`BoundarySampler.trace_eigenvalues`).

The Dirichlet-to-Neumann identity
---------------------------------
For a PDE precision with ``power = 1`` the block has a closed form. Let ``K`` be the assembled stiffness on the free
DoFs, partitioned into ``B`` (on ``Gamma``) and ``I`` (the rest). Then

    ``Sigma_BB = S^{-1}``,   ``S = K_BB - K_BI K_II^{-1} K_IB``   (discrete Dirichlet-to-Neumann / Steklov-Poincare),
    ``Sigma_IB Sigma_BB^{-1} = -K_II^{-1} K_IB``                  (the ``K``-harmonic lifting),
    ``Sigma_II - Sigma_IB Sigma_BB^{-1} Sigma_BI = K_II^{-1}``    (homogeneous Dirichlet covariance).

That is the classical Markov (Wiener) decomposition of a Gaussian free field: given its boundary values, the field
is the harmonic lifting plus an independent field with homogeneous data. ``power = 1`` is the only Markov case, but
for any power the same *regression* quantities ``Sigma_BB`` and ``P = Sigma_{.,B} Sigma_BB^{-1}`` are what a
boundary sampler needs, and both come from ``n_B`` applications of ``Sigma_h`` (matrix-free, so also for fractional
powers, posteriors and GPU sources).

Restricting and conditioning
----------------------------
``bnd="name"``
    Sample, then keep the trace (DoFs off ``Gamma`` zeroed). The result is a finite element function supported on
    ``Gamma``, i.e. a usable random Dirichlet datum, and its law is the exact marginal for every family.
``bnd="name", bnd_values=g``
    The whole field conditioned on its trace being ``g``: ``u* = u - P (R u - g)``. Since ``(u, Ru)`` is jointly
    Gaussian and ``Cov(u - P R u, R u) = Sigma_{.,B} - P Sigma_BB = 0``, ``u - P R u`` is independent of the trace,
    and ``u*`` is exact hard Dirichlet conditioning (the Matheron / Gibbs form). There is no mode with a *drawn*
    ``g``: conditioning on an independent draw of the trace marginal reproduces the law of a plain draw. ``P`` is
    never formed: for a sparse precision ``Q`` (power 1, assembled precisions) ``P d`` is the discrete harmonic
    lifting ``(d, -Q_II^-1 Q_IB d)``, one sparse solve; otherwise it is ``Sigma E Sigma_BB^-1 d``, one application
    of ``Sigma_h`` plus a Cholesky solve with the ``n_B x n_B`` block. For copula families the latent Gaussian must
    be conditioned before the pointwise map, so only restriction is offered there.

Regularity
----------
``Sigma_BB`` is a principal block of an SPD matrix and always exists: the discrete trace is bounded regardless of
the field's regularity. Continuum trace-class-ness is a different question. For ``C = A^-power`` the trace
covariance ``gamma C gamma*`` is a pseudo-differential operator of order ``-(2 power - 1)`` on the
``(d-1)``-dimensional boundary (Seeley, Grubb), with eigenvalues ``~ j^{-(2 power - 1)/(d-1)}``: trace class exactly
when ``2 power > d`` -- the same threshold the bulk point-value guard uses (``E|u(x)|^2 < inf`` iff
``delta_x in H^{-power}``, i.e. ``2 power > d``). So the sampler needs no new regularity
condition, but whether the returned data converges to an ``L^2(Gamma)`` function as ``h -> 0`` does;
:meth:`BoundarySampler.trace_eigenvalues` measures the exponent.

Reverting this prototype: delete this file, the lines tagged ``[bnd]`` in ``ngsrng/generator.py`` and
``tests/test_boundary.py``; nothing else in the library calls into it.
"""

import ngsolve as ngs
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .backend import Rebuildable, available_memory
from .covariance import as_covariance, block_columns
from .space import SpaceInfo, assemble, to_scipy

try:  # torch rows arrive from the GPU engine; the algebra below is written to run in either backend
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None

__all__ = ["boundary_mass", "boundary_dofs", "boundary_covariance", "BoundarySampler", "attach_boundary"]

_TOL = 1e-13
# Largest dense lifting P (n_free x n_bnd entries) kept for the correction of a general source: below it a product
# with P is cheaper than an application of Sigma_h; above it, P (which grows like N^((2d-1)/d) under refinement, unlike
# a fixed low rank) is never formed
DENSE_LIFTING_VALUES = 2**24
# Values per dense temporary of the conditioning step. Samples are conditioned in blocks of about this many values,
# independently of how the caller chunks them: blocks of 2^20 values (the CPU in-place chunk) held 4 samples of a
# 263k-dof field, too few right-hand sides to keep the threaded solves busy (conditioning cost 7.5 plain draws).
BLOCK_VALUES = 2**24


def _index(arr, ix):
    """A dof index list usable on ``arr`` (numpy arrays take int arrays, torch wants tensors)."""
    if _torch is not None and isinstance(arr, _torch.Tensor):
        return _torch.as_tensor(np.asarray(ix), dtype=_torch.long, device=arr.device)
    return np.asarray(ix)


def _vector(x):
    """Mean/loc values as a flat float64 array of length ndof (GridFunctions and tensors accepted)."""
    if x is None:
        return None
    if isinstance(x, ngs.GridFunction):
        return np.array(x.vec.FV().NumPy(), dtype=np.float64)
    return np.asarray(x.cpu().numpy() if _torch is not None and isinstance(x, _torch.Tensor) else x,
                      dtype=np.float64).ravel()


def boundary_mass(fes, name):
    """Assembled ``int_Gamma phi_i phi_j ds`` (CSR, full ``ndof`` size) for the boundary named ``name``.

    Its diagonal is ``||gamma phi_i||^2_{L^2(Gamma)}``: zero exactly for the DoFs whose basis function vanishes on
    ``Gamma``. That is the criterion used to find the trace DoFs -- independent of the space's internal DoF
    numbering, valid for any Lagrange order and any straight-sided facet geometry.
    """
    kind = type(fes).__name__
    if "L2" in kind.upper() or "DG" in kind.upper():
        raise NotImplementedError(f"{kind} has no nodal trace: an L2/DG function has no boundary values to sample. "
                                  "Use a continuous space (H1), or sample the boundary data on a separate H1 space.")
    u, v = fes.TnT()
    a = ngs.BilinearForm(fes)
    a += ngs.InnerProduct(u, v) * ngs.ds(name)
    a.Assemble()
    return to_scipy(a.mat)


def boundary_dofs(fes, name, free_only=True):
    """DoF numbers whose basis function has a non-zero trace on the boundary named ``name``.

    These carry the Dirichlet data: the nodes on ``Gamma`` plus, for order >= 2, the facet DoFs lying on ``Gamma``.
    ``free_only=False`` also returns DoFs that the space fixes through ``dirichlet=``.
    """
    diag = boundary_mass(fes, name).diagonal()
    dofs = np.flatnonzero(diag > _TOL * float(np.abs(diag).max(initial=0.0)))  # relative: unit independent
    if not free_only:
        return dofs
    free = np.asarray(fes.FreeDofs(), dtype=bool)
    return dofs[free[dofs]]


def boundary_covariance(source, name, fes=None):
    """Covariance of the boundary DoFs, ``(Sigma_h)_{B,B}``, of any Gaussian-type source on the boundary ``name``.

    Accepts anything :func:`ngsrng.covariance.as_covariance` understands: a :class:`~ngsrng.Gaussian`, a low-rank or
    operator covariance, a factor, or a posterior. For a ``power = 1`` precision this matrix is the inverse of the
    assembled discrete Dirichlet-to-Neumann map; costs ``n_B`` applications of ``Sigma_h``.
    """
    return BoundarySampler(source, name, fes=fes).sigma_bb


class BoundarySampler(Rebuildable):
    """Boundary (trace) machinery for one Gaussian-type source on one boundary.

    Only applications of the coefficient covariance are used, so it is matrix-free and works for fractional powers,
    for posteriors, and for GPU-backed sources.

    Attributes
    ----------
    dofs : (n_bnd,) int
        The global DoF numbers carrying the trace (all free).
    sigma_bb : (n_bnd, n_bnd) ndarray
        Covariance of the boundary DoFs. Symmetric positive definite.
    lifting : (nfree, n_bnd) ndarray
        ``Sigma_{.,B} Sigma_BB^{-1}`` on the free DoFs, with identity rows at ``B``: the regression of the
        coefficients on the boundary data. For ``power = 1`` its interior rows are ``-K_II^{-1} K_IB``, the
        ``K``-harmonic lifting of the Dirichlet data.
    """

    # pickling: the per-device solvers and the interior factor are made again on first use
    _transient = ("_device_ops", "_qblocks")

    def _restore(self):
        self._device_ops, self._qblocks = {}, None

    def __init__(self, source, name, fes=None):
        if name not in set(str(b) for b in _boundaries(source, fes)):
            raise ValueError(f"'{name}' is not a boundary name of this mesh "
                             f"(available: {sorted(str(b) for b in _boundaries(source, fes))})")
        self.source = source
        self.fes = fes if fes is not None else source.space.fes
        self.space = SpaceInfo(self.fes)
        self.name = str(name)
        self.dofs = boundary_dofs(self.fes, self.name)
        if self.dofs.size == 0:
            raise ValueError(f"no free DoF has a trace on '{self.name}': the space fixes that boundary through "
                             "dirichlet=, so its boundary values are not random. Build the space without a "
                             "Dirichlet condition on that part (keep one elsewhere) and sample its boundary data.")
        self.n_bnd = int(self.dofs.size)
        free = self.space.free
        self._pos = np.searchsorted(free, self.dofs)
        self._free = np.zeros(free.size, dtype=bool)
        self._free[self._pos] = True
        # restriction keeps every DoF with a trace, the Dirichlet ones too: zeroing a fixed trace DoF would change the
        # boundary values (and break the range of a Bernstein-built function)
        self._off = np.setdiff1d(np.arange(self.fes.ndof), boundary_dofs(self.fes, self.name, free_only=False))
        self._int_pos = np.flatnonzero(~self._free)  # free dofs off Gamma: positions among the free dofs ...
        self._int_dofs = free[self._int_pos]          # ... and dof numbers; the lifting acts only there
        self._device_ops = {}      # per device: the correction's operators for torch rows
        self._chol = None          # Cholesky factor of sigma_bb (correction of general sources)
        self._residual = None
        self._src = as_covariance(source, self.fes)
        self._sigma_bb = None      # n_bnd applications of Sigma_h, on first use: "restrict" draws never need it
        self._cols = None          # Sigma_h[:, B] (free rows), kept only when it is at most DENSE_LIFTING_VALUES
        self._dense = None         # the corresponding small dense lifting, used by the correction
        self._regression = None    # dense lifting and P as attributes: diagnostics, on request only
        # Sigma^-1 assembled and sparse (power 1, assembled precisions): the lifting is the sparse harmonic one
        self._sparse_form = _sparse_precision(source)
        self._q = None
        self._qblocks = None
        self.info = {"name": self.name, "n_bnd": self.n_bnd, "dofs": self.dofs.copy()}

    @property
    def sigma_bb(self):
        """Covariance of the boundary DoFs (n_bnd x n_bnd): ``n_bnd`` applications of ``Sigma_h``, ``CHUNK`` at a
        time, keeping only the boundary rows (dense ``n_bnd^2``, never ``n_free x n_bnd``)."""
        if self._sigma_bb is None:
            need, avail = 24.0 * self.n_bnd**2, available_memory()
            if avail is not None and need > avail:
                raise MemoryError(f"the boundary covariance block of '{self.name}' ({self.n_bnd} trace DoFs) needs "
                                  f"~{need / 2**30:.1f} GiB, {avail / 2**30:.1f} GiB available")
            block = np.empty((self.n_bnd, self.n_bnd))
            small = self.space.nfree * self.n_bnd <= DENSE_LIFTING_VALUES
            cols = np.empty((self.space.nfree, self.n_bnd)) if small else None
            step = block_columns(self.space.nfree, getattr(self._src, "nrand", 0) or 0)
            for j0 in range(0, self.n_bnd, step):
                j1 = min(self.n_bnd, j0 + step)
                e = np.zeros((self.space.nfree, j1 - j0))
                e[self._pos[j0:j1], np.arange(j1 - j0)] = 1.0
                c = np.asarray(self._src.sigma(e), dtype=np.float64)
                block[:, j0:j1] = c[self._pos]
                if small:
                    cols[:, j0:j1] = c
            self._sigma_bb = 0.5 * (block + block.T)
            self._cols = cols
        return self._sigma_bb

    def _regress(self):
        """Dense lifting ``Sigma_{.,B} Sigma_BB^{-1}`` (n_free x n_bnd) and P (ndof x n_bnd): diagnostics only;
        sampling applies them matrix-free (:meth:`_correction`)."""
        if self._regression is None:
            _dense_guard(self.fes.ndof, self.n_bnd, "the lifting P")
            eye = np.eye(self.n_bnd)
            lifting = self._correction(eye)[:, self.space.free].T  # column j: P e_j
            P = np.zeros((self.fes.ndof, self.n_bnd))
            P[self.space.free, :] = lifting
            self._regression = (lifting, P)
        return self._regression

    @property
    def lifting(self):
        """``Sigma_{.,B} Sigma_BB^{-1}`` on the free DoFs (n_free x n_bnd, dense; a diagnostic), identity rows at
        ``B``."""
        return self._regress()[0]

    @property
    def P(self):
        """The lifting on all DoFs (ndof x n_bnd, dense; a diagnostic; zero rows at fixed DoFs)."""
        return self._regress()[1]

    def _correction(self, d):
        """Rows ``(P d_k)^T`` (k x ndof, float64) for boundary residuals ``d`` (k x n_bnd), without forming P: ``d``
        on the trace DoFs, :meth:`_interior_lift` on the other free DoFs, zero on fixed DoFs."""
        d = np.asarray(d, dtype=np.float64).reshape(-1, self.n_bnd)
        out = np.zeros((d.shape[0], self.fes.ndof))
        out[:, self._int_dofs] = self._interior_lift(d)
        out[:, self.dofs] = d
        return out

    def _interior_lift(self, d):
        """``P d_k`` on the free DoFs off ``Gamma``, as rows (k x n_int, float64), for residuals ``d`` (k x n_bnd).

        Sparse precision ``Q`` (power 1, assembled precisions): ``-Q_II^-1 Q_IB d``, one sparse solve with the interior
        block (the discrete harmonic lifting). Otherwise ``Sigma E Sigma_BB^-1 d``: a Cholesky solve with the boundary
        block and one application of ``Sigma_h`` -- or, for a small problem, a product with the stored dense lifting.
        """
        if self._sparse_form is not None:
            qii, qib = self._sparse_blocks()
            return -np.asarray(qii.solve(qib @ d.T)).reshape(-1, d.shape[0]).T
        self._factor_bb()
        if self._cols is not None:  # small problem: the dense lifting is cheaper than an application of Sigma_h
            return d @ self._dense_lifting()[self._int_pos].T
        y = np.zeros((self.space.nfree, d.shape[0]))
        y[self._pos] = sla.cho_solve(self._chol, d.T)
        return np.asarray(self._src.sigma(y), dtype=np.float64)[self._int_pos].T

    def _interior_lift_torch(self, d):
        """:meth:`_interior_lift` for a float64 torch tensor ``d``, computed on its device: the interior block solved by
        cuDSS on NVIDIA GPUs or AMG-preconditioned CG elsewhere (sparse precisions), the source's own pipeline
        (general Gaussians on the same device), else through the host."""
        if d.device.type == "cpu":
            return _torch.from_numpy(self._interior_lift(d.numpy()))
        ops = self._device(d.device)
        if ops is None:
            return _torch.from_numpy(self._interior_lift(d.cpu().numpy())).to(d.device)
        kind, a, b = ops
        if kind == "sparse":  # a: cuDSS factor of Q_II, b: Q_IB
            return -a.apply(b @ d.T).T
        if kind == "dense":  # a: interior rows of the lifting
            return d @ a.T
        y = _torch.zeros(self.space.nfree, d.shape[0], device=d.device, dtype=_torch.float64)  # a: chol(Sigma_BB)
        y[_torch.as_tensor(self._pos, device=d.device)] = _torch.cholesky_solve(d.T, a)
        sig = b.apply(b.apply_t(y))  # b: the source Gaussian on this device, Sigma = L L^T
        return sig[_torch.as_tensor(self._int_pos, device=d.device)].T.double()

    def _device(self, dev):
        key = str(dev)
        if key not in self._device_ops:
            self._device_ops[key] = self._build_device_ops(dev)
        return self._device_ops[key]

    def _build_device_ops(self, dev):
        from .backend import is_allocation_failure, sparse_to_torch

        f64 = _torch.float64
        if self._sparse_form is not None:
            from .solvers import CudssDirect, TorchAMGPCG, cudss_available, pyamg_available

            q, ii, bb = self._q_free(), self._int_pos, self._pos
            qib = sparse_to_torch(q[ii][:, bb].tocsr(), dev, f64)
            if cudss_available():
                try:
                    return "sparse", CudssDirect(q[ii][:, ii], dev, f64), qib
                except Exception as err:  # the factor does not fit even after releasing torch's cache: AMG-PCG
                    if not is_allocation_failure(err) or not pyamg_available():
                        raise
            if pyamg_available():  # no GPU sparse direct solver (AMD): AMG-preconditioned CG, solved tightly
                return "sparse", TorchAMGPCG(q[ii][:, ii], dev, f64, tol=1e-12), qib
            return None
        self._factor_bb()
        if self._cols is not None:
            return "dense", _torch.from_numpy(np.ascontiguousarray(self._dense_lifting()[self._int_pos])).to(dev), None
        from .backend import same_device, torch_device_of

        g = self.source  # a sampler computing on this device (torch engine, low-rank model, Matheron posterior)
        if torch_device_of(g) is not None and same_device(torch_device_of(g), dev) and hasattr(g, "apply_t"):
            return "sigma", _torch.linalg.cholesky(_torch.from_numpy(self.sigma_bb).to(dev)), g
        return None

    def _factor_bb(self):
        if self._chol is None:
            self._chol = sla.cho_factor(self.sigma_bb, lower=True)

    def _dense_lifting(self):
        """Sigma E Sigma_BB^-1 on the free DoFs (n_free x n_bnd), stored only for small problems."""
        if self._dense is None:
            self._dense = sla.cho_solve(self._chol, self._cols.T).T
            self._dense[self._pos] = np.eye(self.n_bnd)  # R P = I exactly
        return self._dense

    def _q_free(self):
        if self._q is None:
            self._q = self.space.assemble_free(self._sparse_form).tocsr()  # Sigma^-1 up to the scale factor
        return self._q

    def _sparse_blocks(self):
        if self._qblocks is None:
            from .distributions.bernstein import SymmetricFactor  # NGSolve sparse Cholesky, threaded right-hand sides

            q, ii = self._q_free(), self._int_pos
            self._qblocks = SymmetricFactor(q[ii][:, ii]), q[ii][:, self._pos].tocsr()
        return self._qblocks

    # ----------------------------------------------------------------------- diagnostics

    @property
    def residual_covariance(self):
        """Cov(u_I | u_B) on the interior free DoFs: the Schur complement of ``Sigma_BB`` in ``Sigma_h``.

        For a ``power = 1`` precision this is ``K_II^{-1}``, the covariance of the field with homogeneous Dirichlet
        data on ``Gamma`` (Markov property): assembled and factorised with sparse linear algebra from the
        precision's own bilinear form. Denser precisions have no sparse conditional covariance (``Sigma_h`` is dense
        there); sampling never needs it, only ``sigma_bb`` and ``lifting``.
        """
        if self._residual is None:
            spec = getattr(self.source, "spec", None)
            form = getattr(spec, "form", None)
            if form is None or getattr(spec, "power", None) != 1:
                raise NotImplementedError(
                    "residual_covariance is the interior precision block K_II^-1 of a power=1 precision; this source "
                    f"is a {type(self.source).__name__} built from {type(spec).__name__}. The sampler itself only "
                    "uses sigma_bb and lifting, which are available matrix-free for any source.")
            K = assemble(self.fes, form)[self.space.free][:, self.space.free]
            ii = np.flatnonzero(~self._free)
            _dense_guard(ii.size, ii.size, "residual_covariance (K_II^-1)")
            lu = spla.splu(sp.csc_matrix(K[ii][:, ii]))
            self._residual = np.asarray(lu.solve(np.eye(ii.size)))  # dense by request: a verification diagnostic
        return self._residual

    def dtn(self, form):
        """Assembled discrete Dirichlet-to-Neumann map ``S = K_BB - K_BI K_II^{-1} K_IB``.

        Provided to check ``sigma_bb == S^{-1}`` for a ``power = 1`` precision independently of this module;
        ``form(u, v)`` is the bilinear form of the precision (e.g. ``InnerProduct(grad(u), grad(v)) * dx``).
        """
        K = sp.csc_matrix(assemble(self.fes, form)[self.space.free][:, self.space.free])
        b, i = self._free, ~self._free
        _dense_guard(int(i.sum()), self.n_bnd, "dtn (K_II^-1 K_IB)")
        lu = spla.splu(K[i][:, i].tocsc())
        # SuperLU takes a dense right-hand side: it has n_bnd columns, the small block of the problem, while the
        # n_int x n_int interior factorisation stays sparse, so this is not a dense solve.
        X = lu.solve(np.asarray(K[i][:, b].todense()))
        return sp.csr_matrix(np.asarray(K[b][:, b].todense()) - np.asarray(sp.csr_matrix(K[b][:, i]) @ X))

    def trace_eigenvalues(self, n=None):
        """Eigenvalues of the discrete trace covariance operator, i.e. of ``Sigma_BB M_Gamma``.

        The trace ``sum_i c_i phi_i`` with ``Cov(c) = Sigma_BB`` has the covariance operator
        ``f -> sum_ij phi_i (Sigma_BB)_ij (phi_j, f)_Gamma``; on the trace space its matrix is ``Sigma_BB M_Gamma``, so
        its eigenvalues are those of the symmetric-definite pencil ``(M_Gamma Sigma_BB M_Gamma, M_Gamma)`` and sum to
        ``tr(M_Gamma Sigma_BB) = E ||gamma u||^2``. They approximate the eigenvalues of the continuum trace covariance
        ``gamma C gamma*``, which decay like ``j^{-(2 power - 1)/(d-1)}``. The pencil is ``n_bnd`` x ``n_bnd``.
        """
        M = boundary_mass(self.fes, self.name).tocsc()[self.dofs][:, self.dofs].toarray()
        A = M @ np.asarray(self.sigma_bb) @ M
        A = 0.5 * (A + A.T)
        n_bnd = A.shape[0]
        if n is None:
            # dense by request: the full spectrum of an n_bnd x n_bnd symmetric-definite pencil. Do not use on a
            # boundary mesh fine enough that n_bnd is large - pass n to get the top n by ARPACK instead.
            return np.sort(sla.eigh(A, M, eigvals_only=True))[::-1]
        k = min(int(n), n_bnd - 1)
        if k < 1:
            return np.array([A[0, 0] / M[0, 0]])
        w = spla.eigsh(A, M=sp.csc_matrix(M), k=k, which="LA", return_eigenvectors=False)
        return np.sort(w)[::-1]

    # ----------------------------------------------------------------------- sampling

    def lift(self, loc, values):
        """Full coefficient vector of ``loc`` plus the lifting of the boundary data ``values`` (the conditional mean).

        For ``power = 1`` and ``loc = 0`` this is the ``K``-harmonic extension of the data into the domain.
        """
        base = np.zeros(self.fes.ndof, dtype=np.float64) if (mean := _vector(loc)) is None else mean.copy()
        return base - self._correction(base[self.dofs] - np.asarray(values, dtype=np.float64).ravel())[0]

    def restrict(self, rows):
        """Zero every DoF that carries no trace (in place); trace DoFs, free or fixed by ``dirichlet=``, keep their
        values. The exact marginal of the boundary DoFs."""
        off = _index(rows, self._off)
        if rows.ndim == 1:
            rows[off] = 0
        else:
            rows[:, off] = 0
        return rows

    def condition(self, rows, values):
        """Condition the samples ``rows`` (samples by DoFs, in place) on their trace being ``values``: ``rows``
        becomes an exact draw of the field given ``u_h|Gamma = values`` (one set of values, or one per sample).

        ``u - P (R u - g)``, with P never formed (:meth:`_interior_lift`), in blocks of about ``BLOCK_VALUES`` values;
        torch rows are conditioned on their own device.
        """
        one = rows.ndim == 1
        r = rows.reshape(1, -1) if one else rows
        g = np.asarray(values, dtype=np.float64).reshape(-1, self.n_bnd)
        if g.shape[0] not in (1, r.shape[0]):
            raise ValueError(f"bnd_values has {g.shape[0]} rows for {r.shape[0]} samples")
        torch_rows = _torch is not None and isinstance(r, _torch.Tensor)
        if torch_rows:
            g = _torch.from_numpy(np.ascontiguousarray(g)).to(r.device)
            dofs, interior = _index(r, self.dofs), _index(r, self._int_dofs)
        tall = max(self.fes.ndof, 0 if self._sparse_form is not None else getattr(self._src, "nrand", 0) or 0)
        step = max(1, BLOCK_VALUES // tall)
        for k0 in range(0, r.shape[0], step):
            rb = r[k0:k0 + step]
            gb = g if g.shape[0] == 1 else g[k0:k0 + step]
            if torch_rows:
                lift = self._interior_lift_torch(rb[:, dofs].double() - gb)
                rb[:, interior] -= lift.to(rb.dtype)
                rb[:, dofs] = gb.to(rb.dtype).expand(rb.shape[0], -1)
            else:
                lift = self._interior_lift(np.asarray(rb[:, self.dofs], dtype=np.float64) - gb)
                rb[:, self._int_dofs] -= lift.astype(rb.dtype, copy=False)
                rb[:, self.dofs] = gb  # exact by construction (R P = I); written for bit-level agreement with the data
        return rows


def _latent(field):
    """``(Gaussian, affine)`` behind a built field: the Gaussian whose DoFs the field's samples are built from.

    ``affine`` says whether the field is an affine function of that Gaussian (loc + scale g), which is what
    conditioning (``bnd_values=``) needs; copula families, mixtures and quadratic fields are nonlinear in it.
    """
    points = getattr(field, "points", None)  # GaussianAtPoints: the latent Gaussian of a pointwise family
    if points is not None and hasattr(points, "gauss"):
        return points.gauss, False
    gauss = getattr(field, "gauss", None)
    if gauss is not None:
        return gauss, type(field).__name__ == "_NormalField"
    if hasattr(field, "space") and hasattr(field, "apply"):  # a bare Gaussian used as a field
        return field, True
    raise NotImplementedError(f"bnd= is not implemented for {type(field).__name__}: it is not built on a Gaussian "
                              "whose DoFs are the sample's finite element coefficients")


def attach_boundary(field, name, values=None, cache=None):
    """Attach the ``bnd=`` specification of a freshly built field (called by :class:`ngsrng.Generator`).

    Sets ``field.bnd`` (what :meth:`ngsrng.Generator._apply_bnd` needs) and adds the boundary block to
    ``field.info``: without ``values`` the draws are restricted to the boundary, with ``values`` the field is
    conditioned on that trace. Raises for the combinations that have no boundary interpretation rather than silently
    changing what the user asked for.
    """
    if getattr(field, "evaluation", None) is not None or getattr(field, "coords", None) is not None:
        raise ValueError(f'bnd="{name}" restricts finite element DoFs; this field delivers values at given points. '
                         "Drop points= to sample the field itself.")
    rep = getattr(field, "representation", None)
    if rep not in (None, "projection", "bernstein"):
        raise ValueError(f'bnd="{name}" restricts finite element DoFs; representation="{rep}" delivers values at '
                         "quadrature points, which have no DoF trace. Use representation='bernstein' (or "
                         "'projection'), which returns coefficients.")
    gauss, affine = _latent(field)
    # ``cache``: {(id(gauss), name): (gauss, sampler)} -- a sampler's set-up (trace DoFs, and for conditioning the
    # n_B applications of Sigma behind Sigma_BB) depends on the law only, so fields of one Gaussian share it
    hit = None if cache is None else cache.get((id(gauss), str(name)))
    sampler = hit[1] if hit is not None else BoundarySampler(gauss, name)
    if cache is not None:
        cache[(id(gauss), str(name))] = (gauss, sampler)
    if values is not None:
        if not affine:
            raise NotImplementedError(
                f"bnd_values conditions an affine (Gaussian) field on its trace; {type(field).__name__} is a "
                "nonlinear function of its Gaussian. Condition the Gaussian itself and build the family on the "
                "conditioned Gaussian, or use bnd= alone (restriction, exact for every family).")
        values = _vector(values)
        if values.size == sampler.fes.ndof:  # a GridFunction (or coefficient vector) of the boundary data
            values = values[sampler.dofs]
        if values.size != sampler.n_bnd:
            raise ValueError(f"bnd_values has {values.size} entries; the boundary '{name}' has {sampler.n_bnd} DoFs "
                             "(in increasing DoF order)")
    field.bnd = {"sampler": sampler, "values": values,
                 "info": {**sampler.info, "mode": "restrict" if values is None else "condition"}}


def _dense_guard(rows, cols, what):
    """Raise MemoryError before a dense diagnostic of ``rows x cols`` float64 entries (with a working copy) that does
    not fit in the available memory."""
    need, avail = 16.0 * rows * cols, available_memory()
    if avail is not None and need > avail:
        raise MemoryError(f"{what} is a dense {rows} x {cols} matrix (~{need / 2**30:.1f} GiB with its working copy), "
                          f"{avail / 2**30:.1f} GiB available; it is a diagnostic for small problems")


def _sparse_precision(source):
    """The bilinear form whose matrix is ``Sigma^-1`` up to a scale factor -- a prior :class:`~ngsrng.Gaussian` with
    a ``power = 1`` PDE precision or an assembled precision -- or None (posteriors, other powers, low rank)."""
    from .gaussian import Gaussian
    from .specs import AssembledPrecision, PDEPrecision

    if not isinstance(source, Gaussian) or source._maps or source._post_maps:
        return None
    spec = source.spec
    if isinstance(spec, AssembledPrecision) or (isinstance(spec, PDEPrecision) and spec.power == 1):
        return spec.form
    return None


def _boundaries(source, fes=None):
    fes = fes if fes is not None else getattr(getattr(source, "space", None), "fes", None)
    if fes is None:
        return ()
    return fes.mesh.GetBoundaries()
