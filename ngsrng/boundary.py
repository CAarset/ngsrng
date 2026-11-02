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

The two modes of ``bnd="name"``
------------------------------
``"restrict"`` (default)
    Sample, then keep the trace (DoFs off ``Gamma`` zeroed). The result is a finite element function supported on
    ``Gamma``, i.e. a usable random Dirichlet datum, and its law is the exact marginal for every family.
``"field"``
    Keep the whole field but replace its trace by an independent draw ``g ~ N(m_B, Sigma_BB)``, or by prescribed
    data: ``u* = u - P (R u - g)``. Since ``(u, Ru)`` is jointly Gaussian and
    ``Cov(u - P R u, R u) = Sigma_{.,B} - P Sigma_BB = 0``, the pair ``(u*, R u*) = (u - P R u + P g, g)`` has the
    same joint law as ``(u, R u)``: the field remains an exact prior draw whose trace is an independent draw of the
    boundary marginal. With prescribed ``g`` it is exact hard Dirichlet conditioning (the Matheron / Gibbs form).
    For copula families the latent Gaussian must be conditioned before the pointwise map, so only ``"restrict"`` is
    offered there.

Regularity
----------
``Sigma_BB`` is a principal block of an SPD matrix and always exists: the discrete trace is bounded regardless of
the field's regularity. Continuum trace-class-ness is a different question: the RKHS of a Whittle-Matern prior is
``H^{power}(D)``, evaluation on a ``d'``-dimensional set is bounded iff ``power > d - d'/2``, and Seeley's trace
calculus makes ``gamma C gamma*`` a pseudo-differential operator of order ``-(2 power - 1)`` on the
``(d-1)``-dimensional boundary, with eigenvalues ``~ j^{-(2 power - 1)/(d-1)}``: trace class exactly when
``2 power > d`` -- the same threshold the bulk point-value guard uses. So the sampler needs no new regularity
condition, but whether the returned data converges to an ``L^2(Gamma)`` function as ``h -> 0`` does;
:meth:`BoundarySampler.trace_eigenvalues` measures the exponent.

Reverting this prototype: delete this file, the lines tagged ``[bnd]`` in ``ngsrng/generator.py`` and
``tests/test_boundary.py``; nothing else in the library calls into it.
"""

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .covariance import as_covariance
from .space import SpaceInfo, assemble, to_scipy

try:  # torch rows arrive from the GPU engine; the algebra below is written to run in either backend
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None

__all__ = ["boundary_mass", "boundary_dofs", "boundary_covariance", "BoundarySampler", "attach_boundary",
           "RESTRICT", "FIELD", "MODES"]

RESTRICT, FIELD = "restrict", "field"
MODES = (RESTRICT, FIELD)
_TOL = 1e-13


def _convert(arr, x):
    """numpy ``x`` in the dtype/device of ``arr`` (a numpy array or a torch tensor)."""
    if _torch is not None and isinstance(arr, _torch.Tensor):
        x = np.asarray(x)
        return _torch.as_tensor(x.copy() if not x.flags.writeable else x, dtype=arr.dtype, device=arr.device)
    return np.asarray(x).astype(arr.dtype, copy=False)


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
    dofs = np.flatnonzero(diag > _TOL * max(float(np.abs(diag).max()), 1.0))
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


class BoundarySampler:
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
        self._off = np.setdiff1d(np.arange(self.fes.ndof), self.dofs)
        self._chol = None
        self._residual = None
        self._src = as_covariance(source, self.fes)
        # sigma_bb, lifting and P need n_bnd applications of Sigma_h and dense n_free x n_bnd blocks: computed on
        # first use, so "restrict" draws (which only need the DoF lists) never form them
        self._regression = None
        self.info = {"name": self.name, "n_bnd": self.n_bnd, "dofs": self.dofs.copy()}

    def _regress(self):
        if self._regression is None:
            free = self.space.free
            E = np.zeros((free.size, self.n_bnd))
            E[self._pos, np.arange(self.n_bnd)] = 1.0
            cols = np.asarray(self._src.sigma(E), dtype=np.float64)  # Sigma_h[:, B], free coordinates
            block = cols[self._pos, :]
            sigma_bb = 0.5 * (block + block.T)
            # Dense linear algebra below is only ever on n_bnd x n_bnd blocks (the trace DoFs); everything of
            # interior size stays sparse. sigma_bb is the one small dense factorisation: n_bnd^2 by construction.
            bb = spla.splu(sp.csc_matrix(sigma_bb))
            lifting = cols @ bb.solve(np.eye(self.n_bnd))
            lifting[self._pos, :] = np.eye(self.n_bnd)                  # R P = I, so R(u - P(Ru - g)) = g exactly
            P = np.zeros((self.fes.ndof, self.n_bnd))
            P[free, :] = lifting
            self._regression = (sigma_bb, lifting, P)
        return self._regression

    @property
    def sigma_bb(self):
        """Covariance of the boundary DoFs (n_bnd x n_bnd)."""
        return self._regress()[0]

    @property
    def lifting(self):
        """``Sigma_{.,B} Sigma_BB^{-1}`` on the free DoFs (n_free x n_bnd), identity rows at ``B``."""
        return self._regress()[1]

    @property
    def P(self):
        """The lifting on all DoFs (ndof x n_bnd; zero rows at fixed DoFs)."""
        return self._regress()[2]

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
            lu = spla.splu(sp.csc_matrix(K[ii][:, ii]))
            self._residual = np.asarray(lu.solve(np.eye(ii.size)))  # dense by request: a verification diagnostic
        return self._residual

    def dtn(self, form):
        """Assembled discrete Dirichlet-to-Neumann map ``S = K_BB - K_BI K_II^{-1} K_IB``.

        Provided to check ``sigma_bb == S^{-1}`` for a ``power = 1`` precision independently of this module;
        ``form`` is the bilinear form of the precision (``lambda u, v: InnerProduct(grad(u), grad(v)) * dx``).
        """
        K = sp.csc_matrix(assemble(self.fes, form)[self.space.free][:, self.space.free])
        b, i = self._free, ~self._free
        lu = spla.splu(K[i][:, i].tocsc())
        # SuperLU takes a dense right-hand side: it has n_bnd columns, the small block of the problem, while the
        # n_int x n_int interior factorisation stays sparse, so this is not a dense solve.
        X = lu.solve(np.asarray(K[i][:, b].todense()))
        return sp.csr_matrix(np.asarray(K[b][:, b].todense()) - np.asarray(sp.csr_matrix(K[b][:, i]) @ X))

    def trace_eigenvalues(self, n=None):
        """Generalised eigenvalues of ``(Sigma_BB, M_Gamma)``, ``M_Gamma`` = boundary mass matrix.

        They approximate the eigenvalues of the continuum trace covariance ``gamma C gamma*``, which decay like
        ``j^{-(2 power - 1)/(d-1)}``: their sum is finite exactly when ``2 power > d``. The pencil is ``n_bnd`` x
        ``n_bnd`` (the trace DoFs), which is why it is the one place a dense ``n_bnd`` matrix is formed from the
        sparse boundary mass.
        """
        M = boundary_mass(self.fes, self.name).tocsc()[self.dofs][:, self.dofs]
        S = sp.csc_matrix(self.sigma_bb)
        n_bnd = S.shape[0]
        if n is None:
            # dense by request: the full spectrum of an n_bnd x n_bnd symmetric-definite pencil. Do not use on a
            # boundary mesh fine enough that n_bnd is large - pass n to get the top n by ARPACK instead.
            from scipy.linalg import eigh as geigh
            w = np.sort(geigh(S.toarray(), M.toarray(), eigvals_only=True))[::-1]
            return w
        k = min(int(n), n_bnd - 1)
        if k < 1:
            return np.array([S.toarray()[0, 0] / M.toarray()[0, 0]])
        w = spla.eigsh(S, M=M, k=k, which="LM", return_eigenvectors=False)
        return np.sort(w)[::-1]

    # ----------------------------------------------------------------------- sampling

    def sample_trace(self, n, rng, loc=None, scale=1.0):
        """``n`` independent draws of the boundary marginal ``N(loc_B, scale^2 Sigma_BB)`` as (n, n_bnd) rows."""
        if self._chol is None:
            w, V = np.linalg.eigh(self.sigma_bb)
            if w.min() <= -1e-8 * max(w.max(), 1e-300):
                raise np.linalg.LinAlgError("the boundary covariance block is not positive semidefinite")
            self._chol = V * np.sqrt(np.clip(w, 0.0, None))
        # drawn sample by sample, so in-place fills in chunks give the draws of one call (ngsrng.inplace)
        out = float(scale) * (rng.standard_normal((int(n), self.n_bnd)) @ self._chol.T)
        loc = _vector(loc)
        if loc is not None:
            out += loc[self.dofs]
        return out

    def lift(self, loc, values):
        """Full coefficient vector of ``loc`` plus the lifting of the boundary data ``values`` (the conditional mean).

        For ``power = 1`` and ``loc = 0`` this is the ``K``-harmonic extension of the data into the domain.
        """
        base = np.zeros(self.fes.ndof, dtype=np.float64) if (mean := _vector(loc)) is None else mean.copy()
        return base + self.P @ (np.asarray(values, dtype=np.float64).ravel() - base[self.dofs])

    def restrict(self, rows):
        """Zero every DoF that carries no trace (in place). The exact marginal of the boundary DoFs."""
        off = _index(rows, self._off)
        if rows.ndim == 1:
            rows[off] = 0
        else:
            rows[:, off] = 0
        return rows

    def condition(self, rows, values=None, rng=None, loc=None, scale=1.0):
        """Rewrite the trace of ``rows`` (samples by DoFs, in place) to ``values``.

        ``values=None`` draws the trace independently from its marginal (needs ``rng``), which leaves the field an
        exact prior sample; with values given, ``rows`` becomes exact conditioning on ``u_h|Gamma = values``.
        """
        one = rows.ndim == 1
        r = rows.reshape(1, -1) if one else rows
        if values is None:
            if rng is None:
                raise ValueError("field mode without prescribed boundary values needs an rng to draw them")
            g = self.sample_trace(r.shape[0], rng, loc, scale)
        else:
            g = np.asarray(values, dtype=np.float64).reshape(-1, self.n_bnd)
            if g.shape[0] == 1 and r.shape[0] != 1:
                g = np.broadcast_to(g, (r.shape[0], self.n_bnd))
            if g.shape[0] != r.shape[0]:
                raise ValueError(f"bnd_values has {g.shape[0]} rows for {r.shape[0]} samples")
        g, P, dofs = _convert(r, g), _convert(r, self.P), _index(r, self.dofs)
        r -= (r[:, dofs] - g) @ P.T
        r[:, dofs] = g  # exact by construction (R P = I); written for bit-level agreement with the data
        return rows


def _latent(field):
    """``(Gaussian, affine)`` behind a built field: the Gaussian whose DoFs the field's samples are built from.

    ``affine`` says whether the field is an affine function of that Gaussian (loc + scale g), which is what
    ``bnd_mode="field"`` needs; copula families, mixtures and quadratic fields are nonlinear in it.
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


def attach_boundary(field, name, mode=None, values=None, generator=None):
    """Attach the ``bnd=`` specification of a freshly built field (called by :class:`ngsrng.Generator`).

    Sets ``field.bnd`` (what :meth:`ngsrng.Generator._apply_bnd` needs) and adds the boundary block to
    ``field.info``. Raises for the combinations that have no boundary interpretation rather than silently
    changing what the user asked for.
    """
    if mode is None:  # prescribed data needs the field mode; a plain bnd= draw is a restriction
        mode = FIELD if values is not None else RESTRICT
    if mode not in MODES:
        raise ValueError(f"bnd_mode must be one of {MODES}, got {mode!r}")
    if getattr(field, "evaluation", None) is not None or getattr(field, "coords", None) is not None:
        raise ValueError(f'bnd="{name}" restricts finite element DoFs; this field delivers values at given points. '
                         "Drop points= to sample the field itself.")
    rep = getattr(field, "representation", None)
    if rep not in (None, "projection", "bernstein"):
        raise ValueError(f'bnd="{name}" restricts finite element DoFs; representation="{rep}" delivers values at '
                         "quadrature points, which have no DoF trace. Use representation='projection' (or "
                         "'bernstein'), which returns coefficients.")
    gauss, affine = _latent(field)
    sampler = BoundarySampler(gauss, name)
    loc, scale = (None, 1.0)
    if affine:
        loc, scale = getattr(field, "loc", None), float(getattr(field, "scale", 1.0))
    if mode == FIELD:
        if not affine:
            raise NotImplementedError(
                f'bnd_mode="field" replaces the trace of an affine (Gaussian) field by an independent draw of its '
                f'boundary marginal; {type(field).__name__} is a nonlinear function of its Gaussian. Use '
                'bnd_mode="restrict" (exact for every family), or condition the Gaussian itself and build the family '
                "on the conditioned Gaussian.")
        if values is None and generator is None:
            raise ValueError("bnd_mode='field' without bnd_values needs a generator to draw the boundary data from")
    elif values is not None:
        raise ValueError('bnd_values needs bnd_mode="field": bnd_mode="restrict" draws from the field and then '
                         "restricts it, so its trace is random and prescribed data would be silently dropped.")
    if values is not None:
        values = _vector(values)
        if values.size == sampler.fes.ndof:  # a GridFunction (or coefficient vector) of the boundary data
            values = values[sampler.dofs]
        if values.size != sampler.n_bnd:
            raise ValueError(f"bnd_values has {values.size} entries; the boundary '{name}' has {sampler.n_bnd} DoFs "
                             "(in increasing DoF order)")
    field.bnd = {"sampler": sampler, "mode": mode, "values": values,
                 "rng": np.random.default_rng(generator._seed()) if generator is not None else None,
                 "loc": loc, "scale": scale,
                 "info": {**sampler.info, "mode": mode,
                          "data": "prescribed" if values is not None else "drawn from the marginal"}}


def _boundaries(source, fes=None):
    fes = fes if fes is not None else getattr(getattr(source, "space", None), "fes", None)
    if fes is None:
        return ()
    return fes.mesh.GetBoundaries()
