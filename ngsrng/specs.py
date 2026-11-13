"""Covariance/precision specifications.

Conventions: on coefficient vectors, a covariance is stored as Sigma_h (dual -> primal), a
precision as Sigma_h^{-1} (primal -> dual). Both are symmetric positive definite.
"""

import warnings
from dataclasses import dataclass
from typing import Callable

import ngsolve as ngs
import numpy as np
import torch


@dataclass
class PDEPrecision:
    """C = scale^2 * A^{-power}, where A is the operator of the bilinear form ``form(u, v)``.

    ``form`` includes its measures (dx, ds, skeleton terms). Boundary conditions come from the space (Dirichlet) and
    the form (Robin), with Neumann implicit. With stiffness K and mass M, the coefficient covariance is
    Sigma_h = (K^{-1} M)^power M^{-1}. ``power`` may be any positive real (Matern smoothness nu = power - d/2);
    non-integer powers use a rational approximation with relative accuracy ``rtol`` on the discrete spectrum.
    """

    form: Callable
    power: float = 2
    scale: float = 1.0
    rtol: float = 1e-7

    def __post_init__(self):
        if self.power <= 0:
            raise ValueError("power must be positive")
        if abs(self.power - round(self.power)) < 1e-12:
            self.power = int(round(self.power))

    @property
    def is_integer(self) -> bool:
        return isinstance(self.power, int)


@dataclass
class AssembledPrecision:
    """Sigma_h^{-1} = Q, assembled from ``form(u, v)`` (an even-order / non-decoupling precision)."""

    form: Callable
    scale: float = 1.0


@dataclass
class OperatorPrecision:
    """A precision known only through its action ("apply-only", tier T0).

    ``apply(c)``: torch tensor of free-dof coefficients (nfree x k) -> torch tensor (nfree x k), SPD. With
    ``convention="primal->dual"`` it is Sigma^-1 itself (e.g. K M^-1 K for a bi-Laplacian prior); with
    ``"dual->dual"`` it is the dual-space operator Q_dd, and Sigma^-1 = Q_dd M.
    Sampling uses c ~ r(M^-1 Q) M^-1 z with r(lambda) ~ lambda^-1/2 (relative accuracy ``rtol``) and one
    multi-shift CG run per sample (tolerance ``tol``). ``spectrum=(lmin, lmax)`` of M^-1 Q may be given; otherwise it
    is estimated (inverse iteration and Lanczos, widened by safety margins).
    """

    apply: Callable
    convention: str = "primal->dual"
    scale: float = 1.0
    rtol: float = 1e-7
    tol: float = 1e-8
    spectrum: tuple | None = None

    def __post_init__(self):
        if self.convention not in ("primal->dual", "dual->dual"):
            raise ValueError("convention must be 'primal->dual' or 'dual->dual'")


# The classical Robin coefficient beta = ROBIN_FACTOR * kappa for A = kappa^2 - Laplace (C = A^-2 in 2D): the empirical
# constant of Roininen, Huttunen & Lasanen (2014), used by Daon & Stadler (2018) and, as beta = sqrt(gamma delta) / 1.42
# for A = delta - div(gamma grad), by hIPPYlib (Villa & O'Leary-Roseberry 2024). Not a default (the best coefficient
# depends on nu and the domain: ngsrng.tune_robin); tune_robin reports it for comparison.
ROBIN_FACTOR = 1 / 1.42


def matern(kappa: float, power: float = 2, scale: float = 1.0, coef=None, robin=None) -> PDEPrecision:
    """Whittle-Matern-type prior: A = kappa^2 - div(coef grad). H1-conforming spaces only (L2 needs a DG form).

    Boundary condition: natural (Neumann) by default, Dirichlet through the space; ``robin=beta`` adds the Robin
    condition (coef grad u) . n + beta u = 0, i.e. the term beta * u * v * ds (to estimate beta for a space, use
    ``matern_correlation(length, nu, boundary="robin", beta=None)`` or :func:`ngsrng.tune_robin`).
    """
    return PDEPrecision(MaternForm(kappa, coef, _robin(robin, kappa)), power, scale)


@dataclass(eq=False)  # compared and hashed by identity, as functions are
class MaternForm:
    """(coef grad u, grad v) + kappa^2 (u, v) [+ beta (u, v)_ds]: the form of :func:`matern`."""

    kappa: float
    coef: object = None
    beta: float | None = None

    def __call__(self, u, v):
        c = self.coef
        g = ngs.grad(u) * ngs.grad(v) if c is None else ngs.InnerProduct(c * ngs.grad(u), ngs.grad(v))
        out = (g + self.kappa**2 * u * v) * ngs.dx
        return out if self.beta is None else out + self.beta * u * v * ngs.ds


def _robin(robin, kappa):
    if robin is None:
        return None
    if isinstance(robin, str):
        raise ValueError("robin= is the Robin coefficient beta itself (robin='auto', beta = kappa / 1.42, was removed: "
                         "that constant suits nu = 1 in 2D only). To estimate beta for the space, use "
                         "matern_correlation(length, nu, boundary='robin', beta=None) or ngsrng.tune_robin")
    beta = float(robin)
    if beta < 0:
        raise ValueError("the Robin coefficient must be nonnegative")
    return beta


def matern_dg(fes, kappa: float, power: float = 2, scale: float = 1.0, penalty: float = 4.0,
              robin=None) -> PDEPrecision:
    """Symmetric interior penalty discretisation of A = kappa^2 - Laplace on an L2 space (natural Neumann BC, or
    Robin with ``robin`` as in :func:`matern`).

    The space must be created with ``dgjumps=True``.
    """
    if not fes.flags.ToDict().get("dgjumps", False):
        raise ValueError("the Matern operator on an L2 space is its interior-penalty (SIPG) discretisation, which "
                         "couples neighbouring elements: create the space with dgjumps=True, e.g. "
                         "L2(mesh, order=p, dgjumps=True)")
    form = SipgMaternForm(kappa, fes.mesh.dim, max(fes.globalorder, 1), penalty, _robin(robin, kappa))
    return PDEPrecision(form, power, scale)


@dataclass(eq=False)  # compared and hashed by identity, as functions are
class SipgMaternForm:
    """Symmetric interior penalty form of kappa^2 - Laplace: the form of :func:`matern_dg`."""

    kappa: float
    dim: int
    order: int
    penalty: float = 4.0
    beta: float | None = None

    def __call__(self, u, v):
        n = ngs.specialcf.normal(self.dim)
        h = ngs.specialcf.mesh_size
        ju, jv = u - u.Other(), v - v.Other()
        mu = 0.5 * n * (ngs.grad(u) + ngs.grad(u.Other()))
        mv = 0.5 * n * (ngs.grad(v) + ngs.grad(v.Other()))
        out = (ngs.grad(u) * ngs.grad(v) + self.kappa**2 * u * v) * ngs.dx + (
            self.penalty * self.order**2 / h * ju * jv - mu * jv - mv * ju
        ) * ngs.dx(skeleton=True)
        return out if self.beta is None else out + self.beta * u * v * ngs.ds(skeleton=True)


CONVENTIONS = ("dual->primal", "primal->primal", "primal->dual")

LOCAL_FORM_WARNING = (
    "a BilinearForm used as a covariance, b(u, v) = E[(f, u)(f, v)], is a local operator, and no local operator is "
    "trace class: N(0, C) would be white-noise-like (its samples are not functions and their norms grow under mesh "
    "refinement). If the form is meant as a precision (its inverse as the covariance), use "
    "precision=AssembledPrecision(form) or PDEPrecision(form, power)"
)


@dataclass
class OperatorCovariance:
    """A covariance known only through its action (apply-only); sampled by an adaptive truncated Karhunen-Loeve
    expansion (:func:`ngsrng.lowrank.operator_kl`).

    ``apply``: a callable on free-dof coefficient blocks (NumPy or torch (nfree x k) -> same), an NGSolve
    ``BaseMatrix`` (any operator expression on full coefficient vectors, e.g. ``Ainv @ M.mat @ Ainv``; Dirichlet
    dofs are ignored) or a ``BilinearForm`` (its matrix). ``convention`` says which matrix it is:

    - "dual->primal" (default; callables and BaseMatrix): Sigma itself, mapping dual coefficients (load vectors) to
      primal ones, e.g. ``A.Inverse()`` or ``Ainv @ M @ Ainv``;
    - "primal->primal": the covariance operator on coefficients, C_h = Sigma M;
    - "primal->dual" (default for a BilinearForm): the covariance form b(u, v) = E[(f, u)(f, v)], matrix M Sigma M.

    The rank grows (doubling) until the trace not captured, estimated with white-noise Hutchinson probes z ~ N(0, M)
    (mesh-independent variance), is below ``tol`` times the total trace; ``rank`` fixes it instead. Truncation
    removes the smallest scales: the error is the uncaptured trace, so this suits smooth (fast spectral decay)
    covariances; rough ones need large ranks.
    """

    apply: Callable
    convention: str | None = None
    tol: float = 1e-2
    rank: int | None = None
    max_rank: int = 4000
    power_iterations: int = 2
    probes: int = 16

    def __post_init__(self):
        form = isinstance(self.apply, ngs.BilinearForm)
        if self.convention is None:
            self.convention = "primal->dual" if form else "dual->primal"
        if self.convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}")
        if form:
            warnings.warn(LOCAL_FORM_WARNING, stacklevel=3)

    def bind(self, fes, seed=None, mean=None, device="cpu"):
        """The sampler: an adaptive KL expansion (:func:`ngsrng.operator_kl`), sampling on ``device``."""
        from .lowrank import operator_kl

        return operator_kl(fes, self, mean=mean, seed=seed, device=device)

    def sigma(self, space) -> Callable:
        """Sigma (dual -> primal) on free-dof coefficient blocks: NumPy (nfree x k) -> NumPy (nfree x k)."""
        op = self.apply.mat if isinstance(self.apply, ngs.BilinearForm) else self.apply
        raw = BaseMatrixApply(op, space) if isinstance(op, ngs.BaseMatrix) else CallableApply(op)
        if self.convention == "dual->primal":
            return raw
        if space.mass_is_diagonal:
            minv = DiagonalSolve(space.mass_free.diagonal())
        else:
            from .distributions.bernstein import SymmetricFactor  # sparse Cholesky: SuperLU was 3-100x slower

            minv = SymmetricFactor(space.mass_free).solve
        return MassScaled(raw, minv, both=self.convention == "primal->dual")


class CallableApply:
    """A user's apply-only operator (torch in, torch or NumPy out) as a NumPy map on free-dof blocks."""

    def __init__(self, op):
        self.op = op

    def __call__(self, y):
        out = self.op(torch.from_numpy(np.ascontiguousarray(y)))
        return out.detach().cpu().double().numpy() if isinstance(out, torch.Tensor) else np.asarray(out)


class DiagonalSolve:
    """y -> D^-1 y for a diagonal matrix (its diagonal)."""

    def __init__(self, diagonal):
        self.d = np.asarray(diagonal, dtype=float)[:, None]

    def __call__(self, y):
        return y / (self.d if y.ndim == 2 else self.d[:, 0])


class MassScaled:
    """Sigma from a covariance operator given in another convention: raw(M^-1 y) (primal->primal) or
    M^-1 raw(M^-1 y) (primal->dual, ``both``)."""

    def __init__(self, raw, minv, both):
        self.raw, self.minv, self.both = raw, minv, both

    def __call__(self, y):
        out = self.raw(self.minv(y))
        return self.minv(out) if self.both else out


class BaseMatrixApply:
    """An NGSolve BaseMatrix on full coefficient vectors as a NumPy map on free-dof blocks. (The operator itself
    must pickle for this to pickle; NGSolve inverses, for example, do not.)"""

    def __init__(self, op, space):
        self.op, self.space = op, space

    def __call__(self, v):
        free = self.space.free
        x, y = ngs.la.BaseVector(self.space.ndof), ngs.la.BaseVector(self.space.ndof)
        v = v[:, None] if v.ndim == 1 else v
        out = np.empty((len(free), v.shape[1]))
        xn = x.FV().NumPy()
        with ngs.TaskManager():
            for k in range(v.shape[1]):
                xn[:] = 0.0
                xn[free] = v[:, k]
                y.data = self.op * x
                out[:, k] = y.FV().NumPy()[free]
        return out
