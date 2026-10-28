"""Gaussian fields at points, and the representation layer of pointwise-defined fields.

Pointwise families are functions of the *point values* of a Gaussian random function g, so they are meaningful only
if g has point values in the continuum limit (continuous sample paths; for Matern-type C = A^-beta with second-order
A: 2 beta > d).
"""

import warnings

import ngsolve as ngs
import numpy as np
import scipy.special as sps
import torch

from ..backend import ngs_sparse, sparse_to_torch
from ..covariance import EXACT_LIMIT
from ..observations import observation_operator
from ..quadrature import QuadratureRule
from ..solvers import NGSolveDirect
from ..space import mass_form

_SQRT1_2 = float(np.sqrt(0.5))


class XP:
    """Minimal array-module switch so each transform is written once for NumPy (CPU) and torch (GPU)."""

    def __init__(self, torch_mode: bool):
        self.torch = torch_mode
        self.m = torch if torch_mode else np

    def ndtr(self, x):
        if not self.torch:
            return sps.ndtr(x)
        # torch.special.ndtr computes (1 + erf) / 2, which cancels in the lower tail (4% relative error at x = -8,
        # zero below -10); the erfc form keeps full relative accuracy in both tails
        tail = 0.5 * torch.special.erfc(x.abs() * _SQRT1_2)
        return torch.where(x < 0, tail, 1.0 - tail)

    def log_ndtr(self, x):
        return torch.special.log_ndtr(x) if self.torch else sps.log_ndtr(x)

    def ndtri(self, x):
        return torch.special.ndtri(x) if self.torch else sps.ndtri(x)

    def where(self, c, a, b):
        return torch.where(c, a, b) if self.torch else np.where(c, a, b)

    def const(self, v, like):
        """A parameter (scalar or NumPy array of point values) in the array type of ``like``."""
        if self.torch and isinstance(v, np.ndarray):
            return torch.as_tensor(v, device=like.device, dtype=like.dtype)
        return v


def prior_of(gauss):
    """The unconditioned Gaussian a posterior (or downdate) was derived from; the Gaussian itself for a prior."""
    prior = getattr(gauss, "_prior", None)
    return gauss if prior is None else prior


def is_conditioned(gauss) -> bool:
    return prior_of(gauss) is not gauss


def warn_copies_of_posterior():
    warnings.warn(
        "this family combines several independent copies of the Gaussian; with a conditioned Gaussian every copy is "
        "conditioned on the same data, which is rarely the intended model (use it with priors)",
        stacklevel=4,
    )


def point_values(gauss) -> str:
    """Whether the Gaussian has point values in the continuum limit: "yes", "no" or "not verified".

    Decided for PDE precisions C = A^-power with a second-order A (the Matern family): point values exist (continuous
    sample paths) iff 2 power > d, which for Matern is nu = power - d/2 > 0. Other specifications are not checked.
    """
    from ..specs import PDEPrecision

    prior = prior_of(gauss)
    spec = getattr(prior, "spec", None)
    if isinstance(spec, PDEPrecision):
        return "yes" if 2 * spec.power > prior.space.dim else "no"
    return "not verified"


class GaussianAtPoints:
    """Values of a Gaussian random function at points: the quadrature points of its space (default), any physical
    ``points`` (m x dim), or the domain points of a :class:`~ngsrng.distributions.bernstein.BernsteinForm`
    (``bernstein=``).

    ``standardize_by``: whose pointwise mean and standard deviation standardise the values,
    z = (g - m_ref) / std_ref:
    - "prior" (default): the prior a conditioned Gaussian was derived from (the Gaussian itself for priors). A
      pointwise family built on a posterior is then exactly the posterior of the same family built on the prior,
      given linear observations of the underlying Gaussian;
    - "own": the Gaussian itself (its copula only: a posterior's mean and variance are divided out).

    ``std``: how std_ref is obtained (all through :func:`ngsrng.covariance.local_covariance`, cached on the
    reference Gaussian):
    - "auto": "exact" when that is cheap (finite-rank Gaussians, up to ``EXACT_STD_LIMIT`` free dofs), "lowrank"
      beyond;
    - "exact": sqrt(diag(E Sigma E^T)) from the entries of Sigma in the pattern of M: exact. Free for finite-rank
      Gaussians; otherwise one application of L and L^T per free dof (batched), so meant for moderate problems;
    - "lowrank": the exact variance of a randomised rank-200 part plus a 64-sample Monte Carlo estimate of the rest
      (unbiased, with an error estimate; ~1400 applications, far more accurate than "mc" for smooth fields);
    - "mc": estimated from ``mc_samples`` samples (relative error of the standard deviation ~ 1/sqrt(2 mc_samples));
    - an array of length npoints;
    - None: no standardisation available (only for transforms that do not need it).

    Raises if the Gaussian is known to have no point values (see :func:`point_values`). If the Gaussian lives on a
    CUDA device (torch engine), point values can be produced on the device (``on_device=True``).
    """

    EXACT_STD_LIMIT = EXACT_LIMIT

    def __init__(self, gauss, order=None, std="auto", mc_samples=4000, points=None, standardize_by="prior",
                 bernstein=None):
        if standardize_by not in ("prior", "own"):
            raise ValueError("standardize_by must be 'prior' or 'own'")
        self.point_values = point_values(gauss)
        if self.point_values == "no":
            raise ValueError(
                "this Gaussian has no point values (C = A^-power with 2 power <= dimension, i.e. Matern nu <= 0): its "
                "samples are not functions in the limit, so pointwise families (uniform, gamma, lognormal, ...) would "
                "degenerate under mesh refinement. Use a smoother correlation (larger power / nu)."
            )
        self.gauss, self.bernstein = gauss, bernstein
        if bernstein is not None:
            self.rule, self.coords = None, None
            self.eval_full = bernstein.evaluation
        elif points is None:
            self.rule, self.coords = QuadratureRule(gauss.space, order), None
            self.eval_full = self.rule.evaluation_full  # E (npoints x ndof)
        else:
            self.rule, self.coords = None, np.atleast_2d(np.asarray(points, dtype=float))
            self.eval_full = observation_operator(gauss.space.fes, points=self.coords)
        self.standardize_by = standardize_by
        self.reference = prior_of(gauss) if standardize_by == "prior" else gauss
        self.mean = self.eval_full @ gauss.mean
        self.center = self.eval_full @ self.reference.mean
        self.device_capable = getattr(gauss, "engine", None) == "torch" and gauss.device.type == "cuda"
        if isinstance(std, str) and std == "auto":
            cheap = hasattr(self.reference, "rinv") or self.reference.space.nfree <= self.EXACT_STD_LIMIT
            std = "exact" if cheap else "lowrank"
        self.std_method = std if isinstance(std, str) or std is None else "given"
        self._t = None
        self.std = self._std(std, mc_samples)
        self._pin()

    @property
    def npoints(self) -> int:
        return self.eval_full.shape[0]

    def parameter(self, value):
        """A distribution parameter as a scalar, or as point values (npoints x 1) if it varies in space.

        Accepts numbers, CoefficientFunctions/GridFunctions (evaluated at the points) and arrays of point values.
        """
        if isinstance(value, (ngs.CoefficientFunction, ngs.GridFunction)):
            if self.bernstein is not None:
                return self.bernstein.cf_values(value)[:, None]
            if self.rule is not None:
                gf = ngs.GridFunction(self.rule.irs)
                gf.Set(value)  # on an IntegrationRuleSpace, Set evaluates at the points
                return gf.vec.FV().NumPy().copy()[:, None]
            mesh = self.gauss.space.mesh
            return np.asarray(value(mesh(*self.coords.T)), dtype=float).reshape(self.npoints, 1)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        if arr.shape[0] != self.npoints:
            raise ValueError(f"spatially varying parameters need one value per point ({self.npoints})")
        return arr.reshape(self.npoints, 1)

    def _std(self, std, mc_samples):
        if std is None:
            return None
        if isinstance(std, str) and std == "exact":
            return np.sqrt(np.clip(self._local("exact").values(self._free_eval()), 0.0, None))
        if isinstance(std, str) and std == "lowrank":
            lc = self._local("lowrank", rank=200, samples=64)
            v, se = lc.values(self._free_eval(), stderr=True)
            rel = 0.5 * se / np.maximum(v, 1e-300)  # relative error of the std is half that of the variance
            warnings.warn(
                f"pointwise standard deviations estimated ({lc.method}): relative error ~{np.median(rel):.1%} "
                f"(median), {rel.max():.1%} (max); pass std='exact' for exact marginals",
                stacklevel=3,
            )
            return np.sqrt(np.clip(v, 0.0, None))
        if isinstance(std, str) and std == "mc":
            warnings.warn(
                f"pointwise standard deviations estimated from {mc_samples} samples (relative error "
                f"~{1 / np.sqrt(2 * mc_samples):.1%}); pass std='exact' or an array for exact marginals",
                stacklevel=3,
            )
            acc = np.zeros(self.npoints)
            for k0 in range(0, mc_samples, 500):
                full = self.reference.sample(min(500, mc_samples - k0)).double().cpu().numpy()
                x = self.eval_full @ full - self.center[:, None]
                acc += (x**2).sum(axis=1)
            return np.sqrt(acc / mc_samples)
        std = np.asarray(std, dtype=float)
        if std.shape != (self.npoints,):
            raise ValueError("std must have one entry per point")
        return std

    def _local(self, method, **kwargs):
        from ..covariance import local_covariance

        return local_covariance(self.reference, method, **kwargs)

    def _free_eval(self):
        return self.eval_full[:, self.reference.space.free]

    def _torch_ops(self):
        if self._t is None:
            g = self.gauss
            center = torch.from_numpy(self.center).to(g.device, g.dtype)
            std = None if self.std is None else torch.from_numpy(self.std).to(g.device, g.dtype)
            self._t = (sparse_to_torch(self.eval_full, g.device, g.dtype), center, std)
        return self._t

    def draw(self, n, on_device=False):
        """n samples of the point values (npoints x n): NumPy (float64), or a device tensor if ``on_device`` and
        possible."""
        n = int(n)
        g = self.gauss
        if on_device and self.device_capable:
            return self._torch_ops()[0] @ g.sample(n)
        if getattr(g, "engine", None) == "ngsolve" and hasattr(g, "_ngs_rows"):
            # CPU engine: the point values of every sample are taken in the worker thread that computes it, from
            # the float64 coefficients (no full-size intermediate, no float32 round trip)
            if getattr(self, "_ngs_eval", None) is None:
                self._ngs_eval = ngs_sparse(self.eval_full)
            return g._ngs_rows(n, np.empty((n, self.npoints)), evaluate=self._ngs_eval).T
        full = g.sample(n)
        return self.eval_full @ full.double().cpu().numpy()

    def standardized(self, n, on_device=False):
        """(g - m_ref) / std_ref at the points: standard normal marginals for priors (npoints x n)."""
        if self.std is None:
            raise ValueError("standardisation needs std ('exact', 'mc' or an array)")
        if on_device and self.device_capable:
            _, center, std = self._torch_ops()
            return self.draw(n, True).sub_(center[:, None]).div_(std[:, None])
        z = self.draw(n)  # a fresh array: standardise in place
        z -= self.center[:, None]
        z /= self.std[:, None]
        return z

    def _pin(self):
        """Points where the reference Gaussian is deterministic (e.g. on a Dirichlet boundary of the covariance):
        their standardised value is set to 0, i.e. the median of the target distribution."""
        if self.std is not None:
            pinned = self.std <= 1e-12 * max(self.std.max(), 1e-300)
            if pinned.any():
                self.std = np.where(pinned, np.inf, self.std)


class PointwiseField:
    """Base class: a pointwise function of Gaussian point values.

    Representations: "quadrature" (values at the quadrature points; a GridFunction on the IntegrationRuleSpace for
    a single sample), "projection" (L2 projection into the FE space) or "points" (values at user-given points;
    plain arrays). Subclasses implement ``_values(n)`` returning (npoints, ..., n) arrays; ``torch_capable`` says
    whether the transform can run on a CUDA device (NumPy otherwise). ``out_dtype`` is the precision the values are
    delivered in (bounded families keep their values strictly inside the support in that precision).
    """

    torch_capable = False
    out_dtype = np.float64

    def __init__(self, points: GaussianAtPoints, representation="quadrature"):
        if points.bernstein is not None:
            representation = "bernstein"
        elif points.rule is None:
            representation = "points"
        elif representation not in ("quadrature", "projection"):
            raise ValueError("representation must be 'quadrature', 'projection' or 'bernstein'")
        self.points, self.representation = points, representation
        self.space = points.gauss.space
        self._mass_solver = None

    @property
    def on_device(self) -> bool:
        return (self.torch_capable and self.points.device_capable
                and self.representation not in ("projection", "bernstein"))

    def _values(self, n):
        raise NotImplementedError

    def _common_info(self) -> dict:
        p = self.points
        return {
            "representation": self.representation,
            "standardisation": p.std_method,
            "standardised_by": p.standardize_by if is_conditioned(p.gauss) else "own (prior)",
            "point_values": p.point_values,
            "points": p.npoints,
            "device": "gpu" if self.on_device else "cpu",
            "gaussian": p.gauss.plan,
        }

    CHUNK_VALUES = 2e7  # Gaussian point values per internal batch (bounds the transforms' intermediates)

    def _chunked(self, n):
        # the Gaussian behind a batch is ndof x n as well (and solvers keep a few such blocks): bound both
        per_sample = max(self.points.npoints, self.space.ndof) * getattr(self, "copies", 1)
        size = max(1, int(self.CHUNK_VALUES // per_sample))
        if n <= size:
            return self._values(n)
        parts = [self._values(min(size, n - k)) for k in range(0, n, size)]
        return torch.cat(parts, -1) if isinstance(parts[0], torch.Tensor) else np.concatenate(parts, -1)

    def sample(self, n=None):
        """``n=None``: one sample (quadrature: a GridFunction on the IntegrationRuleSpace; projection: on the FE
        space; points: an array). Otherwise an array: (npoints, ..., n) point values, or (ndof, ..., n)
        coefficients."""
        vals = self._chunked(1 if n is None else n)
        if self.representation == "projection":
            vals = self.project(vals)
        elif self.representation == "bernstein":
            vals = self.bernstein_coefficients(vals)
        if n is not None:
            return vals
        if vals.ndim != 2 or self.representation == "points":
            return vals[..., 0]
        return self.as_gridfunction(vals[:, 0])

    def as_gridfunction(self, values) -> ngs.GridFunction:
        if self.representation == "points":
            raise ValueError("values at user-given points have no GridFunction representation")
        if isinstance(values, torch.Tensor):
            values = values.double().cpu().numpy()
        if self.representation == "quadrature":
            return self.points.rule.to_gridfunction(values)
        # projection and bernstein: FE coefficients
        gf = ngs.GridFunction(self.space.fes)
        gf.vec.FV().NumPy()[:] = values
        return gf

    def project(self, values) -> np.ndarray:
        """L2 projection of point values (npoints, ..., n) into the FE space: coefficients (ndof, ..., n).

        Dirichlet dofs are zero (zero load there).
        """
        if isinstance(values, torch.Tensor):
            values = values.double().cpu().numpy()
        if self._mass_solver is None:
            self._mass_solver = NGSolveDirect(self.space, mass_form)
        shape = values.shape
        flat = values.reshape(shape[0], -1)
        load = self.points.rule.load(flat)[self.space.free]
        sol = self._mass_solver.apply(torch.from_numpy(np.ascontiguousarray(load))).numpy()
        out = np.zeros((self.space.ndof, flat.shape[1]))
        out[self.space.free] = sol
        return out.reshape((self.space.ndof,) + shape[1:])

    MARGIN = 1e-9  # relative safety margin of the Bernstein representation (conversion rounding is ~1e-12)

    def bernstein_coefficients(self, values) -> np.ndarray:
        """FE coefficients (ndof, ..., n) of sum_j v_j B_j for values v at the Bernstein domain points (npoints, ...,
        n): every point of the FE function lies in the convex hull of the values (convex hull property).

        In floating point the conversion to NGSolve's basis has a rounding error (measured at the rule points, ~1e-12
        relative); values are first pulled into the interior of the family's convex range by a margin at least 1000
        times larger (``_pull_inside``), so the guarantee survives rounding.
        """
        if isinstance(values, torch.Tensor):
            values = values.double().cpu().numpy()
        shape = values.shape
        flat = values.reshape(shape[0], -1, shape[-1])
        scale = np.maximum(np.abs(flat).max(axis=(0, 1)), 1e-300)  # per sample
        delta = self.MARGIN
        for _ in range(3):
            inside = self._pull_inside(values, delta * scale).reshape(shape[0], -1)
            coeffs, resid = self.points.bernstein.to_coefficients(inside)
            needed = (1e3 * resid.reshape(-1, shape[-1]) / scale).max()
            if needed <= delta:
                break
            delta = 10 * needed
        else:
            raise RuntimeError("could not make the Bernstein conversion safe against rounding")
        return coeffs.reshape((coeffs.shape[0],) + shape[1:])

    def _pull_inside(self, values, margin):
        """Move values (npoints, ..., n) into the interior of the family's range by ``margin`` (per sample, shape
        (n,)). Default: nothing to do (the range is not constrained, or closed and only reached exactly)."""
        return values

    def clamp_open(self, x, support, xp):
        """Keep values strictly inside a finite support (lo, hi) *in the output precision*: the transforms are exact
        to rounding, but e.g. Phi(z) rounds to 1 for z > 8.3 in float64 and the float32 cast of 1 - 6e-8 is 1."""
        if support is None:
            return x
        dt = np.dtype(self.out_dtype)
        out = x
        for bound, toward, op in ((support[0], np.inf, "max"), (support[1], -np.inf, "min")):
            b = np.asarray(bound, dtype=float)
            if not np.all(np.isfinite(b)):
                continue
            inner = np.nextafter(b.astype(dt), np.asarray(toward, dtype=dt)).astype(np.float64)
            inner = xp.const(inner, out) if inner.ndim else float(inner)
            if xp.torch:
                inner = inner if isinstance(inner, torch.Tensor) else torch.tensor(inner, dtype=out.dtype,
                                                                                   device=out.device)
                out = torch.maximum(out, inner) if op == "max" else torch.minimum(out, inner)
            else:
                out = np.maximum(out, inner) if op == "max" else np.minimum(out, inner)
        return out
