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

from ..backend import (
    COMPUTE,
    DeviceCache,
    Rebuildable,
    compute_device,
    ngs_sparse,
    sparse_to_torch,
    torch_device_of,
)
from ..covariance import EXACT_LIMIT
from ..inplace import InPlace
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


def prior_of(gauss):
    """The unconditioned Gaussian a posterior (or downdate) was derived from; the Gaussian itself for a prior."""
    prior = getattr(gauss, "_prior", None)
    return gauss if prior is None else prior


def is_conditioned(gauss) -> bool:
    return prior_of(gauss) is not gauss


def reference_of(gauss):
    """The Gaussian whose pointwise mean and standard deviation standardise "prior"-standardised families: the prior
    a conditioned Gaussian was derived from and, for a field conditioned on boundary data when it was built (a
    :class:`~ngsrng.StationaryMatern` on a space with Dirichlet dofs), the field before that conditioning. The
    boundary data then enter the family through the transform, as observations do for posteriors."""
    prior = prior_of(gauss)
    unconditioned = getattr(prior, "unconditioned", None)
    return prior if unconditioned is None else unconditioned


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


class GaussianAtPoints(Rebuildable):  # pickles; the NGSolve evaluation matrix is made on first use
    """Values of a Gaussian random function at points: the quadrature points of its space (default), any physical
    ``points`` (m x dim), or the domain points of a :class:`~ngsrng.distributions.bernstein.BernsteinForm`
    (``bernstein=``).

    ``standardize_by``: whose pointwise mean and standard deviation standardise the values,
    z = (g - m_ref) / std_ref:
    - "prior" (default): the prior a conditioned Gaussian was derived from (the Gaussian itself for priors; for a
      StationaryMatern conditioned on Dirichlet data, the field before that conditioning: :func:`reference_of`). A
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

    Raises if the Gaussian is known to have no point values (see :func:`point_values`). If the Gaussian computes on a
    torch device (the torch engine, a low-rank model on a GPU), point values can be produced there (``on_device=True``).
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
        self.reference = reference_of(gauss) if standardize_by == "prior" else gauss
        self.mean = self.eval_full @ gauss.mean
        self.center = self.eval_full @ self.reference.mean
        self.torch_device = torch_device_of(gauss)  # where on_device draws run (None: NumPy / NGSolve)
        self.device_capable = self.torch_device is not None and self.torch_device.type != "cpu"
        self._device = DeviceCache()  # evaluation matrices, the projection's operators, per device and dtype
        if isinstance(std, str) and std == "auto":
            cheap = hasattr(self.reference, "rinv") or self.reference.space.nfree <= self.EXACT_STD_LIMIT
            std = "exact" if cheap else "lowrank"
        self.std_method = std if isinstance(std, str) or std is None else "given"
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
            # fixed probes: a deterministic estimate that leaves the sampler's own randomness alone (ngsrng.seeding)
            ref, probes = self.reference, np.random.default_rng(0)
            dev = compute_device("auto", ref)
            if dev is not None:  # on the reference's GPU: only the probes go there, only the sums come back
                ev = sparse_to_torch(self._free_eval(), dev, torch.float64)
                acc = torch.zeros(self.npoints, dtype=torch.float64, device=dev)
                for k0 in range(0, mc_samples, 500):
                    eta = torch.from_numpy(probes.standard_normal((min(500, mc_samples - k0), ref.nrand)).T).to(dev)
                    acc += ((ev @ ref.apply(eta).to(torch.float64)) ** 2).sum(dim=1)
                return np.sqrt(acc.cpu().numpy() / mc_samples)
            acc = np.zeros(self.npoints)
            for k0 in range(0, mc_samples, 500):
                k = min(500, mc_samples - k0)
                eta = torch.from_numpy(probes.standard_normal((k, ref.nrand)).T)
                full = np.repeat(ref.mean[:, None], k, axis=1)
                full[ref.space.free] += ref.apply(eta).double().cpu().numpy()
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

    def _torch_ops(self, dtype=None):
        """(evaluation matrix, center, std) on the Gaussian's torch device in ``dtype`` (default: its output dtype)."""
        dtype = getattr(self.gauss, "dtype", torch.float64) if dtype is None else dtype
        key = ("ops", dtype)
        if key not in self._device:
            dev = self.torch_device
            center = torch.from_numpy(self.center).to(dev, dtype)
            std = None if self.std is None else torch.from_numpy(self.std).to(dev, dtype)
            self._device[key] = (sparse_to_torch(self.eval_full, dev, dtype), center, std)
        return self._device[key]

    def projection_ops(self, device):
        """The L2 projection on ``device`` (float64): B on the free dofs (nfree x npoints, the load of point values),
        a solver for the free mass matrix, and the free dofs' indices (None when every dof is free)."""
        key = ("projection", str(device))
        if key not in self._device:
            from ..solvers import device_spd_solver

            s = self.gauss.space
            b = self.rule.b.tocsr()[s.free]
            free = None if s.nfree == s.ndof else torch.from_numpy(s.free).to(device)
            self._device[key] = (sparse_to_torch(b, device, torch.float64), device_spd_solver(s.mass_free, device),
                                 free)
        return self._device[key]

    def draw(self, n, on_device=False, dtype=None):
        """n samples of the point values (npoints x n): NumPy (float64), or a tensor on the Gaussian's torch device if
        ``on_device`` and possible (in ``dtype``; default its output dtype)."""
        n = int(n)
        g = self.gauss
        if on_device and self.torch_device is not None:
            ev = self._torch_ops(dtype)[0]
            return ev @ g.sample(n).to(ev.dtype)
        if getattr(g, "engine", None) == "ngsolve" and hasattr(g, "_ngs_rows"):
            # CPU engine: the point values of every sample are taken in the worker thread that computes it, from
            # the float64 coefficients (no full-size intermediate, no float32 round trip)
            if getattr(self, "_ngs_eval", None) is None:
                self._ngs_eval = ngs_sparse(self.eval_full)
            return g._ngs_rows(n, np.empty((n, self.npoints)), evaluate=self._ngs_eval).T
        full = g.sample(n)
        return self.eval_full @ full.double().cpu().numpy()

    def standardized(self, n, on_device=False, dtype=None):
        """(g - m_ref) / std_ref at the points: standard normal marginals for priors (npoints x n)."""
        if self.std is None:
            raise ValueError("standardisation needs std ('exact', 'mc' or an array)")
        if on_device and self.torch_device is not None:
            _, center, std = self._torch_ops(dtype)
            return self.draw(n, True, dtype).sub_(center[:, None]).div_(std[:, None])
        z = self.draw(n)  # a fresh array: standardise in place
        z -= self.center[:, None]
        z /= self.std[:, None]
        return z

    def _pin(self):
        """Points where the reference Gaussian is deterministic (e.g. on a Dirichlet boundary of the covariance, or
        outside the part of the mesh the space lives on): their standardised value is set to 0, i.e. the median of the
        target distribution."""
        if self.std is not None:
            pinned = self.std <= 1e-12 * max(self.std.max(), 1e-300)
            if pinned.any():
                self.std = np.where(pinned, np.inf, self.std)


def _standardised_by(points) -> str:
    """For ``info``: whose mean and standard deviation standardise the values."""
    unconditioned = getattr(prior_of(points.gauss), "unconditioned", None)
    if unconditioned is not None and points.reference is unconditioned:
        return "prior (the field before its boundary update)"
    return points.standardize_by if is_conditioned(points.gauss) else "own (prior)"


class PointwiseField(InPlace):
    """Base class: a pointwise function of Gaussian point values.

    Representations: "quadrature" (values at the quadrature points; a GridFunction on the IntegrationRuleSpace for
    a single sample), "projection" (L2 projection into the FE space) or "points" (values at user-given points;
    plain arrays). Subclasses implement ``_values(n)`` returning (npoints, ..., n) arrays; ``torch_capable`` says
    whether the transform has a torch path (NumPy otherwise). ``out_dtype`` is the precision the values are
    delivered in (bounded families keep their values strictly inside the support in that precision). In-place fills
    (``sample_into``, :mod:`ngsrng.inplace`) run ``sample`` on chunks of ``work_values`` internal values.

    ``compute`` says where everything after the Gaussian draw runs (point evaluation, transform, projection or
    Bernstein conversion): "auto" (on the Gaussian's GPU when the family has a torch path, so the samples never leave
    it; on the host otherwise), "device" (on the Gaussian's torch device, also a CPU one) or "host" (NumPy and
    NGSolve on the CPU, the Gaussian's samples copied there). The projection and the Bernstein conversion run in
    float64 either way.
    """

    torch_capable = False
    out_dtype = np.float64

    def __init__(self, points: GaussianAtPoints, representation="quadrature", compute="auto"):
        if points.bernstein is not None:
            representation = "bernstein"
        elif points.rule is None:
            representation = "points"
        elif representation not in ("quadrature", "projection"):
            raise ValueError("representation must be 'quadrature', 'projection' or 'bernstein'")
        if compute not in COMPUTE:
            raise ValueError(f"compute must be one of {', '.join(COMPUTE)}, not {compute!r}")
        self.points, self.representation, self.compute = points, representation, compute
        self.space = points.gauss.space
        self._mass_solver = None

    @property
    def on_device(self) -> bool:
        """Whether the draws are processed on the Gaussian's torch device (see ``compute``)."""
        if self.compute == "host" or not self.torch_capable or self.points.torch_device is None:
            return False
        return self.compute == "device" or self.points.device_capable

    def check_compute(self, name="this family"):
        """Raise if ``compute="device"`` cannot be honoured (no torch device, no torch path)."""
        if self.compute != "device":
            return
        compute_device("device", self.points.gauss, f"{name} with compute='device'")
        if not self.torch_capable:
            raise ValueError(f"compute='device': {name} has no torch path here (a user function, shape parameters "
                             "that vary in space, or method='ppf': scipy on the CPU); use compute='auto' or 'host'")

    @property
    def _latent_dtype(self):
        """The dtype of device draws: float64 for the coefficient representations (projection, Bernstein), as on the
        host; the Gaussian's output dtype for point values."""
        return torch.float64 if self.representation in ("projection", "bernstein") else None

    def _latent_draw(self, n, standardize=True):
        """n draws of the latent point values (standardised or raw) where this field computes."""
        p, dev, dt = self.points, self.on_device, self._latent_dtype
        draw = p.standardized if standardize else p.draw
        return draw(n, dev) if dt is None else draw(n, dev, dtype=dt)

    def _values(self, n):
        raise NotImplementedError

    def _common_info(self) -> dict:
        p = self.points
        return {
            "representation": self.representation,
            "standardisation": p.std_method,
            "standardised_by": _standardised_by(p),
            "point_values": p.point_values,
            "points": p.npoints,
            "device": "gpu" if self.on_device and p.torch_device.type == "cuda" else "cpu",
            "compute": self.compute,
            "gaussian": p.gauss.plan,
        }

    CHUNK_VALUES = 2e7  # Gaussian point values per internal batch (bounds the transforms' intermediates)

    def _per_sample(self):
        # the Gaussian behind a batch is ndof x n as well (and solvers keep a few such blocks): bound both
        return max(self.points.npoints, self.space.ndof) * getattr(self, "copies", 1)

    def sample_into(self, rows):
        # an "auto" quantile inversion picks a table or the exact ppf from the number of values per call: decide it
        # for the whole fill, so the result does not depend on the chunking
        values = rows.shape[0] * self.points.npoints * getattr(self, "copies", 1)
        for inv in self._inversions():
            inv._resolve(values, self.on_device)
        return super().sample_into(rows)

    def _inversions(self):
        from .inversion import Inversion

        found = []
        for v in vars(self).values():
            found += [x for x in (v if isinstance(v, (list, tuple)) else [v]) if isinstance(x, Inversion)]
        return found

    def _chunked(self, n):
        size = max(1, int(self.CHUNK_VALUES // self._per_sample()))
        if n <= size:
            return self._values(n)
        parts = [self._values(min(size, n - k)) for k in range(0, n, size)]
        return torch.cat(parts, -1) if isinstance(parts[0], torch.Tensor) else np.concatenate(parts, -1)

    def sample(self, n=None):
        """``n=None``: one sample (quadrature: a GridFunction on the IntegrationRuleSpace; projection: on the FE
        space; points: an array). Otherwise an array: (npoints, ..., n) point values, or (ndof, ..., n)
        coefficients."""
        return self._represent(self._chunked(1 if n is None else n), n)

    def _represent(self, vals, n):
        """Point values (npoints, ..., n) in the field's representation, as ``sample(n)`` returns them."""
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

        Dirichlet dofs are zero (zero load there). Device tensors are projected on their device (cuDSS on NVIDIA GPUs,
        Jacobi PCG elsewhere), as torch tensors.
        """
        if isinstance(values, torch.Tensor) and self.on_device:
            return self._project_device(values)
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

    def _project_device(self, values):
        b_free, solver, free = self.points.projection_ops(values.device)
        shape = values.shape
        sol = solver.apply(b_free @ values.reshape(shape[0], -1).to(torch.float64))
        if free is None:
            return sol.reshape((self.space.ndof,) + shape[1:])
        out = torch.zeros((self.space.ndof, sol.shape[1]), dtype=torch.float64, device=values.device)
        out[free] = sol
        return out.reshape((self.space.ndof,) + shape[1:])

    MARGIN = 1e-9  # relative safety margin of the Bernstein representation (conversion rounding is ~1e-12)

    def bernstein_coefficients(self, values) -> np.ndarray:
        """FE coefficients (ndof, ..., n) of sum_j v_j B_j for values v at the Bernstein domain points (npoints, ...,
        n): every point of the FE function lies in the convex hull of the values (convex hull property).

        In floating point the conversion to NGSolve's basis has a rounding error (measured at the rule points, ~1e-12
        relative); values are first pulled into the interior of the family's convex range by a margin at least 1000
        times larger (``_pull_inside``), so the guarantee survives rounding. Device tensors are converted on their
        device (float64, torch tensors).
        """
        if isinstance(values, torch.Tensor) and self.on_device:
            values = values.to(torch.float64)
            flat = values.reshape(values.shape[0], -1, values.shape[-1])
            scale = flat.abs().amax(dim=(0, 1)).clamp_min(1e-300)
        else:
            if isinstance(values, torch.Tensor):
                values = values.double().cpu().numpy()
            flat = values.reshape(values.shape[0], -1, values.shape[-1])
            scale = np.maximum(np.abs(flat).max(axis=(0, 1)), 1e-300)  # per sample
        shape = values.shape
        delta = self.MARGIN
        for _ in range(3):
            inside = self._pull_inside(values, delta * scale).reshape(shape[0], -1)
            coeffs, resid = self.points.bernstein.to_coefficients(inside)
            needed = float((1e3 * resid.reshape(-1, shape[-1]) / scale).max())
            if needed <= delta:
                break
            delta = 10 * needed
        else:
            raise RuntimeError("could not make the Bernstein conversion safe against rounding")
        return coeffs.reshape((coeffs.shape[0],) + shape[1:])

    def _pull_inside(self, values, margin):
        """Move values (npoints, ..., n) into the interior of the family's range by ``margin`` (per sample, shape
        (n,)). NumPy arrays or torch tensors (then ``margin`` is one too). Default: nothing to do (the range is not
        constrained, or closed and only reached exactly)."""
        return values

    def clamp_open(self, x, support, xp):
        """Keep values strictly inside a finite support (lo, hi) *in the output precision*: the transforms are exact
        to rounding, but e.g. Phi(z) rounds to 1 for z > 8.3 in float64 and the float32 cast of 1 - 6e-8 is 1. The
        family's support is fixed, so the bounds are made once per array type, device and dtype."""
        if support is None:
            return x
        key = ("clamp", str(x.device) if xp.torch else "numpy", x.dtype, np.dtype(self.out_dtype).str)
        cache = self.__dict__.setdefault("_clamp", DeviceCache())
        if key not in cache:
            dt = np.dtype(self.out_dtype)
            bounds = []
            for bound, toward, op in ((support[0], np.inf, "max"), (support[1], -np.inf, "min")):
                b = np.asarray(bound, dtype=float)
                if not np.all(np.isfinite(b)):
                    continue
                inner = np.nextafter(b.astype(dt), np.asarray(toward, dtype=dt)).astype(np.float64)
                if xp.torch:
                    inner = torch.as_tensor(inner, dtype=x.dtype, device=x.device)
                elif not inner.ndim:
                    inner = float(inner)
                bounds.append((inner, op))
            cache[key] = bounds
        out = x
        for inner, op in cache[key]:
            if xp.torch:
                out = torch.maximum(out, inner) if op == "max" else torch.minimum(out, inner)
            else:
                out = np.maximum(out, inner) if op == "max" else np.minimum(out, inner)
        return out
