"""Correlated random fields: several fields, each with its own law, correlated with one another.

``ngsrng.correlated([d1, d2, ...], cross_correlation)`` takes fields built as usual (``rng.distribution(...)``: any
pointwise family, its parameters, representation, ``bnd=``, and the Gaussian it is built on, conditioned or not) and
draws them jointly::

    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    g1, g2 = ngsrng.correlated([d, d], 0.8).sample()           # two GridFunctions, correlated

The model (intrinsic coregionalisation)
---------------------------------------
Every family here is a map of a latent Gaussian: f_i = F_i^-1(Phi(z_i)) for the pointwise families, with z_i the
standardised latent values at the points, or loc + scale g for Gaussian fields. With W_1, ..., W_k independent draws
of the latents, m_i their means and R = L L^T the k x k cross-correlation (Cholesky factor L),

    Z_i = m_i + sum_j L_ij (W_j - m_j),        f_i = the family's map of Z_i.

- **Every component keeps its law.** The rows of L have unit length, so Z_i has the mean and covariance of W_i: the
  same latent law, hence the same field (marginals, spatial correlation, the Bernstein range guarantee, ``bnd=``).
- **Cross-covariance R_ij k(x, y)**: the components share the spatial correlation k and are correlated with one
  another through R (separable cross-covariance). The mixing happens in the latents, before the maps, so R is the
  correlation of the latent Gaussians. For uniform (any continuous) marginals Spearman's rank correlation of the
  values is (6 / pi) arcsin(R_ij / 2) (0.786 for R_ij = 0.8), which is also the Pearson correlation of uniform
  values; a target rank correlation r needs R_ij = 2 sin(pi r / 6).
- **Conditioning holds per component.** Only the fluctuations W_j - m_j are mixed; each component keeps its own
  mean. A component built on a posterior keeps its conditioning exactly, and the components may be conditioned on
  different data (posteriors of one prior at the same points with the same noise share their covariance). No
  transformation of the data is needed, and copies of one posterior are fine.
- **What the components must share**: the latent covariance (after the pointwise standardisation), and the points
  where the latents live (the same space and representation, or the same ``points=``); for Gaussian fields, the
  covariance of the coefficients. The constructor checks both (the covariance with two random probes).
- **Cost**: the components' own draws plus a k x k mix per point and sample.

Which fields: the pointwise families (``uniform``, ``lognormal``, ``gamma``, ``beta``, ..., ``normal`` with a
``representation=``, ``marginal``, ``indicator``, ``transformed``) and Gaussian FE fields (``normal()`` without
``points=``). Families built from several latents (chi-square sums, Student-t, Wishart, Dirichlet, ...) are not
pointwise maps of one latent, and Gaussian fields and pointwise families live in different latent spaces
(coefficients and point values), so neither can be mixed here.
"""

import math
import numbers

import ngsolve as ngs
import numpy as np
import scipy.linalg as sla
import torch

from .backend import device_copy
from .inplace import InPlace, as_tensor, write

__all__ = ["correlated", "CorrelatedFields"]

PROBE_RTOL = 1e-6  # relative difference of the probed latent covariances above which the components differ


def correlated(components, cross_correlation):
    """Draw several fields jointly, correlated through their latent Gaussians (see :mod:`ngsrng.correlated`).

    Parameters
    ----------
    components : sequence of FieldDistributions (``rng.distribution(name, ...)``), one per component. The same
        distribution may appear several times (independent copies, then correlated).
    cross_correlation : the correlation R of the components' latent Gaussians: a number for two components, or a
        k x k symmetric positive semidefinite matrix with unit diagonal.

    Returns
    -------
    CorrelatedFields : ``sample(size=None, output=None, out=None)`` returns one result per component, each as the
    component's own ``sample`` would return it.
    """
    return CorrelatedFields(components, cross_correlation)


class CorrelatedFields(InPlace):
    """Several fields drawn jointly with cross-correlation ``cross_correlation`` (see :func:`correlated`)."""

    def __init__(self, components, cross_correlation):
        from .generator import FieldDistribution

        comps = list(components)
        if len(comps) < 2:
            raise ValueError("correlated() needs at least two components")
        for d in comps:
            if not isinstance(d, FieldDistribution):
                raise TypeError("components are distributions from rng.distribution(name, ...), "
                                f"not {type(d).__name__}")
        self.components = tuple(comps)
        self.cross_correlation = _correlation_matrix(cross_correlation, len(comps))
        self._mix = _factor(self.cross_correlation)
        self._identity = bool(np.array_equal(self._mix, np.eye(len(comps))))
        self._parts = [_Part(d) for d in comps]
        _check_shared(self._parts)
        self.space = self._parts[0].field.space

    @property
    def info(self) -> dict:
        return {"law": "correlated components: the components' laws, cross-correlated through their latent "
                       "Gaussians (intrinsic coregionalisation)",
                "cross_correlation": self.cross_correlation.tolist(),
                "components": [d.info for d in self.components]}

    def __len__(self):
        return len(self.components)

    def __repr__(self):
        names = [d.name for d in self.components]
        return f"CorrelatedFields({names}, cross_correlation={self.cross_correlation.tolist()})"

    # ------------------------------------------------------------------ sampling

    def sample(self, size=None, output=None, out=None):
        """One result per component (a tuple): with ``size=None`` one sample each (a GridFunction, or point values),
        otherwise arrays (*size, nvalues), as each component's ``sample(size)`` returns them. ``out``: one buffer per
        component (*size, nvalues), filled in place and returned."""
        if out is not None:
            return self._sample_out(size, out)
        shape = () if size is None else ((int(size),) if isinstance(size, numbers.Integral)
                                         else tuple(int(s) for s in size))
        n = math.prod(shape)
        rows = [p.buffer(n) for p in self._parts]
        self._fill(rows, min(p.chunk for p in self._parts))
        return tuple(p.deliver(r, size, shape, output) for p, r in zip(self._parts, rows))

    def rvs(self, size=None, output=None, out=None):
        """Alias of ``sample``."""
        return self.sample(size, output, out)

    def sample_into(self, rows):
        """Fill one (n, nvalues) buffer per component in place (:mod:`ngsrng.inplace`)."""
        rows = [as_tensor(r) for r in rows]
        if len(rows) != len(self._parts):
            raise ValueError(f"out needs one buffer per component ({len(self._parts)})")
        n = rows[0].shape[0]
        flat = []
        for p, r in zip(self._parts, rows):
            if r.shape[0] != n or r[0].numel() != p.nvalues:
                raise ValueError(f"each buffer holds {n} samples of {p.nvalues} values (component {p.name})")
            if p.keep64 and r.dtype != torch.float64:
                raise ValueError("the Bernstein representation's range guarantee is about float64 coefficients: "
                                 "pass float64 buffers")
            flat.append(r.view(n, p.nvalues))
        self._fill(flat, min(p.fill_chunk for p in self._parts))  # in-place: bounded by work_values
        return rows

    def _sample_out(self, size, out):
        bufs = list(out)
        shape = None if size is None else ((int(size),) if isinstance(size, numbers.Integral)
                                           else tuple(int(s) for s in size))
        for b in bufs:
            t = as_tensor(b)
            if not t.is_contiguous():
                raise ValueError("out buffers must be contiguous")
            if shape is not None and tuple(t.shape[: len(shape)]) != shape:
                raise ValueError(f"out has shape {tuple(t.shape)}, size is {shape}")
        n = math.prod(shape) if shape is not None else as_tensor(bufs[0]).shape[0]
        self.sample_into([as_tensor(b).reshape(n, -1) for b in bufs])
        return tuple(out)

    def _fill(self, rows, step):
        """Draw n joint samples into the (n, nvalues) buffers, ``step`` samples at a time: the latents of every
        component, mixed, mapped, represented, and the ``bnd=`` step. The values do not depend on ``step``: within the
        draw every latent continues its stream, and "auto" inversions are decided for the whole draw."""
        rows = [as_tensor(r) for r in rows]  # views: NumPy buffers are filled in place
        n = rows[0].shape[0]
        if n == 0:
            return
        for p in self._parts:
            p.resolve(n)
        for k0 in range(0, n, step):
            k = min(step, n - k0)
            z = [p.field._latent(k) for p in self._parts]
            if not self._identity:
                z = self._mixed(z)
            for p, zi, r in zip(self._parts, z, rows):
                block = r[k0:k0 + k]
                write(block, p.represent(p.field._from_latent(zi), k))
                p.boundary(block)

    def _mixed(self, z):
        """Z_i = m_i + sum_j L_ij (W_j - m_j), in the latents' own array type (NumPy, or torch on their device)."""
        torch_mode = any(isinstance(x, torch.Tensor) for x in z)
        if torch_mode:
            dev = next(x.device for x in z if isinstance(x, torch.Tensor))
            z = [torch.as_tensor(x, dtype=torch.float64).to(dev) for x in z]
            means = [device_copy(self, p.mean, dev, torch.float64)[:, None] for p in self._parts]
        else:
            z = [np.asarray(x, dtype=np.float64) for x in z]
            means = [p.mean[:, None] for p in self._parts]
        fluct = [x - m for x, m in zip(z, means)]
        out = []
        for i, mi in enumerate(means):
            terms = [self._mix[i, j] * fluct[j] for j in range(len(fluct)) if self._mix[i, j] != 0.0]
            out.append(mi + sum(terms[1:], terms[0]))
        return out


class _Part:
    """One component: its field and generator, and what the joint draw needs from them."""

    def __init__(self, dist):
        from .distributions.transforms import MarginalField
        from .generator import _NormalField

        field, gen = dist.field, dist.generator
        self.dist, self.field, self.gen, self.name = dist, field, gen, dist.name
        if isinstance(field, MarginalField):
            self.kind = "points"
        elif isinstance(field, _NormalField) and field.evaluation is None:
            self.kind = "coefficients"
        else:
            raise NotImplementedError(
                f"{dist.name}: correlated() mixes pointwise maps of one latent Gaussian (uniform, lognormal, gamma, "
                "beta, ..., normal with a representation=, marginal, indicator, transformed) and Gaussian FE fields "
                "(normal() without points=); this field is built from several latents or values at given points")
        if self.kind == "points":
            self.points = field.points
            self.nvalues = field.space.ndof if field.representation in ("projection", "bernstein") \
                else field.points.npoints
            self.chunk = max(1, int(field.CHUNK_VALUES // field._per_sample()))  # as the field's own draw
        else:
            self.points = None
            self.nvalues = field.space.ndof
            self.chunk = max(1, int(field.work_values // max(field._per_sample(), 1)))
        self.mean = field._latent_mean()
        self.discrete, self.keep64 = gen._output_kind(field, dist.name)

    @property
    def fill_chunk(self) -> int:
        """Samples per chunk of an in-place fill (``work_values`` internal values, as other samplers' fills)."""
        return max(1, int(self.field.work_values // max(self.field._per_sample(), 1)), self.field._min_chunk())

    @property
    def gauss(self):
        return self.field.gauss if self.kind == "coefficients" else self.points.gauss

    def resolve(self, n):
        """Decide "auto" quantile inversions for the whole draw, as an in-place fill does (chunk independent)."""
        if self.kind == "points":
            for inv in self.field._inversions():
                inv._resolve(n * self.points.npoints, self.field.on_device)

    def represent(self, vals, k):
        if self.kind == "points":
            return self.field._represent(vals, k)
        return vals

    def boundary(self, rows):
        if getattr(self.field, "bnd", None) is not None:  # [bnd] in place on the (k, ndof) rows
            self.gen._apply_bnd(self.field, rows)

    def buffer(self, n):
        if self.kind == "points":
            dev = self.gauss.device if self.field.on_device else None
        else:
            dev = self.field.latent_device
        if dev is not None:
            return torch.empty((n, self.nvalues), dtype=torch.float64, device=dev)
        return np.empty((n, self.nvalues))

    def deliver(self, rows, size, shape, output):
        """The component's result from its (n, nvalues) rows, as its own ``sample(size)`` returns it."""
        if size is None:
            row = rows[0]
            if self.kind == "coefficients" or self.field.representation != "points":
                return self._gridfunction(row)
            return self.gen._convert(row, output, self.discrete, self.keep64)
        vals = rows.reshape(*shape, self.nvalues)
        return self.gen._convert(vals, output, self.discrete, self.keep64)

    def _gridfunction(self, row):
        row = row.detach().cpu().numpy() if isinstance(row, torch.Tensor) else np.asarray(row)
        if self.kind == "coefficients":
            gf = ngs.GridFunction(self.field.space.fes)
            gf.vec.FV().NumPy()[:] = row
            return gf
        return self.field.as_gridfunction(row)

    # ---- the latent covariance, applied to probes (for the compatibility check) ----
    def probe(self, v):
        """Covariance of the latent fluctuations applied to ``v`` (nvalues_latent x k): at the points, after the
        standardisation, for pointwise families; of the free coefficients for Gaussian fields."""
        g = self.gauss
        free = g.space.free

        def sigma(y):
            t = torch.from_numpy(np.ascontiguousarray(y))
            return g.apply(g.apply_t(t)).detach().double().cpu().numpy()

        if self.kind == "coefficients":
            return sigma(v[free])
        e = self.points.eval_full[:, free]
        d = self.field._latent_scale()[:, None]
        d = np.where(np.isfinite(d), d, 0.0)
        return d * (e @ sigma(e.T @ (d * v)))

    def latent_size(self):
        return self.points.npoints if self.kind == "points" else self.field.space.ndof

    def latent_key(self):
        """Where the latents live: the points (and their evaluation) or the coefficients of the space."""
        if self.kind == "points":
            return ("points", self.points.npoints, self.field.space.fes)
        return ("coefficients", self.field.space.ndof, self.field.space.fes)


def _correlation_matrix(r, k) -> np.ndarray:
    arr = np.asarray(r, dtype=float)
    if arr.ndim == 0:
        if k != 2:
            raise ValueError(f"a single cross-correlation is for two components; {k} need a {k} x {k} matrix")
        if not -1.0 <= float(arr) <= 1.0:
            raise ValueError("a correlation lies in [-1, 1]")
        arr = np.array([[1.0, float(arr)], [float(arr), 1.0]])
    if arr.shape != (k, k):
        raise ValueError(f"cross_correlation must be {k} x {k} for {k} components, got {arr.shape}")
    if not np.allclose(arr, arr.T, rtol=0, atol=1e-12) or not np.allclose(np.diag(arr), 1.0, rtol=0, atol=1e-12):
        raise ValueError("cross_correlation must be symmetric with unit diagonal")
    if np.linalg.eigvalsh(0.5 * (arr + arr.T)).min() < -1e-10:
        raise ValueError("cross_correlation must be positive semidefinite")
    return 0.5 * (arr + arr.T)


def _factor(r) -> np.ndarray:
    """L with L L^T = R and rows of unit length: Cholesky (lower triangular: component 1 is its own draw) when R is
    positive definite, a symmetric square root otherwise (e.g. a correlation of +-1)."""
    try:
        return sla.cholesky(r, lower=True)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(r)
        return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def _check_shared(parts):
    """The components must have their latents at the same places with the same covariance."""
    keys = [p.latent_key() for p in parts]
    first = keys[0]
    for p, key in zip(parts[1:], keys[1:]):
        if key[0] != first[0]:
            raise ValueError("Gaussian fields (FE coefficients) and pointwise families (point values) have their "
                             "latents in different places and cannot be mixed; use normal(..., representation=...) "
                             "for a Gaussian component among pointwise ones")
        if key[2] is not first[2] or key[1] != first[1]:
            raise ValueError(f"{p.name}: the components must live on the same space, with the same representation "
                             "(or the same points=)")
        if p.kind == "points" and p.points is not parts[0].points:
            a, b = p.points.eval_full, parts[0].points.eval_full
            if a.shape != b.shape or (a != b).nnz:
                raise ValueError(f"{p.name}: the components' latents must live at the same points (same space, "
                                 "representation, order and points=)")
    # the covariance of the latent fluctuations: two random probes (skipped for shared latents)
    distinct = {(id(p.gauss), id(p.points)) for p in parts}
    if len(distinct) == 1:
        return
    v = np.random.default_rng(0).standard_normal((parts[0].latent_size(), 2))
    ref = parts[0].probe(v)
    scale = max(np.abs(ref).max(), 1e-300)
    for p in parts[1:]:
        if (id(p.gauss), id(p.points)) == (id(parts[0].gauss), id(parts[0].points)):
            continue
        diff = np.abs(p.probe(v) - ref).max() / scale
        if diff > PROBE_RTOL:
            raise ValueError(
                f"{p.name}: the components' latent Gaussians differ in covariance (relative difference {diff:.1e}). "
                "Correlated components share one latent covariance: build them on one prior, or on posteriors of "
                "one prior conditioned at the same points with the same noise (their data may differ)")
