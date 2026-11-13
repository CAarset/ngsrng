"""Finite observations of a random function: the m-dimensional Gaussian y = G f + e.

The data of a Bayesian inverse problem live in R^m: an observation map G: X -> R^m (point values, local averages,
LinearForms, matrices, matrix-free maps through their adjoint; see :func:`ngsrng.observation_operator`) applied to a
random function f, plus noise e ~ N(0, Gamma): white, coloured (a covariance matrix), or itself the observation of
another random field (an :class:`ObservedGaussian`, e.g. a PDE-driven disturbance). :class:`ObservedGaussian` samples
y exactly, on the field's device and in the requested dtype:

- ``"factor"``: the m x m covariance K = scale^2 G Sigma G^T (+ Gamma) once, its eigendecomposition, and
  F = V diag(sqrt(w)) over the eigenvalues w > rtol max(w); a draw is F z, one dense product per batch. Exact up to
  the cutoff: ||K - F F^T||_2 <= rtol max(w) (default rtol 1e-12; ``plan["dropped_trace"]`` records the fraction of
  the trace left out). K comes from Y = L^T G^T (m applications of the field's factor L^T, on its device, so with
  cuDSS on NVIDIA GPUs) and one product Y^T Y, or from m applications of Sigma when Y does not fit in memory. For m up
  to about 10^4, when many draws amortise the set-up.
- ``"pathwise"``: a draw of f, then G applied to it (on the device as a torch sparse or dense matrix; on the CPU engine
  in the sampling threads). Exact, no set-up; one field draw per sample. For large m, or few draws.

``"auto"`` takes the factor up to ``FACTOR_LIMIT`` observations and pathwise beyond (the factor also for covariances
without a sampler, e.g. :class:`ngsrng.OperatorCovariance`). The choice and its costs are in ``plan``.
"""

import time

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
import torch

from .backend import (
    Rebuildable,
    available_memory,
    device_copy,
    device_free_memory,
    mean_array,
    ngs_sparse,
    resolve_device,
    resolve_dtype,
    same_device,
    sparse_to_torch,
    torch_device_of,
)
from .inplace import InPlace, as_tensor, chunks, write
from .observations import CHUNK, dense_if_full, observation_operator
from .seeding import as_source
from .space import SpaceInfo

FACTOR_LIMIT = 10_000  # "auto": the factor up to this many observations (m x m float64: 0.8 GB at the limit)
RTOL = 1e-12  # factor: eigenvalues below rtol * max are dropped


class ObservedGaussian(InPlace, Rebuildable):
    """y = G (loc + scale (f - mean)) + e: finitely many linear observations of a Gaussian random function, plus noise.

    Parameters
    ----------
    field : the random function: a ``Gaussian`` (prior or posterior, any engine and device), ``LowRankGaussian``,
        ``MatheronPosterior``, or a covariance without a sampler (an :class:`ngsrng.OperatorCovariance`, with ``fes``;
        factor route only)
    operator, functionals, points, averages, adjoint, m : the observation map G (see
        :func:`ngsrng.observation_operator`; ``operator`` is its ``matrix``)
    noise : None, a variance, variances (m,), a covariance matrix (m, m), or an independent ``ObservedGaussian`` with m
        observations (coloured noise from another random field)
    loc, scale : the field's mean (None: its own; a number, CoefficientFunction or GridFunction) and a factor on its
        fluctuation
    method : "auto" | "factor" | "pathwise" (see the module docstring); rtol the factor's eigenvalue cutoff
    seed : the randomness of the factor's draws and of the noise (:mod:`ngsrng.seeding`); pathwise draws of the field
        use the field's own randomness
    device, dtype : of the draws (default: the field's)

    ``sample(n)`` returns (m x n), ``sample_into(rows)`` fills (n x m) rows in place (any device and dtype); ``mean``,
    ``covariance()`` (exact, m x m), ``rank`` (factor) and ``plan`` describe it.
    """

    def __init__(self, field, operator=None, *, fes=None, functionals=None, points=None, averages=None, adjoint=None,
                 m=None, noise=None, loc=None, scale=1.0, method="auto", rtol=RTOL, seed=None, device=None,
                 dtype=None, factor=None):
        space = getattr(field, "space", None)
        if space is None:
            if fes is None:
                raise ValueError("a covariance without a space needs fes=")
            space = SpaceInfo(fes)
        self.field, self.space = field, space
        self.G = dense_if_full(observation_operator(space.fes, operator, functionals, points, averages, adjoint, m))
        self.evaluation = self.G  # (bnd= does not apply: these are observations, not FE coefficients)
        self.m = self.G.shape[0]
        self.scale, self.rtol = float(scale), float(rtol)
        self.device = resolve_device(device if device is not None else getattr(field, "device", "cpu"))
        self.dtype = resolve_dtype(dtype) if dtype is not None else getattr(field, "dtype", torch.float64)
        self.source = as_source(seed)
        self._torch_cache = {}
        loc_coeffs = _coefficients(loc, space, getattr(field, "mean", None))
        self._mean = np.asarray(self.G @ loc_coeffs, dtype=float).ravel()
        self._noise_sampler, self._noise_cov, self._noise_chol, self._noise_std = None, None, None, None
        self._set_noise(noise)
        can_sample = hasattr(field, "sample")
        if method == "auto":
            method = "factor" if (self.m <= FACTOR_LIMIT or not can_sample) else "pathwise"
            why = (f"m = {self.m} <= {FACTOR_LIMIT}" if self.m <= FACTOR_LIMIT else "no sampler") \
                if method == "factor" else f"m = {self.m} > {FACTOR_LIMIT}"
        else:
            why = "requested"
        if method not in ("factor", "pathwise"):
            raise ValueError("method must be 'auto', 'factor' or 'pathwise'")
        if method == "pathwise" and not can_sample:
            raise ValueError("pathwise sampling needs a sampler (Gaussian, LowRankGaussian, MatheronPosterior): "
                             "a covariance given as an operator has none; use method='factor'")
        self.method = method
        self.plan = {"method": method, "reason": why, "m": self.m,
                     "operator": "sparse" if sp.issparse(self.G) else "dense",
                     "noise": self._noise_label(noise), "field": getattr(field, "plan", type(field).__name__)}
        self._K, self.rank = None, None
        if method == "factor":
            self._factorize(factor)

    # ------------------------------------------------------------------ noise

    def _set_noise(self, noise):
        if noise is None:
            return
        if hasattr(noise, "sample_into") and hasattr(noise, "covariance"):
            if getattr(noise, "m", None) != self.m:
                raise ValueError(f"the noise has {getattr(noise, 'm', '?')} observations, G has {self.m}")
            self._noise_sampler = noise
            return
        arr = np.asarray(noise, dtype=float)
        if arr.ndim == 0:
            arr = np.full(self.m, float(arr))
        if arr.ndim == 1:
            if arr.shape != (self.m,) or (arr < 0).any():
                raise ValueError("noise variances must be non-negative, one per observation")
            self._noise_std, self._noise_cov = np.sqrt(arr), np.diag(arr)
        elif arr.shape == (self.m, self.m):
            self._noise_cov = 0.5 * (arr + arr.T)
            self._noise_chol = np.linalg.cholesky(self._noise_cov)
        else:
            raise ValueError(f"noise covariance has shape {arr.shape}, expected ({self.m}, {self.m})")

    @staticmethod
    def _noise_label(noise):
        if noise is None:
            return "none"
        if hasattr(noise, "sample_into"):
            return "an observed random field (independent)"
        arr = np.asarray(noise)
        return "white" if arr.ndim <= 1 else "coloured (covariance matrix)"

    def _noise_covariance(self):
        if self._noise_sampler is not None:
            return self._noise_sampler.covariance()
        return self._noise_cov

    # ------------------------------------------------------------------ the factor

    def _factorize(self, factor):
        """F with F F^T = K up to the cutoff; ``factor``: a precomputed :meth:`factor_data` (shared set-up)."""
        t0 = time.perf_counter()
        if factor is None:
            k, route = self._field_covariance()
            k = self.scale**2 * k
            noise = self._noise_cov if self._noise_sampler is None else None  # folded in: one product per draw
            if noise is not None:
                k = k + torch.as_tensor(noise, dtype=torch.float64, device=k.device)
            w, v = torch.linalg.eigh(k.to(self.device))
            wmax = float(w.max()) if w.numel() else 0.0
            keep = w > self.rtol * wmax if wmax > 0 else torch.zeros_like(w, dtype=torch.bool)
            f = v[:, keep] * torch.sqrt(w[keep])
            total = float(w.clamp(min=0).sum())
            dropped = float(w[~keep].clamp(min=0).sum()) / total if total > 0 else 0.0
            factor = {"F": f, "K": k.cpu().numpy(), "dropped_trace": dropped, "route": route,
                      "noise_folded": noise is not None, "setup_s": time.perf_counter() - t0}
        self._F64 = factor["F"].to(self.device, torch.float64)
        self._F = self._F64.to(self.dtype)
        self._K = factor["K"]
        self._folded = factor["noise_folded"]
        self._factor = factor
        self.rank = int(self._F.shape[1])
        self.plan.update(rank=self.rank, rtol=self.rtol, dropped_trace=factor["dropped_trace"],
                         covariance=factor["route"], setup_s=round(factor["setup_s"], 3))

    def factor_data(self) -> dict:
        """The factor's set-up (F, K and their description), to share with another sampler of the same law."""
        return self._factor

    def _free_operator(self):
        free = self.space.free
        return self.G if len(free) == self.space.ndof else self.G[:, free]

    def _field_covariance(self):
        """G Sigma G^T (m x m, float64 torch) and how it was computed."""
        f, gf = self.field, self._free_operator()
        gt = gf.T.tocsc() if sp.issparse(gf) else gf.T

        def block(i):
            b = gt[:, i:i + CHUNK]
            return torch.from_numpy(b.toarray() if sp.issparse(b) else np.ascontiguousarray(b))

        if hasattr(f, "apply_t") and hasattr(f, "nrand"):
            dev = torch_device_of(f) or torch.device("cpu")
            if 8.0 * f.nrand * self.m < 0.25 * _free_memory(dev):  # Y = L^T G^T, then Y^T Y
                y = torch.empty((f.nrand, self.m), dtype=torch.float64, device=dev)
                for i in range(0, self.m, CHUNK):
                    y[:, i:i + CHUNK] = f.apply_t(block(i).to(dev)).to(dev, torch.float64)
                return y.T @ y, f"Y = L^T G^T ({f.nrand} x {self.m}), K = Y^T Y"
            gdev = self._operator_torch(dev, torch.float64, free=True)
            k = torch.empty((self.m, self.m), dtype=torch.float64, device=dev)
            for i in range(0, self.m, CHUNK):
                k[:, i:i + CHUNK] = gdev @ f.apply(f.apply_t(block(i).to(dev))).to(dev, torch.float64)
            return 0.5 * (k + k.T), f"{self.m} applications of Sigma = L L^T"
        from .covariance import as_covariance

        src = as_covariance(f, self.space.fes)
        k = np.empty((self.m, self.m))
        for i in range(0, self.m, CHUNK):
            k[:, i:i + CHUNK] = np.asarray(gf @ src.sigma(block(i).numpy()))
        return torch.from_numpy(0.5 * (k + k.T)), f"{self.m} applications of Sigma"

    # ------------------------------------------------------------------ draws

    @property
    def mean(self) -> np.ndarray:
        """E[y] (m,)."""
        return self._mean.copy()

    def covariance(self) -> np.ndarray:
        """The exact covariance of y, scale^2 G Sigma G^T + Gamma (m x m; for the factor route, before its cutoff)."""
        if self._K is None:
            k = self.scale**2 * self._field_covariance()[0].cpu().numpy()
            self._K = k
            self._folded = False
        noise = self._noise_covariance()
        return self._K.copy() if (noise is None or self._folded) else self._K + noise

    def sample(self, n=None):
        """n draws as an (m x n) tensor (n=None: one, (m,)), on ``device`` in ``dtype``."""
        ns = 1 if n is None else int(n)
        rows = torch.empty((ns, self.m), dtype=self.dtype, device=self.device)
        self._rows(rows)
        return rows[0] if n is None else rows.T

    def sample_into(self, rows):
        """Fill rows (n x m; NumPy or torch, any float dtype and device) with n draws, in place."""
        rows = as_tensor(rows)
        if rows[0].numel() != self.m:
            raise ValueError(f"out holds {rows[0].numel()} values per sample, there are {self.m} observations")
        view = rows if rows.ndim == 2 else rows.view(rows.shape[0], self.m)
        direct = same_device(view.device, self.device) and view.dtype == self.dtype and view.is_contiguous()
        for k0, k1 in chunks(view.shape[0], self._per_sample(), self.work_values, self._min_chunk()):
            if direct:
                self._rows(view[k0:k1])
            else:
                write(view[k0:k1], self.sample(k1 - k0))
        return rows

    def _per_sample(self) -> int:
        return self.m + (self.rank if self.rank is not None else getattr(self.field, "nrand", 0)
                         + getattr(self.space, "ndof", 0))

    def _min_chunk(self) -> int:
        from .backend import available_threads

        return available_threads() if self.method == "pathwise" and getattr(self.field, "engine", "") == "ngsolve" \
            else 1

    def _rows(self, rows):
        """k draws into rows (k x m, contiguous, on the device in the dtype)."""
        k = rows.shape[0]
        if self.method == "factor":
            torch.matmul(self._standard_normal(k, self.rank), self._F.T, out=rows)
        else:
            rows.copy_(self._pathwise(k))
        rows += self._torch("mean", lambda dev, dt: torch.as_tensor(self._mean, device=dev, dtype=dt))
        if self._noise_sampler is not None:
            e = torch.empty_like(rows)
            self._noise_sampler.sample_into(e)
            rows += e
        elif self._noise_cov is not None and not (self.method == "factor" and self._folded):
            e = self._standard_normal(k, self.m)
            if self._noise_std is not None:
                e *= self._torch("std", lambda dev, dt: torch.as_tensor(self._noise_std, device=dev, dtype=dt))
            else:
                e = e @ self._torch("chol", lambda dev, dt: torch.as_tensor(self._noise_chol, device=dev, dtype=dt)).T
            rows += e

    def _standard_normal(self, k, n) -> torch.Tensor:
        """k x n standard normals: on the CPU from the NumPy stream, sample by sample (a fill does not depend on how
        it is split into chunks); on a GPU from the torch generator there."""
        streams = self.source.streams()
        if self.device.type == "cpu":
            return torch.from_numpy(streams.numpy().standard_normal((k, n))).to(self.dtype)
        return torch.randn((k, n), generator=streams.torch(self.device), device=self.device, dtype=self.dtype)

    def _pathwise(self, k) -> torch.Tensor:
        """scale G (f - mean) for k draws of the field: (k x m), float64."""
        f = self.field
        if getattr(f, "engine", None) == "ngsolve" and hasattr(f, "_ngs_rows"):
            transform = None if self.scale == 1.0 else self._scale_in_place
            if sp.issparse(self.G):  # evaluated in the sampling threads, from the float64 coefficients
                vals = f._ngs_rows(k, np.empty((k, self.m)), evaluate=self._ngs_operator(), add_mean=False,
                                   transform=transform)
            else:
                vals = f._ngs_rows(k, np.empty((k, self.space.ndof)), add_mean=False, transform=transform) @ self.G.T
            return torch.from_numpy(vals).to(self.device, torch.float64)
        if getattr(f, "engine", None) == "torch":  # on the field's device: zero-mean coefficients, then G there
            eta = torch.randn((f.nrand, k), generator=f.generator, device=f.device, dtype=f._work_dtype)
            c = f.apply(eta).to(torch.float64)
            return (self.scale * (self._operator_torch(c.device, torch.float64, free=True) @ c)).T.to(self.device)
        x = torch.as_tensor(f.sample(k))  # any other sampler (ndof x k)
        dev = torch_device_of(f)
        if dev is not None and x.device.type != "cpu":  # a low-rank model or Matheron posterior on a GPU: G there
            x = x.to(torch.float64) - device_copy(self, mean_array(f), x.device, torch.float64)[:, None]
            return (self.scale * (self._operator_torch(x.device, torch.float64) @ x)).T.to(self.device)
        x = x.to("cpu", torch.float64).numpy()
        y = np.asarray(self.G @ (x - np.asarray(f.mean)[:, None]))
        return torch.from_numpy(self.scale * y.T).to(self.device, torch.float64)

    def _scale_in_place(self, r):
        r *= self.scale

    def _ngs_operator(self):
        if getattr(self, "_ngs_g", None) is None:
            self._ngs_g = ngs_sparse(sp.csr_matrix(self.G))
        return self._ngs_g

    def _operator_torch(self, device, dtype, free=False):
        g = self._free_operator() if free else self.G

        def make(dev, dt):
            return sparse_to_torch(g, dev, dt) if sp.issparse(g) else torch.as_tensor(np.asarray(g), device=dev,
                                                                                     dtype=dt)

        return self._torch(("G", free), make, device, dtype)

    def _torch(self, name, make, device=None, dtype=None):
        device, dtype = device or self.device, dtype or self.dtype
        key = (name, str(device), dtype)
        if key not in self._torch_cache:
            self._torch_cache[key] = make(device, dtype)
        return self._torch_cache[key]

    # pickling (ngsrng.backend.Rebuildable): the NGSolve operator and the torch copies are made again on first use
    _transient = ("_torch_cache",)

    def _restore(self):
        self._torch_cache = {}

    @property
    def info(self) -> dict:
        noise = "" if self.plan["noise"] == "none" else " + e"
        return {"law": f"Gaussian observations y = G f{noise} (m = {self.m})", "plan": self.plan}

    def __repr__(self):
        return "ObservedGaussian(\n" + "\n".join(f"  {k}: {v}" for k, v in self.plan.items() if k != "field") + "\n)"


def _coefficients(loc, space, field_mean):
    """The mean as a coefficient vector (all dofs): the field's (None), or loc interpolated."""
    if loc is None:
        return np.zeros(space.ndof) if field_mean is None else np.asarray(field_mean, dtype=float)
    if isinstance(loc, np.ndarray) and loc.shape == (space.ndof,):
        return loc.astype(float)
    gf = ngs.GridFunction(space.fes)
    if isinstance(loc, ngs.GridFunction):
        gf.vec.data = loc.vec
    else:
        gf.Set(loc if isinstance(loc, ngs.CoefficientFunction) else ngs.CoefficientFunction(float(loc)))
    return gf.vec.FV().NumPy().copy()


def _free_memory(device) -> float:
    if torch.device(device).type == "cuda":
        return device_free_memory(device)
    avail = available_memory()
    return float(avail) if avail is not None else 4.0 * 2**30
