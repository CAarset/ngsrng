"""Covariance operators as data: pointwise variance k(x, x), traces, covariance functions and sample (Wishart)
covariances, for any Gaussian (prior, posterior, finite rank) or any covariance given as an operator.

A random function f in V_h with coefficient covariance Sigma (free dofs) has the covariance kernel
k_h(x, y) = phi(x)^T Sigma phi(y). Two facts make its diagonal cheap to represent exactly:

- k_h(x, x) on an element T involves Sigma_ij only for dofs i, j of T: the *local covariance*, i.e. the entries of
  Sigma in the sparsity pattern of the mass matrix. On T it is a polynomial of degree 2p in reference coordinates,
  so the variance is an FE function of L2(order 2p), exactly (also on curved elements, quads and hexes).
- The L2 trace is its integral: tr(C_h) = tr(Sigma M) = sum_ij Sigma_ij M_ij = int k_h(x, x) dx.

So variance maps and traces need Sigma only on local pairs, not samples. How those entries are obtained depends on
what is known about Sigma (:func:`local_covariance`):

- finite rank, Sigma = W W^T (LowRankGaussian: KL expansions, sample covariances, operator covariances and their
  posteriors): exact, r products per entry;
- a prior plus a low-rank update (posteriors, downdates, Matheron posteriors): the prior's local covariance (computed
  once, cached) plus the exact update (as in Bui-Thanh, Ghattas, Martin & Stadler 2013);
- otherwise "exact" from the n_free columns Sigma e_j (batched), "lowrank" (the exact local covariance of a randomised
  low-rank part plus a Monte Carlo estimate of the residual only: the pointwise analogue of Hutch++ / Diag++), or
  "mc" (Monte Carlo).

Traces (:func:`covariance_trace`) are available from the same data or by stochastic trace estimators that use only
applications of the covariance: Hutchinson, Hutch++ (Meyer, Musco, Musco & Woodruff 2021) and XNysTrace (Epperly,
Tropp & Webber 2024).
"""

from dataclasses import dataclass, field

import ngsolve as ngs
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .backend import compute_device, numpy_blas_threads, sparse_to_torch
from .randomized import euclidean_orth, gram_orth, randomized_svd
from .space import SpaceInfo, coefficient_vector

# Largest free-dof count for which "auto" computes the local covariance exactly from all columns of Sigma (one
# application per free dof) when nothing cheaper is known.
EXACT_LIMIT = 4000
CHUNK = 256
# Values per dense temporary when Sigma is applied to many columns (256 MB in float64). Applying Sigma = L L^T to
# CHUNK columns at once made noise-space blocks of nrand x 256 values, several GB for 1e5-1e6 dofs.
BLOCK_VALUES = 2**25


def block_columns(*rows) -> int:
    """Columns per block when applying Sigma (or L, L^T) to many vectors: at most CHUNK and at most BLOCK_VALUES
    values in the tallest (rows x columns) temporary, but at least 32, so that threaded solves stay busy."""
    return int(max(32, min(CHUNK, BLOCK_VALUES // max(1, *rows))))


def _np(x) -> np.ndarray:
    return x.detach().cpu().double().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=float)


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x))


def _up(x: np.ndarray, device) -> torch.Tensor:
    """A NumPy block on a torch device (float64)."""
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64)).to(device)


# ------------------------------------------------------------------ sources


class CovarianceSource:
    """What is known about a covariance Sigma (dual -> primal, free dofs), with NumPy maps on (nfree x k) blocks.

    ``sigma``: y -> Sigma y. Optional structure: a factor Sigma = L L^T (``l``: (nrand x k) -> (nfree x k), ``lt``
    its transpose), an exact finite-rank factor ``lowrank`` (W, nfree x r, Sigma = W W^T), or ``update``: a function
    returning (prior source, U, D) with Sigma = Sigma_prior + U D U^T.

    ``device``: a torch device when the source computes on one and the set-ups should run there (``compute``); then
    ``tl``, ``tlt`` and ``tsigma`` are the same maps on float64 tensors on that device (the NumPy maps stay
    available for everything else).
    """

    def __init__(self, space, sigma, *, l=None, lt=None, nrand=None, lowrank=None, update=None, mean=None,
                 owner=None, label="", device=None, tl=None, tlt=None, tsigma=None):
        self.space, self.sigma = space, sigma
        self.l, self.lt, self.nrand = l, lt, nrand
        self.lowrank, self.update, self.label = lowrank, update, label
        self.device, self.tl, self.tlt, self.tsigma = device, tl, tlt, tsigma
        self.mean = np.zeros(space.ndof) if mean is None else mean
        self.cache = _cache_of(owner)
        # NumPy BLAS threads during the computation: limited for the NGSolve engine, whose solves they would slow
        # down by busy-waiting between dense steps; None leaves them unchanged
        self.blas = _blas_threads(owner)

    @property
    def has_factor(self) -> bool:
        return self.l is not None

    def samples(self, n, rng):
        """n zero-mean samples (nfree x n); needs a factor. On the device (torch) when the source has one; the white
        noise comes from ``rng`` (NumPy) either way. Sample j is made from the j-th run of nrand consecutive draws,
        a block of columns at a time: the noise is never formed for all n samples at once (nrand x n values), and
        the samples do not depend on the block size."""
        step = block_columns(self.nrand, self.space.nfree)
        out = None
        for i in range(0, n, step):
            noise = rng.standard_normal((min(step, n - i), self.nrand)).T
            f = self.tl(_up(noise, self.device)) if self.device is not None else self.l(noise)
            if out is None:
                out = (torch.empty((f.shape[0], n), dtype=f.dtype, device=f.device) if isinstance(f, torch.Tensor)
                       else np.empty((f.shape[0], n)))
            out[:, i:i + f.shape[1]] = f
        return out


def _blas_threads(owner):
    from .specs import OperatorCovariance

    pipeline = getattr(owner, "prior", owner)  # a MatheronPosterior runs its prior's pipeline
    if getattr(pipeline, "engine", None) == "ngsolve":
        return getattr(pipeline, "_cpu_blas_threads", 1)
    if isinstance(owner, OperatorCovariance) and isinstance(owner.apply, (ngs.BaseMatrix, ngs.BilinearForm)):
        return 1
    return None


def _cache_of(owner) -> dict:
    """A cache dict on ``owner``, tied to its identity: shallow copies (e.g. posteriors derived from a prior by
    ``copy.copy``) must not share their original's results."""
    if owner is None:
        return {}
    try:
        stored = owner.__dict__.get("_ngsrng_covariance_cache")
        if stored is None or stored[0] != id(owner):
            stored = (id(owner), {})
            owner.__dict__["_ngsrng_covariance_cache"] = stored
    except AttributeError:  # e.g. pybind objects
        return {}
    return stored[1]


def as_covariance(source, fes=None, convention=None, compute="auto") -> CovarianceSource:
    """A :class:`CovarianceSource` from a Gaussian, LowRankGaussian, MatheronPosterior, :class:`OperatorCovariance`,
    NGSolve ``BaseMatrix`` or ``BilinearForm``, or a callable (the last four need ``fes``; ``convention`` as in
    OperatorCovariance). ``compute`` (see :func:`ngsrng.backend.compute_device`): whether set-ups run on the source's
    torch device ("auto": if it is a GPU) or with NumPy on the host."""
    from .lowrank import LowRankGaussian
    from .posterior import MatheronPosterior
    from .specs import OperatorCovariance

    if isinstance(source, CovarianceSource):
        return source
    if isinstance(source, LowRankGaussian):
        s = source.space
        ops = _FiniteRank(source.b[s.free] @ source.rinv)
        dev = compute_device(compute, source, "a covariance set-up")  # a LowRankGaussian on a GPU, or None
        tops = _FiniteRank(source._factor(True)) if dev is not None else None
        return CovarianceSource(s, ops.sigma, l=ops.factor, lt=ops.factor_t, nrand=ops.w.shape[1], lowrank=ops.w,
                                mean=source.mean, owner=source, label=f"finite rank {ops.w.shape[1]}",
                                **_torch_maps(dev, tops))
    if hasattr(source, "apply") and hasattr(source, "apply_t") and hasattr(source, "space"):
        s = source.space
        step = block_columns(source.nrand, s.nfree)
        ops = _Factored(source, step)
        dev = compute_device(compute, source, "a covariance set-up")
        update = None
        if isinstance(source, MatheronPosterior):
            update = _MatheronUpdate(source, compute)
        elif getattr(source, "_prior", None) is not None and hasattr(source, "_maps"):
            update = _derived_update(source, compute)
        return CovarianceSource(s, ops.sigma, l=ops.factor, lt=ops.factor_t, nrand=source.nrand,
                                update=update,
                                mean=source.mean, owner=source, label=type(source).__name__,
                                **_torch_maps(dev, _TorchFactored(source, step, dev) if dev is not None else None))
    if fes is None:
        raise ValueError("a covariance given as an operator needs its space: pass fes=")
    if not isinstance(source, OperatorCovariance):
        source = OperatorCovariance(source, convention=convention)
    compute_device(compute, source, "a covariance set-up")  # "device" raises: an operator acts on NumPy blocks
    s = SpaceInfo(fes)
    return CovarianceSource(s, source.sigma(s), owner=source, label=f"operator ({source.convention})")


def _torch_maps(dev, ops) -> dict:
    if dev is None:
        return {}
    return dict(device=dev, tl=ops.factor, tlt=ops.factor_t, tsigma=ops.sigma)


class _FiniteRank:
    """Sigma = W W^T with an explicit W (nfree x r)."""

    def __init__(self, w):
        self.w = w

    def sigma(self, y):
        return self.w @ (self.w.T @ y)

    def factor(self, e):
        return self.w @ e

    def factor_t(self, y):
        return self.w.T @ y


class _Factored:
    """Sigma = L L^T through a sampler's ``apply`` (L) and ``apply_t`` (L^T), ``step`` columns at a time."""

    def __init__(self, source, step):
        self.source, self.step = source, step

    def factor(self, e):
        st = self.step
        return np.hstack([_np(self.source.apply(_t(e[:, i:i + st]))) for i in range(0, e.shape[1], st)])

    def factor_t(self, y):
        st = self.step
        return np.hstack([_np(self.source.apply_t(_t(y[:, i:i + st]))) for i in range(0, y.shape[1], st)])

    def sigma(self, y):  # block by block: never the whole nrand x k noise-space intermediate
        st = self.step
        return np.hstack([self.factor(self.factor_t(y[:, i:i + st])) for i in range(0, y.shape[1], st)])


class _TorchFactored:
    """Sigma = L L^T through a sampler's ``apply`` and ``apply_t`` on float64 tensors on its device, ``step`` columns
    at a time."""

    def __init__(self, source, step, device):
        self.source, self.step, self.device = source, step, device

    def _blocks(self, op, x):
        out = None
        for i in range(0, x.shape[1], self.step):
            col = op(x[:, i:i + self.step].contiguous()).to(torch.float64)
            if out is None:
                out = torch.empty((col.shape[0], x.shape[1]), dtype=torch.float64, device=col.device)
            out[:, i:i + col.shape[1]] = col
        return out

    def factor(self, e):
        return self._blocks(self.source.apply, e)

    def factor_t(self, y):
        return self._blocks(self.source.apply_t, y)

    def sigma(self, y):  # block by block: never the whole nrand x k noise-space intermediate
        return self._blocks(lambda b: self.factor(self.factor_t(b)), y)


class _MatheronUpdate:
    """Sigma = Sigma_prior - B K_d^-1 B^T of a Matheron posterior."""

    def __init__(self, source, compute="auto"):
        self.source, self.compute = source, compute

    def __call__(self):
        src = self.source
        return (as_covariance(src.prior, compute=self.compute), src.bmat,
                -sla.cho_solve(src.kd_chol, np.eye(src.bmat.shape[1])))


class _DerivedUpdate:
    """Sigma - Sigma_prior = Q D Q^T for a Gaussian derived by low-rank maps (see :func:`_derived_update`)."""

    def __init__(self, g, ranks, compute="auto"):
        self.g, self.ranks, self.compute = g, ranks, compute

    def __call__(self):
        g = self.g
        prior = as_covariance(g._prior, compute=self.compute)
        n = g.space.nfree
        ell = min(int(sum(self.ranks)) + 10, n)
        y0 = np.random.default_rng(0).standard_normal((n, ell))
        if prior.device is not None:  # on the prior's device: only Q and D come back
            mine = _TorchFactored(g, block_columns(g.nrand, n), prior.device)

            def diff_t(y):
                return mine.sigma(y) - prior.tsigma(y)

            y = diff_t(_up(y0, prior.device))
            u, sv, _ = torch.linalg.svd(y, full_matrices=False)
            q = u[:, sv > (sv[0] * 1e-11 if sv.numel() else 0.0)]
            d = q.T @ diff_t(q)
            return prior, _np(q), _np(0.5 * (d + d.T))

        def diff(y):
            return _sigma(g, y) - prior.sigma(y)

        y = diff(y0)
        u, sv, _ = np.linalg.svd(y, full_matrices=False)
        q = u[:, sv > (sv[0] * 1e-11 if sv.size else 0.0)]
        d = q.T @ diff(q)
        return prior, q, 0.5 * (d + d.T)


def _derived_update(g, compute="auto"):
    """Sigma - Sigma_prior = Q D Q^T for a Gaussian derived from a prior by low-rank maps (posteriors, downdates).

    The difference has rank at most r = sum of the noise-map ranks + 2 x the FE-update ranks, so a sketch with
    r + 10 columns captures its range exactly (with probability one); D = Q^T (Sigma - Sigma_prior) Q. Cost: 2 (r + 10)
    applications of each covariance.
    """
    ranks = [getattr(m, "rank", None) for m in g._maps] + [2 * getattr(m, "rank", np.nan) for m in g._post_maps]
    if any(r is None or not np.isfinite(r) for r in ranks):
        return None
    return _DerivedUpdate(g, ranks, compute)


def _sigma(g, y):
    step = block_columns(g.nrand, g.space.nfree)
    return np.hstack([_np(g.apply(g.apply_t(_t(y[:, i:i + step])).to(torch.float64)))
                      for i in range(0, y.shape[1], step)])


# ------------------------------------------------------------------ local covariance


def _variance_projection(space: SpaceInfo):
    """L2(order 2p) space, and the same-rule projection of point values into it (exact for functions in it)."""
    from .quadrature import QuadratureRule

    p = space.order
    target = SpaceInfo(ngs.L2(space.mesh, order=2 * p))
    order = max(4 * p, 2)
    src = QuadratureRule(space, order)
    dst = QuadratureRule(target, order)
    e_dst = dst.evaluation_full
    lu = spla.splu((e_dst.T @ sp.diags(dst.weights) @ e_dst).tocsc())

    def project(values):
        return lu.solve(e_dst.T @ (dst.weights[:, None] * values.reshape(len(dst.weights), -1)))

    return target.fes, src.evaluation, project


@dataclass
class LocalCovariance:
    """Sigma on element-local dof pairs (free dofs): Sigma_loc = S + sum_k U_k D_k U_k^T + (Monte Carlo part).

    ``pattern``: sparse S on the pattern of M (or None); ``terms``: (U, D) pairs (D None for the identity);
    ``mc``: (A, B) with the estimate (1/s) sum_j a_j b_j^T (B is A for samples), carrying a standard error.
    ``method`` and ``applications`` (applications of Sigma, L or L^T) describe how it was computed.
    """

    space: SpaceInfo
    pattern: sp.csr_matrix | None = None
    terms: list = field(default_factory=list)
    mc: tuple | None = None
    method: str = ""
    applications: int = 0

    def plus(self, terms, method=None, applications=0) -> "LocalCovariance":
        """The local covariance of Sigma + sum_k U_k D_k U_k^T: ``terms`` are (U, D) pairs, U (nfree x r) and D
        (r x r, or None for the identity), e.g. a posterior update (:meth:`ngsrng.LinearObservations.posterior_update`).
        ``method`` and ``applications`` record how the terms were computed (applications of Sigma, L or L^T)."""
        method = f"{self.method} + {len(terms)} low-rank term(s)" if method is None else method
        return LocalCovariance(self.space, self.pattern, self.terms + list(terms), self.mc, method,
                               self.applications + applications)

    @property
    def exact(self) -> bool:
        return self.mc is None

    def values(self, e, stderr=False):
        """Variance k_h(x, x) at points, from their evaluation matrix e (npoints x nfree); with ``stderr`` also the
        standard error of the Monte Carlo part (zero where exact)."""
        e = sp.csr_matrix(e)
        v = np.zeros(e.shape[0])
        if self.pattern is not None:
            v += np.asarray(e.multiply(e @ self.pattern).sum(axis=1)).ravel()
        for u, d in self.terms:
            eu = e @ u
            v += np.einsum("ij,ij->i", eu if d is None else eu @ d, eu)
        se = np.zeros_like(v)
        if self.mc is not None:
            a, b = self.mc
            ea = e @ a
            t = ea * (ea if b is a else e @ b)
            v += t.mean(axis=1)
            se = t.std(axis=1, ddof=1) / np.sqrt(t.shape[1])
        return (v, se) if stderr else v

    def trace(self):
        """(tr(Sigma M), standard error): the L2 trace of the covariance operator, int k_h(x, x) dx."""
        m = self.space.mass_free
        t = 0.0
        if self.pattern is not None:
            t += float(self.pattern.multiply(m).sum())
        for u, d in self.terms:
            mu = m @ u
            t += float(np.sum(u * mu)) if d is None else float(np.sum((u.T @ mu) * d))
        se = 0.0
        if self.mc is not None:
            a, b = self.mc
            per = np.sum(a * (m @ b), axis=0)
            t += float(per.mean())
            se = float(per.std(ddof=1) / np.sqrt(len(per)))
        return t, se

    def variance(self, stderr=False):
        """The variance k_h(x, x) as a GridFunction of L2(order 2p): exact (the FE variance is in that space). With
        ``stderr``, also the standard error of the Monte Carlo part (projected; zero for exact methods)."""
        if not hasattr(self, "_projection"):
            self._projection = _variance_projection(self.space)
        fes, e, project = self._projection
        v, se = self.values(e, stderr=True)
        out = []
        for vals in (v, se) if stderr else (v,):
            gf = ngs.GridFunction(fes)
            gf.vec.FV().NumPy()[:] = project(vals)[:, 0]
            out.append(gf)
        return tuple(out) if stderr else out[0]

    def std(self) -> ngs.CoefficientFunction:
        """Pointwise standard deviation sqrt(k_h(x, x)) (a CoefficientFunction)."""
        v = self.variance()
        return ngs.sqrt(ngs.IfPos(v, v, 0.0))


def _check_method_arg(method):
    if not isinstance(method, str):  # e.g. a space passed positionally: NGSolve's __eq__ would raise a TypeError
        raise TypeError(f"method must be a string, not {type(method).__name__} (pass a space as fes=...)")


def local_covariance(source, method="auto", *, fes=None, convention=None, rank=100, samples=None,
                     power_iterations=2, seed=0, compute="auto") -> LocalCovariance:
    """The covariance on element-local dof pairs, from which pointwise variances and traces follow.

    ``method``:

    - "auto": "exact" when that is cheap (finite rank; prior plus low-rank update, with the prior by "auto"; up to
      ``EXACT_LIMIT`` free dofs), "lowrank" otherwise;
    - "exact": exact. Finite rank: from the factor. Prior plus update: the prior's local covariance (exact, cached on
      the prior) plus the exact update. Otherwise all columns Sigma e_j: n_free applications of Sigma, batched;
    - "lowrank": a randomised basis Q of the dominant range (``rank`` columns, ``power_iterations`` subspace
      iterations), the exact local covariance of the part of Sigma it captures (Sigma - (I-P) Sigma (I-P)^T,
      P = Q Q^T M), and Monte Carlo with ``samples`` (default 32) samples for the residual (I-P) Sigma (I-P)^T only
      (white-noise probes for apply-only covariances). Unbiased; the error shrinks with the residual's variance, so
      it pays off for fast spectral decay (smooth fields, long correlation lengths). The basis is computed in the
      smaller of the noise space and FE space (FE space for element-local white noise: less memory and time; the
      result's ``method`` says which). For a prior plus low-rank update, applied to the prior;
    - "mc": Monte Carlo with ``samples`` (default 1000) samples (needs a sampler: not for operator covariances).

    ``compute``: where the set-up runs: "auto" (on the source's GPU, if it has one: the applications of L, L^T or
    Sigma and the dense algebra stay there, only the final blocks come to the host), "device" (the source's torch
    device, also a CPU one), "host" (NumPy on the CPU). The result is the same to rounding (the random probes come
    from NumPy either way) and is cached on the source object.
    """
    _check_method_arg(method)
    src = as_covariance(source, fes, convention, compute)
    requested = method
    if method == "auto":
        cheap = src.lowrank is not None or src.update is not None or src.space.nfree <= EXACT_LIMIT
        method = "exact" if cheap else "lowrank"
    if method not in ("exact", "lowrank", "mc"):
        raise ValueError("method must be 'auto', 'exact', 'lowrank' or 'mc'")
    if samples is None:
        samples = 1000 if method == "mc" else 32
    # a prior plus update: the prior part by the requested method ("auto" stays "auto" for the prior)
    prior_method = ("auto" if requested == "auto" else method) if (
        src.update is not None and src.lowrank is None and method != "mc") else None
    if prior_method is not None:
        key = ("update", prior_method, rank, samples, power_iterations, seed)
    else:
        key = (method,) if method == "exact" else (method, rank, samples, power_iterations, seed)
    key += (str(src.device) if src.device is not None else "host",)
    if key in src.cache:
        return src.cache[key]
    with numpy_blas_threads(src.blas):
        lc = _local_covariance(src, method, prior_method, rank, samples, power_iterations, seed)
    src.cache[key] = lc
    return lc


def _local_covariance(src, method, prior_method, rank, samples, power_iterations, seed):
    rng = np.random.default_rng(seed)
    if method == "exact" and src.lowrank is not None:
        lc = LocalCovariance(src.space, terms=[(src.lowrank, None)], method=f"exact ({src.label})")
    elif prior_method is not None:
        prior, u, d = src.update()
        base = local_covariance(prior, prior_method, rank=rank, samples=samples, power_iterations=power_iterations,
                                seed=seed)
        lc = base.plus([(u, d)], f"prior ({base.method}) + exact rank-{u.shape[1]} update",
                       2 * u.shape[1] + 20)
    elif method == "exact":
        lc = _columns(src)
    elif method == "lowrank":
        lc = _lowrank(src, rank, samples, power_iterations, rng)
    else:
        lc = _monte_carlo(src, samples, rng)
    return lc


def pointwise_variance(source, method="auto", **kwargs) -> ngs.GridFunction:
    """The pointwise variance k_h(x, x) of a random function as a GridFunction of L2(order 2p) (exact for the
    exact methods; see :func:`local_covariance` for the methods and options)."""
    return local_covariance(source, method, **kwargs).variance()


def _columns(src) -> LocalCovariance:
    s = src.space
    pattern = s.mass_free.tocsc()
    rows, ptr = pattern.indices, pattern.indptr
    vals = np.empty(len(rows))
    n = s.nfree
    if src.device is not None:  # unit blocks on the device; only the pattern's entries come back
        dev = src.device
        step = block_columns(n, src.nrand or 0)
        for j0 in range(0, n, step):
            j1 = min(j0 + step, n)
            unit = torch.zeros((n, j1 - j0), dtype=torch.float64, device=dev)
            unit[torch.arange(j0, j1, device=dev), torch.arange(j1 - j0, device=dev)] = 1.0
            cols = src.tsigma(unit)
            lo, hi = ptr[j0], ptr[j1]
            col_of = np.repeat(np.arange(j1 - j0), np.diff(ptr[j0:j1 + 1]))
            vals[lo:hi] = _np(cols[torch.from_numpy(rows[lo:hi]).to(dev), torch.from_numpy(col_of).to(dev)])
        sigma = pattern.copy()
        sigma.data = vals
        return LocalCovariance(s, pattern=sigma.tocsr(), method="exact (all columns)",
                               applications=n * (2 if src.has_factor else 1))
    for j0 in range(0, n, CHUNK):
        j1 = min(j0 + CHUNK, n)
        unit = np.zeros((n, j1 - j0))
        unit[np.arange(j0, j1), np.arange(j1 - j0)] = 1.0
        cols = src.sigma(unit)
        for k, j in enumerate(range(j0, j1)):
            vals[ptr[j]:ptr[j + 1]] = cols[rows[ptr[j]:ptr[j + 1]], k]
    sigma = pattern.copy()
    sigma.data = vals
    return LocalCovariance(s, pattern=sigma.tocsr(), method="exact (all columns)",
                           applications=n * (2 if src.has_factor else 1))


def _white_noise(space):
    """White-noise sampler z = H omega ~ N(0, M) (free dofs), with H^T; and the M^-1 solve."""
    from .noise import DiagonalNoise, LocalGramNoise

    noise = DiagonalNoise(space.mass_free) if space.mass_is_diagonal else LocalGramNoise(space).explicit()
    noise.setup(torch.device("cpu"), torch.float64)
    if space.mass_is_diagonal:
        dm = space.mass_free.diagonal()[:, None]
        minv = lambda y: y / dm  # noqa: E731
    else:
        from .distributions.bernstein import SymmetricFactor  # sparse Cholesky: SuperLU was 3-100x slower

        minv = SymmetricFactor(space.mass_free).solve
    return noise, minv


def _stable_orth(mass):
    """M-orthonormaliser for powered FE-space blocks: a Euclidean QR first, then Cholesky QR in the M inner product.
    The columns of (Sigma M)^q Y spread over the eigenvalues of Sigma M to the power q, and Cholesky QR alone squares
    that spread (its Gram matrix), so it would fail or drop the small directions; after the Euclidean QR the M-Gram
    matrix is as well conditioned as M."""
    orth = gram_orth(mass)
    return lambda y: orth(euclidean_orth(y))


def _subspace_iteration(y, sigma, mass_op, orth, power_iterations):
    """M-orthonormal basis of the range of (Sigma M)^q y, re-orthonormalised after every application of Sigma."""
    for _ in range(power_iterations):
        y = sigma(mass_op @ orth(y))
    return orth(y)


def _fe_start(src, k, rng):
    """Start block of the FE-space subspace iteration, Sigma D^(1/2) Omega (D = diag M, Omega nfree x k standard
    normal), on the source's device if it has one; and its cost in applications of L, L^T or Sigma. D^(1/2) Omega is
    white noise (covariance M) up to the spectral equivalence of diag M and M, which holds uniformly in the mesh size,
    also on graded meshes (M Omega, covariance M^2, would weight the large elements); it is drawn in FE space, not in
    the noise space, where nrand x k draws would cost more than the iteration."""
    z = np.sqrt(src.space.mass_free.diagonal())[:, None] * rng.standard_normal((src.space.nfree, k))
    if src.device is not None:
        return src.tsigma(_up(z, src.device)), 2 * k
    return src.sigma(z), (2 if src.has_factor else 1) * k


def _basis(src, rank, power_iterations, rng):
    """M-orthonormal Q (nfree x k) of the dominant range, and the exact captured part as a term [Q, Y] D [Q, Y]^T
    = Q Y^T + Y Q^T - Q B Q^T (Y = Sigma M Q, B = Q^T M Y); returns (Q, term, applications, where). With a device
    source the basis is computed there and Q is returned there (the term on the host). The captured part is exact
    and the Monte Carlo residual unbiased for any Q; Q only decides how much variance is left to Monte Carlo.

    With a factor Sigma = L L^T and no more noise than free dofs (nrand <= nfree), Q is the randomised SVD of L in
    the noise space: the range of (Sigma M)^q L Omega (q = ``power_iterations``), (2q + 3) k applications of L or
    L^T. Otherwise (more noise than free dofs: element-local or stiffness-shaped white noise; or an operator
    covariance) Q comes from subspace iteration on Sigma M in FE space, on nfree x k blocks: the range of
    (Sigma M)^q Sigma D^(1/2) Omega (:func:`_fe_start`), 2 (q + 2) k applications of L or L^T, or (q + 2) k of Sigma.
    Note: the FE-space form and its analysis (half a power iteration further than the noise-space start, for one more
    application per column; the start white noise up to a mesh-independent factor) are recent and still to be
    double-checked; the measurements agree with them (same errors and standard errors as the noise-space form, less
    memory and time where nrand is several times nfree).
    """
    s = src.space
    n = s.nfree
    if src.has_factor and src.nrand <= n:
        return _noise_basis(src, rank, power_iterations, rng) + ("noise space",)
    dev = src.device  # a factor on the device (operators have no device)
    mass = s.mass_free
    mass_op = sparse_to_torch(mass, dev, torch.float64) if dev is not None else mass
    sigma = src.tsigma if dev is not None else src.sigma
    per = 2 if src.has_factor else 1  # applications of L, L^T or Sigma per application of Sigma
    k = min(rank, n)
    y, apps = _fe_start(src, k, rng)
    q = _subspace_iteration(y, sigma, mass_op, _stable_orth(mass), power_iterations)
    y = sigma(mass_op @ q)
    apps += per * (power_iterations + 1) * k
    b = _np(q.T @ (mass_op @ y))
    b = 0.5 * (b + b.T)
    r = q.shape[1]
    d = np.block([[-b, np.eye(r)], [np.eye(r), np.zeros((r, r))]])
    qy = _np(torch.cat([q, y], dim=1)) if dev is not None else np.hstack([q, y])
    return q, (qy, d), apps, "FE space"


def _noise_basis(src, rank, power_iterations, rng):
    """_basis by the randomised SVD of the factor L, in the noise space (nrand <= nfree)."""
    s = src.space
    mass = s.mass_free
    n = s.nfree
    if src.device is not None:
        mass_t = sparse_to_torch(mass, src.device, torch.float64)
        k = min(rank, n, src.nrand)
        q, sv, v = randomized_svd(src.tl, lambda y: src.tlt(mass_t @ y), src.nrand, k, oversampling=0,
                                  power_iterations=power_iterations, seed=int(rng.integers(2**31)),
                                  orth=gram_orth(mass), device=src.device)
        y = src.tl(v * sv)  # Sigma M Q = L (L^T M Q) = L V S
        b = np.diag(_np(sv) ** 2)
        apps = (2 * power_iterations + 2) * k + q.shape[1]
        r = q.shape[1]
        d = np.block([[-b, np.eye(r)], [np.eye(r), np.zeros((r, r))]])
        return q, (_np(torch.cat([q, y], dim=1)), d), apps
    k = min(rank, n, src.nrand)
    q, sv, v = randomized_svd(src.l, lambda y: src.lt(mass @ y), src.nrand, k, oversampling=0,
                              power_iterations=power_iterations, seed=int(rng.integers(2**31)),
                              orth=gram_orth(mass))
    y = src.l(v * sv)  # Sigma M Q = L (L^T M Q) = L V S
    b = np.diag(sv**2)
    apps = (2 * power_iterations + 2) * k + q.shape[1]
    r = q.shape[1]
    d = np.block([[-b, np.eye(r)], [np.eye(r), np.zeros((r, r))]])
    return q, (np.hstack([q, y]), d), apps


def _lowrank(src, rank, samples, power_iterations, rng) -> LocalCovariance:
    s = src.space
    mass = s.mass_free
    q, term, apps, where = _basis(src, rank, power_iterations, rng)
    if src.device is not None:  # residual samples on the device too; only they come back
        mass_t = sparse_to_torch(mass, src.device, torch.float64)
        f = src.samples(samples, rng)
        g = _np(f - q @ (q.T @ (mass_t @ f)))
        mc = (g, g)
        apps += samples
    elif src.has_factor:
        f = src.samples(samples, rng)
        g = f - q @ (q.T @ (mass @ f))  # residual samples (I - P) f ~ N(0, (I-P) Sigma (I-P)^T)
        mc = (g, g)
        apps += samples
    else:
        noise, minv = _white_noise(s)
        w = _np(noise.apply(_t(rng.standard_normal((noise.nrand, samples)))))
        a = src.sigma(w - mass @ (q @ (q.T @ w)))  # Sigma (I - P)^T w
        a = a - q @ (q.T @ (mass @ a))  # (I - P) Sigma (I - P)^T w;  E[a (M^-1 w)^T] = residual covariance
        mc = (a, minv(w))
        apps += samples
    return LocalCovariance(s, terms=[term], mc=mc, applications=apps,
                           method=f"low rank {q.shape[1]} (exact, basis in {where}) + Monte Carlo residual "
                                  f"({samples} samples)")


def _monte_carlo(src, samples, rng) -> LocalCovariance:
    if not src.has_factor:
        # Without a square root of Sigma, the only unbiased pointwise estimators are cross products such as
        # (Sigma w)(x) (M^-1 w)(x), whose relative variance grows like the number of dofs per correlation volume
        # (mesh dependent). They are fine for the small, rough residual in "lowrank", not on their own.
        raise ValueError("Monte Carlo variances need samples, i.e. a Gaussian with a sampler; for a covariance given "
                         "as an operator use method='exact' or 'lowrank'")
    f = _np(src.samples(samples, rng))
    return LocalCovariance(src.space, mc=(f, f), applications=samples, method=f"Monte Carlo ({samples} samples)")


# ------------------------------------------------------------------ traces


@dataclass
class TraceEstimate:
    """tr(C) = tr(Sigma M) = int k(x, x) dx (or tr(Sigma W) for a weight W), with a standard error (0 when exact) and
    the cost in applications of Sigma, L or L^T."""

    value: float
    stderr: float
    method: str
    applications: int

    def __float__(self):
        return float(self.value)


def covariance_trace(source, method="auto", *, fes=None, convention=None, probes=60, seed=0,
                     variance_method="auto", weight=None, **local_options) -> TraceEstimate:
    """The L2 trace tr(C) = tr(Sigma M) = int k(x, x) dx of a covariance, or with ``weight`` W the weighted trace
    tr(Sigma W) = E ||f - m||_W^2.

    ``weight``: a symmetric positive semidefinite map W on free-dof coefficient blocks (NumPy (nfree x k) -> (nfree x
    k), or a matrix), default the mass matrix M. For a linear image B f of the field, e.g. a PDE solution driven by f,
    W = B^T M_B B gives the mean energy E ||B (f - m)||^2 of the image in the image space's mass matrix M_B. With a
    weight, the source needs a factor Sigma = L L^T (Gaussians, low-rank models, posteriors) and the method is one of
    the randomised estimators ("auto": "xnystrace"), on A = L^T W L.

    ``method``:

    - "auto": "exact" when cheap (as in :func:`local_covariance`), "xnystrace" otherwise;
    - "exact": from the exact local covariance;
    - "variance": int k(x, x) dx from :func:`local_covariance` with method ``variance_method`` and ``local_options``
      (e.g. rank, samples);
    - "hutchinson": mean of z^T Sigma z over ``probes`` white-noise probes z ~ N(0, M), or of ||f||^2 over samples
      f ~ N(0, C) when a sampler is known (same estimator law, half the cost). Error ~ sqrt(2 tr(C^2) / probes);
    - "hutch++": Hutch++ (Meyer, Musco, Musco & Woodruff 2021): the exact trace on a randomised dominant subspace
      plus Hutchinson on the rest; error ~ 1/probes for fast spectral decay;
    - "xnystrace": XNysTrace (Epperly, Tropp & Webber 2024): a Nystrom approximation with leave-one-out
      ("exchangeable") correction, using every probe for both; the most accurate per application for PSD operators.

    The randomised estimators work on the Euclidean PSD operator A = L^T M L (when a factor Sigma = L L^T is known)
    or A = H^T Sigma H (white-noise factor M = H H^T), both with tr(A) = tr(Sigma M).
    """
    _check_method_arg(method)
    src = as_covariance(source, fes, convention)
    if weight is not None:
        if method == "auto":
            method = "xnystrace"
        if method not in ("hutchinson", "hutch++", "xnystrace"):
            raise ValueError("a weighted trace needs a randomised method: 'xnystrace', 'hutch++' or 'hutchinson'")
        if not src.has_factor:
            raise ValueError("a weighted trace needs a source with a factor Sigma = L L^T (a Gaussian, a low-rank "
                             "model or a posterior), not an apply-only covariance")
    if method == "auto":
        cheap = src.lowrank is not None or src.update is not None or src.space.nfree <= EXACT_LIMIT
        method = "exact" if cheap else "xnystrace"
    if method in ("exact", "variance"):
        lc = local_covariance(src, "exact" if method == "exact" else variance_method, seed=seed, **local_options)
        value, se = lc.trace()
        return TraceEstimate(value, se, f"{method}: {lc.method}", lc.applications)
    if method not in ("hutchinson", "hutch++", "xnystrace"):
        raise ValueError("method must be 'auto', 'exact', 'variance', 'hutchinson', 'hutch++' or 'xnystrace'")
    with numpy_blas_threads(src.blas):
        return _randomized_trace(src, method, probes, seed, weight)


def _randomized_trace(src, method, probes, seed, weight=None):
    rng = np.random.default_rng(seed)
    mass = src.space.mass_free
    if weight is None:
        def apply_w(y):
            return mass @ y
    elif callable(weight):
        def apply_w(y):
            return np.asarray(weight(y), dtype=float)
    else:
        def apply_w(y):
            return np.asarray(weight @ y, dtype=float)
    if src.has_factor:
        n, cost_apply, cost_quad = src.nrand, 2, 1

        def apply_a(x):
            return src.lt(apply_w(src.l(x)))

        def quad(x):
            lx = src.l(x)
            return np.sum(lx * apply_w(lx), axis=0)
    else:
        noise, _ = _white_noise(src.space)
        n, cost_apply, cost_quad = noise.nrand, 1, 1

        def apply_a(x):
            return _np(noise.apply_t(_t(src.sigma(_np(noise.apply(_t(x)))))))

        def quad(x):
            hx = _np(noise.apply(_t(x)))
            return np.sum(hx * src.sigma(hx), axis=0)

    if method == "hutchinson":
        t = quad(rng.standard_normal((n, probes)))
        return TraceEstimate(float(t.mean()), float(t.std(ddof=1) / np.sqrt(probes)), "hutchinson",
                             probes * cost_quad)
    if method == "hutch++":
        k = max(probes // 3, 1)
        q, _ = np.linalg.qr(apply_a(rng.standard_normal((n, k))))
        exact = float(np.trace(q.T @ apply_a(q)))
        g = rng.standard_normal((n, k))
        t = quad(g - q @ (q.T @ g))
        return TraceEstimate(exact + float(t.mean()), float(t.std(ddof=1) / np.sqrt(k)), "hutch++",
                             2 * k * cost_apply + k * cost_quad)
    value, se = xnystrace(apply_a, n, probes, rng)
    return TraceEstimate(value, se, "xnystrace", probes * cost_apply)


def xnystrace(apply_a, n, m, rng):
    """XNysTrace (Epperly, Tropp & Webber 2024) for a PSD operator on R^n with m probes: (estimate, standard error).

    ``apply_a``: NumPy (n x k) -> (n x k), any symmetric positive semidefinite operator (matrix-free); ``rng``: a
    ``numpy.random.Generator`` for the Gaussian probes. Public: :func:`covariance_trace` uses it for covariances.

    With Y = A Omega, G = Omega^T Y = C^T C and Z = Y C^-1 (the Nystrom approximation is Z Z^T), the leave-one-out
    estimate for probe i is tr(Z Z^T) - (||Z v_i||^2 - 1) / ||v_i||^2, v_i = C^-T e_i: the trace of the Nystrom
    approximation without probe i plus the Hutchinson correction omega_i^T (A - A_(i)) omega_i = 1 / X_ii
    (X = G^-1). A shift nu = eps ||Y||_F keeps G positive definite and is subtracted again.
    """
    om = rng.standard_normal((n, m))
    y = apply_a(om)
    nu = np.finfo(float).eps * np.linalg.norm(y)
    y = y + nu * om
    g = om.T @ y
    g = 0.5 * (g + g.T)
    for _ in range(8):
        try:
            c = np.linalg.cholesky(g).T
            break
        except np.linalg.LinAlgError:  # numerically indefinite: shift further
            extra = 10 * nu + np.finfo(float).eps * np.abs(g).max()
            y, g, nu = y + extra * om, g + extra * (om.T @ om), nu + extra
    ci = sla.solve_triangular(c, np.eye(m))  # C^-1 (upper)
    z = y @ ci
    tz = z.T @ z
    v = ci.T  # v[:, i] = C^-T e_i
    xdiag = np.sum(v * v, axis=0)
    xtx = np.einsum("ki,kl,li->i", v, tz, v)
    est = np.trace(tz) - (xtx - 1.0) / xdiag - nu * n
    return float(est.mean()), float(est.std(ddof=1) / np.sqrt(m))


# ------------------------------------------------------------------ covariance functions and sample covariances


def covariance_function(source, point, *, correlation=False, fes=None, convention=None):
    """The covariance k_h(x0, .) = Sigma phi(x0) of the value at ``point`` with the whole field, as a GridFunction
    (one application of Sigma). ``correlation=True`` divides by sqrt(k(x0, x0) k(x, x)) (exact variance by
    :func:`local_covariance`; a CoefficientFunction)."""
    from .observations import observation_operator

    src = as_covariance(source, fes, convention)
    s = src.space
    row = observation_operator(s.fes, points=np.atleast_2d(np.asarray(point, dtype=float)))
    col = src.sigma(row[:, s.free].toarray().T)
    gf = ngs.GridFunction(s.fes)
    gf.vec.FV().NumPy()[s.free] = col[:, 0]
    if not correlation:
        return gf
    v0 = float((row[:, s.free] @ col)[0, 0])
    exact = [lc for lc in src.cache.values() if lc.exact]  # an exact one if already computed
    var = (exact[0] if exact else local_covariance(src)).variance()
    return gf / ngs.sqrt(v0 * ngs.IfPos(var, var, 1e-300))


def functional_covariance(source, operator=None, *, fes=None, convention=None, **observations) -> np.ndarray:
    """The covariance matrix G Sigma G^T of finitely many linear functionals of the field: point values, local
    averages, LinearForms, matrices or matrix-free maps (``operator`` and ``observations`` as in
    :func:`ngsrng.observation_operator`). Exact, one application of Sigma per functional. Its diagonal is the exact
    variance at points; for a posterior it is the predictive covariance of noise-free observations."""
    from .observations import dense_if_full, observation_operator

    src = as_covariance(source, fes, convention)
    s = src.space
    g = dense_if_full(observation_operator(s.fes, operator, **observations))[:, s.free]
    gt = g.T.tocsc() if sp.issparse(g) else g.T
    out = np.empty((g.shape[0], g.shape[0]))
    with numpy_blas_threads(src.blas):
        for i in range(0, g.shape[0], CHUNK):  # nfree x CHUNK at a time, not nfree x m
            blk = gt[:, i:i + CHUNK]
            blk = blk.toarray() if sp.issparse(blk) else np.ascontiguousarray(blk)
            out[:, i:i + CHUNK] = np.asarray(g @ src.sigma(blk))
    return 0.5 * (out + out.T)


def sample_covariance(source, n=None, *, fes=None, mean=None, seed=None, device=None):
    """The sample covariance of n samples as a finite-rank Gaussian N(m, C_n) (an ensemble model).

    ``source``: a Gaussian (or any object with ``sample``) from which n samples are drawn, or the samples themselves
    (an (n x ndof) array as returned by ``rng.normal(size=n)``, or a list of GridFunctions; then ``fes`` is needed).
    With the known mean m (default for Gaussians), C_n = (1/n) sum_j (f_j - m)(f_j - m)*, and n C_n ~ W_n(C), a
    Wishart operator with n degrees of freedom. With ``mean="sample"`` (default for given samples) the sample mean
    and 1/(n - 1): (n - 1) C_n ~ W_{n-1}(C). In both cases E[C_n] = C and
    E ||C_n - C||_HS^2 = (tr(C)^2 + ||C||_HS^2) / df, so the ensemble size needed grows with the effective rank
    tr(C)^2 / ||C||_HS^2; with few samples, long-range correlations are spurious. The result has rank <= n and
    supports everything a LowRankGaussian does (conditioning: an ensemble Kalman analysis; exact pointwise variance).
    ``device``: where the result samples (default: the source's GPU if it has one, or that of GPU samples; else the
    CPU).
    """
    from .lowrank import LowRankGaussian
    from .seeding import inherit

    if device is None:
        if hasattr(source, "sample") and hasattr(source, "space"):
            device = compute_device("auto", source) or "cpu"
        else:
            device = source.device if isinstance(source, torch.Tensor) and source.device.type != "cpu" else "cpu"
    if hasattr(source, "sample") and hasattr(source, "space"):
        space = source.space
        if n is None:
            raise ValueError("give the number of samples n")
        x = _np(source.sample(int(n)))  # ndof x n
        m = source.mean if mean is None else mean
    else:
        if fes is None:
            raise ValueError("samples need their space: pass fes=")
        space = SpaceInfo(fes)
        if isinstance(source, (list, tuple)):
            x = np.column_stack([_np(g.vec.FV().NumPy()) if hasattr(g, "vec") else _np(g) for g in source])
        else:
            x = _np(source).T
        m = "sample" if mean is None else mean
    k = x.shape[1]
    if isinstance(m, str):
        if m != "sample":
            raise ValueError("mean must be None, 'sample', or a mean function/vector")
        m, df = x.mean(axis=1), k - 1
    else:
        m, df = coefficient_vector(m, space), k
    if df < 1:
        raise ValueError("need at least two samples with the sample mean")
    b = (x - m[:, None]) / np.sqrt(df)
    out = LowRankGaussian(space.fes, b, np.eye(k), mean=m, seed=inherit(seed, source), device=device)  # seeding
    out.df = df
    out.plan["structure"] = f"sample covariance of {k} samples (Wishart, {df} degrees of freedom)"
    return out
