"""Finite-rank Gaussians N(m, B K^-1 B*) on FE spaces, and truncated Karhunen-Loeve expansions of any Gaussian."""

import warnings

import ngsolve as ngs
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import torch

from .backend import available_memory, numpy_blas_threads
from .inplace import InPlace
from .observations import observation_operator, whiten, whitener
from .randomized import gram_orth, randomized_svd
from .seeding import as_source, inherit
from .space import SpaceInfo, coefficient_columns, coefficient_vector


class LowRankGaussian(InPlace):
    """Finite-rank Gaussian N(mean, B K^-1 B*) on a FE space (B: ndof x r coefficient columns, K: r x r SPD).

    Samples are B K^-1/2 zeta with zeta ~ N(0, I_r). The function-space structure is carried by the columns of B, so
    no white noise in the FE space is needed. If ``d`` is given, it is checked to be the L2 adjoint B^T M or the
    Euclidean transpose B^T, so the operator and matrix conventions cannot be mixed up silently. Columns of B must
    vanish at Dirichlet dofs (those are fixed to the mean).
    """

    def __init__(self, fes, b, k, mean=None, d=None, seed=None):  # seed: see ngsrng.seeding
        self.space = SpaceInfo(fes)
        self.b = coefficient_columns(b, fes.ndof)
        free = self.space.free
        self._bf = self.b if len(free) == fes.ndof else self.b[free]  # free rows, copied once rather than per apply
        self.k = np.asarray(k, dtype=float)
        self.rinv = sla.solve_triangular(np.linalg.cholesky(self.k).T, np.eye(self.k.shape[0]))  # K^-1 = rinv rinv^T
        if d is not None:
            d = np.asarray(d.toarray() if sp.issparse(d) else d, dtype=float)
            ops = {"L2 adjoint B^T M": (self.space.mass @ self.b).T, "transpose B^T": self.b.T}
            if not any(np.allclose(d, ref, rtol=1e-8, atol=1e-12 * np.abs(ref).max()) for ref in ops.values()):
                raise ValueError("D is neither the L2 adjoint B^T M nor the transpose B^T of B")
        self._mean = np.zeros(fes.ndof) if mean is None else coefficient_vector(mean, self.space)
        self.source = as_source(seed)
        self.device = torch.device("cpu")
        self.plan = {"space": self.space.describe(), "structure": f"finite rank {self.b.shape[1]}: B K^-1/2 zeta"}
        self._prior = None  # set on conditioned low-rank Gaussians and on KL expansions of posteriors

    @property
    def nrand(self) -> int:
        return self.b.shape[1]

    @property
    def mean(self):
        return self._mean.copy()

    def apply(self, eta: torch.Tensor) -> torch.Tensor:
        """eta (r x k) -> zero-mean free-dof coefficients (nfree x k): B K^-1/2 eta."""
        e = eta.detach().cpu().double().numpy()
        return torch.from_numpy(self._bf @ (self.rinv @ e))

    def apply_t(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint of ``apply``: (nfree x k) -> (r x k)."""
        v = y.detach().cpu().double().numpy()
        return torch.from_numpy(self.rinv.T @ (self._bf.T @ v))

    def sample(self, n=None, out=None):
        """One sample (GridFunction) or n samples (ndof x n). ``out``: an ndof x n buffer filled in place and returned
        (:mod:`ngsrng.inplace`)."""
        if out is not None:
            return self._sample_out(n, out)
        ns = 1 if n is None else int(n)
        zeta = self.source.streams().numpy().standard_normal((ns, self.nrand)).T  # sample by sample: chunking-free
        x = self._mean[:, None] + self.b @ (self.rinv @ zeta)
        if n is None:
            gf = ngs.GridFunction(self.space.fes)
            gf.vec.FV().NumPy()[:] = x[:, 0]
            return gf
        return torch.from_numpy(x)

    def condition(self, operator=None, *, noise=1.0, data=None, seed=None, **observations):
        """Exact posterior under y = G f + e, e ~ N(0, Gamma): again finite-rank, with r x r algebra only.

        With F = Gamma^-1/2 G B, the posterior is N(m + B K_post^-1 F^T Gamma^-1/2 (y - G m), B K_post^-1 B*) with
        K_post = K + F^T F. Observations as in :func:`ngsrng.observation_operator`.
        """
        g = observation_operator(self.space.fes, operator, **observations)
        kind, data_w = whitener(noise, g.shape[0])
        f = whiten(kind, data_w, np.asarray(g @ self.b))
        k_post = self.k + f.T @ f
        mean = self.mean
        if data is not None:
            resid = whiten(kind, data_w, np.asarray(data, dtype=float) - g @ mean)
            mean = mean + self.b @ np.linalg.solve(k_post, f.T @ resid)
        post = LowRankGaussian(self.space.fes, self.b, k_post, mean=mean, seed=inherit(seed, self))
        post._prior = self if self._prior is None else self._prior
        return post


def karhunen_loeve(gauss, rank=None, oversampling=None, power_iterations=None, seed=None,
                   method="auto", sketch_seed=0) -> LowRankGaussian:
    """Truncated Karhunen-Loeve expansion: the best rank-``rank`` L2 approximation of a Gaussian's covariance.

    The modes are eigenfunctions of the covariance operator C (L2-self-adjoint), i.e. of C_h = Sigma M with
    M-orthonormal coefficient vectors; the eigenvalues are the squared singular values of the factor L
    (Sigma = L L^T) as a map from white noise to (V_h, L2). Works for any Gaussian with ``apply``/``apply_t``
    (priors, posteriors, any engine). ``method``:

    - "subspace" (the default for "auto"): randomised subspace iteration on the factor L (Aarset 2024, Algorithm 2;
      Halko, Martinsson & Tropp 2011, Algs. 4.4 & 5.1, see :mod:`ngsrng.randomized`): Q_0 = M-orth(L Omega), then
      ``power_iterations`` times Qt = qr(L^T M Q), Q = M-orth(L Qt), and an SVD of Q^T M L. Re-orthonormalising after
      every application and taking the SVD directly keep small eigenvalues accurate (also in float32 pipelines).
      Cost: (2q + 2) block applications of L or L^T; default q = 4;
    - "covariance": randomised range finder on C_h = L L^T M with ``power_iterations`` passes and Rayleigh-Ritz
      via the eigendecomposition of Z^T Z, Z = L^T M Q. Cost: (2q + 3) block applications; default q = 3. It starts
      from C_h Omega (half a power iteration more than L Omega), so at equal q it converges slightly further, but it
      squares the condition number in the final eigendecomposition and loses small eigenvalues first.

    The result has ``eigenvalues`` (descending) and ``modes`` (ndof x rank, L2-orthonormal) attributes. The default
    oversampling is max(10, rank / 2); slowly decaying spectra need more oversampling or power iterations.

    For a finite-rank Gaussian (:class:`LowRankGaussian`: KL expansions, sample covariances, operator covariances,
    their posteriors) the expansion is computed exactly from its r x r Gram matrix, and ``rank=None`` keeps all modes.

    ``seed``: the randomness of the returned sampler (default: inherited from ``gauss``, see :mod:`ngsrng.seeding`).
    ``sketch_seed``: the fixed seed of the randomised range finder, so that the modes are deterministic.
    """
    if isinstance(gauss, LowRankGaussian):
        return _exact_kl(gauss, rank, seed)
    if rank is None:
        raise ValueError("karhunen_loeve needs a rank (it is exact, without truncation, only for finite-rank "
                         "Gaussians)")
    method = "subspace" if method == "auto" else method
    if method not in ("subspace", "covariance"):
        raise ValueError("method must be 'auto', 'subspace' or 'covariance'")
    if power_iterations is None:
        power_iterations = 4 if method == "subspace" else 3
    s = gauss.space
    mass = s.mass_free
    oversampling = max(10, rank // 2) if oversampling is None else oversampling
    orth = gram_orth(mass)

    def apply_l(x):
        return gauss.apply(torch.from_numpy(np.ascontiguousarray(x))).double().cpu().numpy()

    def apply_lt(x):
        return gauss.apply_t(torch.from_numpy(np.ascontiguousarray(x))).double().cpu().numpy()

    # NumPy's OpenBLAS threads busy-wait after the dense QR / SVD steps and slow the NGSolve applications in
    # between: limit them as the Gaussian's own pipeline does
    with numpy_blas_threads(getattr(gauss, "_cpu_blas_threads", 1) if getattr(gauss, "engine", "") != "torch"
                            else None):
        if method == "subspace":
            nrand = gauss.nrand
            u, sv, _ = randomized_svd(apply_l, lambda y: apply_lt(mass @ y), nrand, min(rank, s.nfree, nrand),
                                      power_iterations=power_iterations, oversampling=oversampling, seed=sketch_seed,
                                      orth=orth, rtol=1e-7)
            lam, vecs = sv**2, u
        else:
            ell = min(rank + oversampling, s.nfree)
            y = apply_l(apply_lt(mass @ np.random.default_rng(sketch_seed).standard_normal((s.nfree, ell))))
            for _ in range(power_iterations):
                y = apply_l(apply_lt(mass @ orth(y)))
            q = orth(y)
            z = apply_lt(mass @ q)
            lam, v = np.linalg.eigh(z.T @ z)
            order = np.argsort(lam)[::-1][:rank]
            lam = np.clip(lam[order], 0.0, None)
            keep = lam > lam[0] * 1e-14
            lam, vecs = lam[keep], q @ v[:, order][:, keep]
    if len(lam) < rank:
        dims = min(s.nfree, gauss.nrand)
        why = (f"the Gaussian has only {dims} dimensions" if dims < rank else
               "the other eigenvalues are numerically zero, below 1e-14 of the largest")
        warnings.warn(f"karhunen_loeve: rank {len(lam)} instead of {rank} ({why})", stacklevel=2)
    modes = np.zeros((s.ndof, len(lam)))
    modes[s.free] = vecs
    out = LowRankGaussian(s.fes, modes, np.diag(1.0 / lam), mean=gauss.mean, seed=inherit(seed, gauss))
    out.eigenvalues, out.modes = lam, modes
    out._prior = getattr(gauss, "_prior", None)  # a KL expansion of a posterior is still referred to its prior
    out.plan["structure"] = f"Karhunen-Loeve truncation ({method}), rank {out.nrand}"
    return out


def _exact_kl(g, rank=None, seed=None) -> LowRankGaussian:
    """Exact KL expansion of Sigma = W W^T (W = B K^-1/2): with W^T M W = V Lambda V^T, the M-orthonormal modes are
    W V Lambda^-1/2 and the eigenvalues Lambda."""
    s = g.space
    w = g.b[s.free] @ g.rinv
    gram = w.T @ (s.mass_free @ w)
    lam, v = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(lam)[::-1]
    lam, v = lam[order], v[:, order]
    keep = lam > max(lam[0], 0.0) * 1e-13
    if rank is not None:
        keep &= np.arange(len(lam)) < rank
    lam, v = lam[keep], v[:, keep]
    modes = np.zeros((s.ndof, len(lam)))
    modes[s.free] = (w @ v) / np.sqrt(lam)
    out = LowRankGaussian(s.fes, modes, np.diag(1.0 / lam), mean=g.mean, seed=inherit(seed, g))
    out.eigenvalues, out.modes = lam, modes
    out._prior = getattr(g, "_prior", None)
    out.plan["structure"] = f"Karhunen-Loeve expansion (exact), rank {len(lam)}"
    return out


def operator_kl(fes, cov, mean=None, seed=None, sketch_seed=0) -> LowRankGaussian:
    """Adaptive truncated KL expansion of an apply-only covariance (:class:`ngsrng.specs.OperatorCovariance`).

    Eigenpairs of C_h = Sigma M (M-orthonormal modes) by a randomised range finder with power iterations and
    Rayleigh-Ritz on Q^T M Sigma M Q. The rank doubles until the uncaptured trace tr(Sigma M) - sum(lambda), estimated
    by Hutchinson with white-noise probes z ~ N(0, M) on the deflated covariance, is at most ``cov.tol`` times the
    total. The result records ``eigenvalues``, ``modes`` and ``trace_residual`` (the estimated relative error in
    trace norm). ``seed``: the randomness of the returned sampler (:mod:`ngsrng.seeding`); ``sketch_seed``: the fixed
    seed of the range finder and the trace probes, so that the modes are deterministic.
    """
    from .noise import DiagonalNoise, LocalGramNoise

    s = SpaceInfo(fes)
    mass = s.mass_free
    rng = np.random.default_rng(sketch_seed)
    apply_sigma = cov.sigma(s)

    def apply_a(x):  # C_h = Sigma M
        return apply_sigma(mass @ x)

    def m_orth(y):
        for _ in range(2):
            r = np.linalg.cholesky(y.T @ (mass @ y) + 1e-300 * np.eye(y.shape[1]))
            y = sla.solve_triangular(r, y.T, lower=True).T
        return y

    noise = DiagonalNoise(mass) if s.mass_is_diagonal else LocalGramNoise(s).explicit()
    noise.setup(torch.device("cpu"), torch.float64)
    z = noise.apply(torch.from_numpy(rng.standard_normal((noise.nrand, cov.probes)))).numpy()
    sz = apply_sigma(z)
    total = float(np.mean(np.sum(z * sz, axis=0)))  # Hutchinson: E z^T Sigma z = tr(Sigma M)

    rank = cov.rank or min(32, cov.max_rank, s.nfree)
    while True:
        ell = min(rank + max(10, rank // 2), s.nfree)
        y = apply_a(rng.standard_normal((s.nfree, ell)))
        for _ in range(cov.power_iterations):
            y = apply_a(m_orth(y))
        q = m_orth(y)
        mq = mass @ q
        t = mq.T @ apply_sigma(mq)
        lam, v = np.linalg.eigh(0.5 * (t + t.T))
        order = np.argsort(lam)[::-1][:rank]
        lam, u = np.clip(lam[order], 0.0, None), q @ v[:, order]
        proj = u.T @ z  # Sigma_r = U Lambda U^T, so z^T Sigma_r z = sum_i lambda_i (u_i^T z)^2
        resid = float(np.mean(np.sum(z * sz, axis=0) - np.sum(lam[:, None] * proj**2, axis=0)))
        rel = max(resid, 0.0) / max(total, 1e-300)
        if cov.rank is not None or rel <= cov.tol:
            break
        if rank >= min(cov.max_rank, s.nfree):
            limit = f"max_rank={cov.max_rank}"
            break
        nxt = min(2 * rank, cov.max_rank, s.nfree)
        avail = available_memory()  # the sketch blocks are dense nfree x ell (about six of them alive at a time)
        if avail is not None and 48.0 * s.nfree * min(nxt + max(10, nxt // 2), s.nfree) > 0.5 * avail:
            limit = "memory"
            break
        rank = nxt
    if cov.rank is None and rel > cov.tol:
        warnings.warn(f"operator KL stopped at rank {rank} ({limit}): uncaptured trace ~{rel:.1e} > tol {cov.tol:g}; "
                      "the field misses that fraction of its variance (rough covariances need high ranks)",
                      stacklevel=3)
    keep = lam > lam[0] * 1e-14
    modes = np.zeros((s.ndof, int(keep.sum())))
    modes[s.free] = u[:, keep]
    out = LowRankGaussian(fes, modes, np.diag(1.0 / lam[keep]), mean=mean, seed=seed)
    out.eigenvalues, out.modes, out.trace_residual = lam[keep], modes, rel
    out.plan["structure"] = f"apply-only covariance: adaptive KL, rank {out.nrand}, uncaptured trace ~{rel:.1e}"
    return out
