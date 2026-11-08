"""Conditioned Gaussians: linear observations, known low-rank downdates, and pathwise conditioning.

Linear Bayesian inverse problem: y = G f + e, e ~ N(0, Gamma), prior f ~ N(m0, C0), with G any finite set of bounded
linear functionals (``ngsrng.observations``). The posterior covariance is C_post = (G* Gamma^-1 G + C0^-1)^-1. For
finite-dimensional data the posterior is equivalent to the prior (Feldman-Hajek), so it is never finite-rank: the
low-rank object is the *update*, C_post = C0 - B K^-1 B* (Woodbury: B = C0 G*, K = Gamma + G C0 G*).

1. ``LinearObservations``: let Sigma0 = L L^T be the factorisation the prior sampler implements (``Gaussian.apply``
   = L, any factor, rectangular allowed, e.g. L = K^-1 H with local mass factors H). Woodbury gives
   Sigma_post = L (I + F^T F)^-1 L^T with F = Gamma^-1/2 G L. With the thin QR F^T = Q R (Q: nrand x l orthonormal,
   l <= m),

       (I + F^T F)^-1/2 = I + Q ((I + R R^T)^-1/2 - I) Q^T,

   so a posterior sample is the prior pipeline applied to modified white noise (``form="noise"``), or equivalently
   the prior sample plus an FE-space low-rank update (``form="fe"``, the default and the cheaper one). Both are exact.
   Setup costs m applications of L^T (plus ``rank`` applications of L for the FE-space form and the mean).

2. ``downdate``: a known C_post = C0 - B K^-1 B*, given directly. With Z = L^T Sigma0^-1 B,
   Sigma_post = L (I - Z K^-1 Z^T) L^T. The middle matrix is PSD exactly when C_post is (Schur complement), and its
   square root is again I + Q A Q^T.

3. ``MatheronPosterior``: pathwise conditioning, an independent exact sampler for (1), for cross-checks.

Finite-rank Gaussians N(m, B K^-1 B*) (truncated Karhunen-Loeve, reduced bases) live in ``ngsrng.lowrank``.
"""

import ngsolve as ngs
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import torch

from .backend import Scratch, available_memory, numpy_blas_threads
from .inplace import InPlace
from .observations import CHUNK, dense_if_full, observation_operator, whiten, whitener
from .randomized import randomized_qr
from .seeding import inherit
from .space import coefficient_columns, coefficient_vector

# ------------------------------------------------------------------ low-rank update maps


def check_memory(prior, columns, rows_per_column, what, hint):
    """Raise MemoryError before a set-up allocates ``columns`` dense float64 columns of ``rows_per_column`` entries in
    total (e.g. nrand + 2 nfree per observation for an exact posterior) beyond the available host memory, and, for
    a prior on a CUDA device, its F^T and Q (2 nrand per column) beyond the free device memory."""
    need = 8.0 * columns * rows_per_column
    avail = available_memory()
    if avail is not None and need > avail:
        raise MemoryError(f"{what} needs ~{need / 2**30:.1f} GiB of dense low-rank factors ({columns} columns), "
                          f"{avail / 2**30:.1f} GiB available. {hint}")
    device = getattr(prior, "device", None)
    if getattr(prior, "engine", None) == "torch" and device is not None and device.type == "cuda":
        dev_need, dev_free = 16.0 * columns * prior.nrand, torch.cuda.mem_get_info(device)[0]
        if dev_need > dev_free:
            raise MemoryError(f"{what} needs ~{dev_need / 2**30:.1f} GiB on {device}, {dev_free / 2**30:.1f} GiB "
                              f"free. {hint}")


class NoiseSpaceMap:
    """Symmetric map eta -> eta + Q A Q^T eta on the prior's white-noise space."""

    def __init__(self, q: np.ndarray, a: np.ndarray, shared=None):
        self.q = np.ascontiguousarray(q, dtype=np.float64)
        self.a = np.ascontiguousarray(a, dtype=np.float64)
        self._torch = {}
        # NGSolve / torch copies of Q, reusable by every map with the same Q (re-weighted posteriors)
        self._shared = {} if shared is None else shared

    @property
    def rank(self) -> int:
        return self.q.shape[1]

    def apply_numpy(self, e: np.ndarray):
        e += self.q @ (self.a @ (self.q.T @ e))

    def apply_ngs(self, eta):
        """In place on an NGSolve BaseVector, with NGSolve's multithreaded kernels.

        NumPy's GEMV would be equally fast, but its OpenBLAS threads slow down the NGSolve solve that follows.
        """
        if getattr(self, "_mv", None) is None:
            if "ngs" not in self._shared:
                mv = ngs.MultiVector(eta, self.rank)
                for k in range(self.rank):
                    mv[k].FV().NumPy()[:] = self.q[:, k]
                self._shared["ngs"] = mv
            self._mv = self._shared["ngs"]
        coeff = self.a @ np.asarray(self._mv.InnerProduct(eta))
        eta.data += self._mv * ngs.Vector(list(coeff))

    def apply_torch(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        if key not in self._torch:
            if key not in self._shared:
                self._shared[key] = torch.from_numpy(self.q).to(*key)
            self._torch[key] = (self._shared[key], torch.from_numpy(self.a).to(*key))
        q, a = self._torch[key]
        return x + q @ (a @ (q.T @ x))


def _qr(ft: torch.Tensor):
    """Householder QR of F^T (nrand x m, column-major): torch's (MKL on the CPU, twice as fast as SciPy's OpenBLAS
    here; F^T and Q are both alive) while two such blocks fit in half the available memory, else SciPy's in place (Q
    overwrites F^T: one block). Returns NumPy (Q, R)."""
    if ft.device.type == "cpu":
        avail = available_memory()
        if avail is not None and 2 * ft.numel() * ft.element_size() > 0.5 * avail:
            return sla.qr(ft.numpy(), mode="economic", overwrite_a=True, check_finite=False)
    q, r = torch.linalg.qr(ft)
    return q.detach().cpu().double().numpy(), r.detach().cpu().double().numpy()


def _rotate_in_place(q: np.ndarray, u: np.ndarray, rows: int = 4096) -> np.ndarray:
    """Q U (n x k, k <= columns of Q) written into Q's first k columns, ``rows`` rows at a time: no second n x k
    array. Returns that view."""
    k = u.shape[1]
    for i in range(0, q.shape[0], rows):
        q[i : i + rows, :k] = q[i : i + rows] @ u
    return q[:, :k]


class FESpaceUpdate:
    """c -> c + P S (G c) applied to the output of the prior pipeline (free-dof coefficient vectors).

    P: (nfree x r) dense, S: (r x m) small dense, G: (m x nfree) observation rows, sparse or dense. For the
    noise-space update eta -> eta + Q A Q^T eta, using Q^T eta = s^-1 V^T F eta and F eta = Gamma^-1/2 G (L eta), the
    same posterior is c -> c + (L Q) A s^-1 V^T Gamma^-1/2 G c. A sample then needs a G product and one pass over L Q
    (nfree x r) instead of two passes over Q (nrand x r). A s^-1 stays bounded as s -> 0, so this is stable.
    """

    def __init__(self, p: np.ndarray, smat: np.ndarray, g: sp.csr_matrix, free: np.ndarray, ndof: int, shared=None):
        self.p = np.ascontiguousarray(p, dtype=np.float64)
        self.s = np.ascontiguousarray(smat, dtype=np.float64)
        self.g = g.tocsr() if sp.issparse(g) else np.ascontiguousarray(g, dtype=np.float64)
        self.free, self.ndof = free, ndof
        self._torch = {}
        self._ngs = None
        # NGSolve / torch copies of P and G, reusable by every update with the same P and G (re-weighted posteriors
        # differ only in S); converting G to NGSolve's format dominates the set-up otherwise
        self._shared = {} if shared is None else shared

    @property
    def rank(self) -> int:
        return self.p.shape[1]

    def _tensors(self, device, dtype):
        key = (device, dtype)
        if key not in self._torch:
            from .backend import sparse_to_torch

            if key not in self._shared:
                if sp.issparse(self.g):
                    g, gt = sparse_to_torch(self.g, device, dtype), sparse_to_torch(self.g.T.tocsr(), device, dtype)
                else:
                    g = torch.from_numpy(self.g).to(device, dtype)
                    gt = g.T
                self._shared[key] = (torch.from_numpy(self.p).to(device, dtype), g, gt)
            p, g, gt = self._shared[key]
            self._torch[key] = (p, torch.from_numpy(self.s).to(device, dtype), g, gt)
        return self._torch[key]

    def apply_torch(self, c: torch.Tensor) -> torch.Tensor:
        p, smat, g, _ = self._tensors(c.device, c.dtype)
        return c + p @ (smat @ (g @ c))

    def apply_t_torch(self, y: torch.Tensor) -> torch.Tensor:
        p, smat, _, gt = self._tensors(y.device, y.dtype)
        return y + gt @ (smat.T @ (p.T @ y))

    def apply_ngs(self, vec):
        """In place on a full-dof NGSolve vector (zeros at Dirichlet dofs stay zero)."""
        if self._ngs is None:
            from .backend import ngs_sparse

            if "ngs" not in self._shared:
                # dense rows: NumPy (one GEMV per sample) instead of an NGSolve sparse matrix of full rows
                gfull = ngs_sparse(self.g, cols=self.free, ncols=self.ndof) if sp.issparse(self.g) else None
                pmv = ngs.MultiVector(vec, self.rank)
                for k in range(self.rank):
                    col = pmv[k].FV().NumPy()
                    col[:] = 0.0
                    col[self.free] = self.p[:, k]
                self._shared["ngs"] = (gfull, pmv)
            gfull, pmv = self._shared["ngs"]
            self._ngs = (gfull, pmv, Scratch(gfull.CreateColVector) if gfull is not None else None)  # per thread
        gfull, pmv, scratch = self._ngs
        if gfull is None:
            obs = self.g @ vec.FV().NumPy()[self.free]
        else:
            y = scratch.get()
            y.data = gfull * vec
            obs = y.FV().NumPy()
        coeff = self.s @ obs
        vec.data += pmv * ngs.Vector(list(coeff))


# ------------------------------------------------------------------ (1) observations of a prior


class LinearObservations:
    """Linear observations y = G f + e, e ~ N(0, Gamma), of a Gaussian ``prior``.

    Parameters
    ----------
    prior : ngsrng.Gaussian (any precision type, engine and device)
    operator, functionals, averages, points, adjoint, m : the observation operator G, any finite set of bounded linear
        functionals (see :func:`observation_operator`)
    noise : variance (scalar), variances (m,), or covariance matrix (m, m)
    rank : None for the exact construction (m applications of L^T; memory nrand x m). An integer k computes a
        randomised rank-k QR of the prior-preconditioned adjoint F^T = L^T G^T Gamma^-1/2 with k + ``oversampling``
        columns and ``power_iterations`` subspace iterations (Aarset 2024, Algorithm 2; see :mod:`ngsrng.randomized`):
        approximate, but only (2q + 2)(k + oversampling) applications of L or L^T. ``low_rank_method="power"`` uses
        plain power iteration instead (for comparison).
    factorize : False builds only the observation operator (for :class:`MatheronPosterior`).

    The factorisation (public, also for code built on it, e.g. optimal experimental design): with the whitened,
    prior-preconditioned forward map F = Gamma^-1/2 G L (L: the prior's factor, C0 = L L^T on the free dofs),
    F^T = Q R with orthonormal Q (nrand x rank). ``R`` (rank x m) = diag(s) V^T in the SVD basis of R,
    ``singular_values`` s (descending), ``lq()`` = L Q (free-dof coefficients, nfree x rank), ``exact`` (False for
    the randomised factorisation, where F^T ~ Q R), ``G`` and ``G_free`` the observation operator on all / the free
    dofs (sparse, or dense for full rows), ``noise_std`` and ``correlated_noise`` the noise model. Every posterior
    follows from them: Sigma_post = C0 - (L Q) (I - (I + R S R^T)^-1) (L Q)^T for noise precisions scaled by S
    (``posterior(noise_scaling=...)``).
    """

    def __init__(
        self,
        prior,
        operator=None,
        *,
        functionals=None,
        points=None,
        averages=None,
        adjoint=None,
        m=None,
        noise=1.0,
        rank=None,
        oversampling=10,
        power_iterations=1,
        seed=0,
        factorize=True,
        low_rank_method="subspace",
    ):
        self.prior = prior
        s = prior.space
        # G (m x ndof): sparse, or dense when its rows are full (a dense matrix, e.g. observations of a PDE solution)
        self.G = dense_if_full(observation_operator(s.fes, operator, functionals, points, averages, adjoint, m))
        self.m = self.G.shape[0]
        self.wkind, self.wdata = whitener(noise, self.m)
        self.noise = noise
        # observation of the free part (no copy when every dof is free)
        self.gf = self.G if len(s.free) == s.ndof else self.G[:, s.free]
        # Gamma^-1/2 G_free (m x nfree), set-up only: sparse stays sparse for diagonal noise
        if self.wkind == "diag":
            gh = sp.diags(self.wdata) @ self.gf if sp.issparse(self.gf) else self.wdata[:, None] * self.gf
        else:
            gh = self.wdata @ self.gf
        self._lq = None
        if not factorize:  # observation operator only (e.g. for MatheronPosterior)
            return
        # dense blocks of the set-up (columns: observations, or the sketch width): F^T and its Q factor (nrand; Q
        # overwrites F^T when memory is short, see _qr; Q U overwrites Q; the randomised sketch keeps three such
        # blocks), L Q and its NGSolve/torch copy for the FE-space form (nfree each)
        width = self.m if rank is None else min(self.m, int(rank) + int(oversampling))
        check_memory(prior, width, (1 if rank is None else 3) * prior.nrand + 2 * s.nfree,
                     "the exact posterior" if rank is None else f"the rank-{rank} posterior",
                     "Use rank=k (randomised low rank: (2q+2)(k+oversampling) applications of L, error set by the "
                     "decay of the data-informed spectrum), or fewer observations." if rank is None else
                     "Use a smaller rank.")
        if rank is None:
            ft = self._lt(gh.T)  # F^T = L^T G^T Gamma^-T/2 (nrand x m), column-major
            del gh
            q, r = _qr(ft)
            del ft
            self.exact = True
        else:
            blas = getattr(prior, "_cpu_blas_threads", 1) if getattr(prior, "engine", "") != "torch" else None
            with numpy_blas_threads(blas):  # see ngsrng.lowrank.karhunen_loeve
                q, r = randomized_qr(
                    lambda x: self._lt(gh.T @ x).cpu().numpy(),  # F^T: whitened observations -> noise
                    lambda y: gh @ self._l(np.ascontiguousarray(y)),  # F: noise -> whitened observations
                    self.m, int(rank), power_iterations=power_iterations, oversampling=oversampling, seed=seed,
                    method=low_rank_method,
                )
            q, r = torch.from_numpy(q), torch.from_numpy(r)
            self.exact = False
        if isinstance(q, torch.Tensor):
            q, r = q.detach().cpu().double().numpy(), r.detach().cpu().double().numpy()
        # Rotate to the SVD basis of R (F^T = Q R = (Q U) s V^T) and drop exactly degenerate directions: then
        # q = Q U has orthonormal columns, r = s V^T, and the FE-space form needs s^-1 only where s > 0.
        u, sv, vt = np.linalg.svd(r, full_matrices=False)
        keep = sv > (np.finfo(float).eps * max(r.shape) * (sv.max() if sv.size else 0.0))
        self.q, self.sv, self.vt = _rotate_in_place(q, u[:, keep]), sv[keep], vt[keep]
        self.r = self.sv[:, None] * self.vt

    def _get_lq(self):
        """L Q (nfree x rank): needed for the FE-space form and the posterior mean."""
        if self._lq is None:
            self._lq = self._l(self.q)
        return self._lq

    def lq(self) -> np.ndarray:
        """L Q (nfree x rank): the factorisation's noise-space basis Q mapped to free-dof coefficients (computed once,
        rank applications of L). See the class docstring."""
        return self._get_lq()

    @property
    def R(self) -> np.ndarray:  # noqa: N802 - the matrix of F^T = Q R
        """R (rank x m) of F^T = Q R, in the SVD basis: diag(singular_values) V^T."""
        return self.r

    @property
    def singular_values(self) -> np.ndarray:
        """Singular values of F = Gamma^-1/2 G L (descending; the data-informed spectrum)."""
        return self.sv

    @property
    def G_free(self):  # noqa: N802 - the matrix G
        """The observation operator on the free dofs (m x nfree; sparse, or dense for full rows)."""
        return self.gf

    @property
    def correlated_noise(self) -> bool:
        """Whether the noise covariance is a full matrix (otherwise a variance per observation)."""
        return self.wkind != "diag"

    @property
    def noise_std(self):
        """Noise standard deviation of every observation (m,), or None for correlated noise."""
        return None if self.correlated_noise else 1.0 / self.wdata

    def data_variance(self, chunk: int = CHUNK) -> np.ndarray:
        """Prior variance of every noise-free observation, Var((G f)_i) = (G C0 G^T)_ii, exactly (m,).

        From the factorisation when it is exact (F F^T = R^T R = Gamma^-1/2 G C0 G^T Gamma^-T/2: free), otherwise
        one application of L^T per observation, ``chunk`` at a time. Add the noise variance for the variance of the
        data."""
        if getattr(self, "exact", False):
            if not self.correlated_noise:
                return np.sum(self.r**2, axis=0) / self.wdata**2
            chol = np.linalg.inv(self.wdata)  # Gamma = C C^T, wdata = C^-1
            return np.sum((self.r @ chol.T) ** 2, axis=0)
        gt = self.gf.T.tocsc() if sp.issparse(self.gf) else self.gf.T
        out = np.empty(self.m)
        for i in range(0, self.m, chunk):
            blk = gt[:, i : i + chunk]
            blk = blk.toarray() if sp.issparse(blk) else np.ascontiguousarray(blk)
            out[i : i + chunk] = (self.prior.apply_t(torch.from_numpy(blk)).double() ** 2).sum(0).cpu().numpy()
        return out

    @property
    def rank(self) -> int:
        return self.q.shape[1]

    def predictive(self, source=None, noise=True, **options):
        """The law of the data y = G f + e as an :class:`ngsrng.ObservedGaussian`: f from ``source`` (default the
        prior: the prior predictive, whose density at the data is the evidence; a posterior gives the posterior
        predictive), e from this noise model (``noise=False``: G f alone). ``options``: method, rtol, seed, device,
        dtype (see :class:`ngsrng.ObservedGaussian`)."""
        from .observed import ObservedGaussian

        return ObservedGaussian(self.prior if source is None else source, self.G,
                                noise=self.noise if noise else None, **options)

    def _lt(self, x) -> torch.Tensor:
        """L^T x (float64), CHUNK columns at a time into one preallocated result; x dense or sparse (densified one
        chunk at a time)."""
        x = x.tocsc() if sp.issparse(x) else x
        out = None
        for i in range(0, x.shape[1], CHUNK):
            blk = x[:, i : i + CHUNK]
            blk = blk.toarray() if sp.issparse(blk) else np.ascontiguousarray(blk)
            col = self.prior.apply_t(torch.from_numpy(blk))
            if out is None:  # column-major: each chunk is a contiguous write, and LAPACK's QR can work in place
                out = torch.empty((x.shape[1], col.shape[0]), dtype=torch.float64, device=col.device).T
            out[:, i : i + col.shape[1]] = col
        return out

    def _l(self, eta: np.ndarray) -> np.ndarray:
        cols = [
            self.prior.apply(torch.from_numpy(np.ascontiguousarray(eta[:, i : i + CHUNK]))).double().cpu().numpy()
            for i in range(0, eta.shape[1], CHUNK)
        ]
        return np.hstack(cols)

    def posterior(self, data=None, seed=None, form="fe", noise_scaling=None, mean=None):
        """The posterior as a ``Gaussian`` (same pipeline and engine as the prior, plus the low-rank update).

        ``data`` sets the posterior mean; without data the mean is the prior mean (enough for covariance studies).
        ``form``: "fe" applies the update to the prior sample in FE space (default: a sparse G product plus one pass
        over L Q per sample; setup adds ``rank`` applications of L). "noise" modifies the white noise before the
        pipeline (two passes over Q per sample; no extra setup). Both are exact and give identical samples.

        ``noise_scaling`` (one weight w_i >= 0 per observation) multiplies the noise precision of observation i by
        w_i, i.e. divides its variance by w_i (w_i = 0 ignores it). The factorisation does not depend on the weights,
        so a re-weighted posterior costs only an l x l eigendecomposition: the Gaussian step of iteratively
        reweighted least squares and of Gibbs samplers for scale-mixture noise (Student-t, Laplace).

        ``mean`` (instead of ``data``) sets the mean directly: a Gaussian with this posterior covariance centred
        elsewhere, e.g. the Laplace approximation of a nonlinear problem, centred at its MAP point, with the
        observation operator linearised there.
        """
        if mean is not None and data is not None:
            raise ValueError("give data or mean, not both")
        post = self._posterior_from(noise_scaling, data, seed, form)
        if mean is not None:
            post._mean = coefficient_vector(mean, self.prior.space)
        return post

    def _posterior_from(self, scaling, data, seed, form):
        """Posterior with the noise precision of observation i multiplied by ``scaling[i]`` >= 0 (None: unchanged).

        The factorisation does not depend on the scaling, so re-scaled posteriors cost only an l x l
        eigendecomposition (public as ``posterior(noise_scaling=...)``).
        """
        sc = self._scaling(scaling)
        # NumPy BLAS threads busy-wait against NGSolve's threads even on these small l x l problems (known issue 9)
        prior = self.prior
        with numpy_blas_threads(getattr(prior, "_cpu_blas_threads", 1) if getattr(prior, "engine", "") != "torch"
                                else None):
            middle = np.eye(self.rank) + (self.r * sc) @ self.r.T  # I + R S R^T
            lam, v = np.linalg.eigh(middle)
            a = (v * (lam**-0.5 - 1.0)) @ v.T
            mean = prior.mean
            if data is not None:
                resid = np.asarray(data, dtype=float) - self.G @ mean
                coeff = np.linalg.solve(middle, self.r @ (sc * whiten(self.wkind, self.wdata, resid)))
                mean[prior.space.free] += self._get_lq() @ coeff
            if form == "fe":
                smat = (a / self.sv[None, :]) @ self.vt
                # times Gamma^-1/2: a column scaling for uncorrelated noise (no m x m diagonal matrix)
                smat = smat * self.wdata[None, :] if self.wkind == "diag" else smat @ self.wdata
        kind = "exact" if self.exact else "randomised"
        shared = self.__dict__.setdefault("_shared_maps", {"noise": {}, "fe": {}})
        if form == "noise":
            label = f"{self.m} linear observations ({kind}, noise-space update, rank {self.rank})"
            return self.prior._derive(label, mean, eta_map=NoiseSpaceMap(self.q, a, shared["noise"]), seed=seed,
                                      exact=self._exact_record(scaling))
        if form != "fe":
            raise ValueError(f"unknown form {form!r}")
        update = FESpaceUpdate(self._get_lq(), smat, self.gf, self.prior.space.free, self.prior.space.ndof,
                               shared["fe"])
        label = f"{self.m} linear observations ({kind}, FE-space update, rank {self.rank})"
        return self.prior._derive(label, mean, post_map=update, seed=seed, exact=self._exact_record(scaling))

    def _scaling(self, scaling) -> np.ndarray:
        sc = np.ones(self.m) if scaling is None else np.asarray(scaling, dtype=float)
        if sc.shape != (self.m,) or (sc < 0).any():
            raise ValueError("noise_scaling must be a non-negative vector with one entry per observation")
        return sc

    def posterior_update(self, noise_scaling=None):
        """The posterior covariance as a low-rank update of the prior's, from the factorisation: (U, D) with

            Sigma_post = Sigma0 + U D U^T  (free dofs),  U = L Q (``lq()``, nfree x rank),  D = (I + R S R^T)^-1 - I,

        S = diag(noise_scaling) (the noise precision of observation i times its entry; default 1). Exact for an exact
        factorisation; for a randomised one it is the covariance of ``posterior(form="noise")``. No solves beyond
        ``lq()`` (once): an r x r inverse per scaling, so many scalings (reweighting, experimental designs) are cheap.
        Pass it to :meth:`ngsrng.LocalCovariance.plus`, or use :meth:`local_covariance`."""
        sc = self._scaling(noise_scaling)
        middle = np.eye(self.rank) + (self.r * sc) @ self.r.T
        d = np.linalg.inv(0.5 * (middle + middle.T)) - np.eye(self.rank)
        return self.lq(), 0.5 * (d + d.T)

    def local_covariance(self, noise_scaling=None, prior_method="exact", **prior_options):
        """The posterior covariance on element-local dof pairs (:class:`ngsrng.LocalCovariance`): its ``variance()``
        (pointwise variance map), ``trace()`` (L2 trace) and ``std()``, for any noise scaling.

        The prior's part comes from :func:`ngsrng.local_covariance` (``prior_method``, ``prior_options``; computed once
        and cached on the prior: "exact" costs n_free applications of the prior covariance, "lowrank" is the
        randomised alternative), the update from :meth:`posterior_update`: no solves per scaling."""
        from .covariance import local_covariance

        base = local_covariance(self.prior, prior_method, **prior_options)
        return base.plus([self.posterior_update(noise_scaling)], method=f"{base.method} + factorised update")

    def _exact_record(self, scaling):
        return "observe", self.gf, self.wkind, self.wdata, None if scaling is None else np.asarray(scaling, float)


# ------------------------------------------------------------------ (2) known downdate


def downdate(prior, b, k, mean=None, seed=None, tol=1e-8):
    """N(mean, C0 - B K^-1 B*) as a Gaussian sharing ``prior``'s pipeline.

    ``b``: (ndof x r) coefficient columns (or a list of GridFunctions) of the functions spanning the update; B* is
    their L2 adjoint, so in coefficient form Sigma_post = Sigma0 - B K^-1 B^T. ``k``: (r x r) SPD. Raises if C_post is
    not positive semidefinite.
    """
    s = prior.space
    bmat = coefficient_columns(b, s.ndof)
    if np.abs(np.delete(bmat, s.free, axis=0)).max(initial=0.0) > 1e-12 * max(np.abs(bmat).max(), 1e-300):
        raise ValueError("B has nonzero Dirichlet entries; the update must leave Dirichlet values fixed")
    k = np.asarray(k, dtype=float)
    prec_b = prior.apply_precision(torch.from_numpy(np.ascontiguousarray(bmat[s.free]))).numpy()
    z = (
        torch.cat(
            [
                prior.apply_t(torch.from_numpy(np.ascontiguousarray(prec_b[:, i : i + CHUNK])))
                for i in range(0, prec_b.shape[1], CHUNK)
            ],
            dim=1,
        )
        .double()
        .cpu()
        .numpy()
    )
    q, r = np.linalg.qr(z)
    middle = np.eye(q.shape[1]) - r @ np.linalg.solve(k, r.T)
    lam, v = np.linalg.eigh(0.5 * (middle + middle.T))
    if lam.min() < -tol * max(1.0, np.abs(lam).max()):
        raise ValueError(
            f"C0 - B K^-1 B* is not positive semidefinite (eigenvalue {lam.min():.3e} of the reduced form)"
        )
    a = (v * (np.sqrt(np.clip(lam, 0.0, None)) - 1.0)) @ v.T
    mean_vec = prior.mean if mean is None else coefficient_vector(mean, s)
    return prior._derive(f"low-rank downdate (rank {bmat.shape[1]})", mean_vec, eta_map=NoiseSpaceMap(q, a), seed=seed,
                         exact=("downdate", bmat[s.free], k))


# ------------------------------------------------------------------ (3) pathwise (Matheron) conditioning


class MatheronPosterior(InPlace):
    """Exact posterior sampler by pathwise conditioning, independent of the noise-space construction:

        f_post = f0 + B K_d^-1 (y + e - G f0),   f0 ~ prior,  e ~ N(0, Gamma),
        B = Sigma0 G^T,  K_d = Gamma + G Sigma0 G^T.

    Setup: m applications of L^T and of L. Per sample: one prior sample plus O(nfree m).
    """

    def __init__(self, obs: LinearObservations, data=None, seed=None):
        prior, s = obs.prior, obs.prior.space
        check_memory(prior, obs.m, prior.nrand + 2 * s.nfree, "Matheron conditioning",
                     "Use LinearObservations(..., rank=k).posterior() for many observations.")
        g = obs.gf  # (m x nfree), sparse or dense
        gamma = np.diag(1.0 / obs.wdata**2) if obs.wkind == "diag" else np.asarray(obs.noise, dtype=float)
        lt = obs._lt(g.T)
        self.bmat = obs._l(lt.cpu().numpy())  # Sigma0 G^T (nfree x m_a)
        kd = gamma + np.asarray(g @ self.bmat)
        self.kd_chol = sla.cho_factor(kd)
        self.gamma_half = np.linalg.cholesky(gamma)
        self.g, self.prior, self.ma = g, prior, obs.m
        self._prior = prior if getattr(prior, "_prior", None) is None else prior._prior
        self._mean = prior.mean
        if data is not None:
            resid = np.asarray(data, dtype=float) - obs.G @ self._mean
            self._mean[s.free] += self.bmat @ sla.cho_solve(self.kd_chol, resid)
        self.source = inherit(seed, prior)  # seed=None: a child of the prior's randomness (ngsrng.seeding)
        self.space, self.device = s, torch.device("cpu")
        self.plan = {**prior.plan, "conditioning": [f"Matheron, {self.ma} observations"]}

    @property
    def nrand(self) -> int:
        return self.prior.nrand + self.ma

    @property
    def mean(self):
        return self._mean.copy()

    def apply(self, eta: torch.Tensor) -> torch.Tensor:
        e = eta.detach().cpu().double()
        e = e[:, None] if e.ndim == 1 else e
        f0 = self.prior.apply(e[: self.prior.nrand]).double().cpu().numpy()
        noise = self.gamma_half @ e[self.prior.nrand :].numpy()
        corr = self.bmat @ sla.cho_solve(self.kd_chol, noise - np.asarray(self.g @ f0))
        out = torch.from_numpy(f0 + corr)
        return out[:, 0] if eta.ndim == 1 else out

    def apply_t(self, y: torch.Tensor) -> torch.Tensor:
        """Adjoint of ``apply``: y -> [L0^T (y - G^T K_d^-1 B^T y); Gamma^T/2 K_d^-1 B^T y]."""
        v = y.detach().cpu().double()
        v = v[:, None] if v.ndim == 1 else v
        w = sla.cho_solve(self.kd_chol, self.bmat.T @ v.numpy())
        top = self.prior.apply_t(torch.from_numpy(np.ascontiguousarray(v.numpy() - np.asarray(self.g.T @ w))))
        top = top.double().cpu()
        out = torch.cat([top, torch.from_numpy(self.gamma_half.T @ w)], dim=0)
        return out[:, 0] if y.ndim == 1 else out

    def sample(self, n=None, out=None):
        """One sample (GridFunction) or n samples (ndof x n). ``out``: an ndof x n buffer filled in place and returned
        (:mod:`ngsrng.inplace`)."""
        if out is not None:
            return self._sample_out(n, out)
        ns = 1 if n is None else int(n)
        eta = torch.from_numpy(self.source.streams().numpy().standard_normal((ns, self.nrand)).T)  # sample by sample
        x = np.repeat(self._mean[:, None], ns, axis=1)
        x[self.space.free] += self.apply(eta).numpy()
        if n is None:
            gf = ngs.GridFunction(self.space.fes)
            gf.vec.FV().NumPy()[:] = x[:, 0]
            return gf
        return torch.from_numpy(x)
