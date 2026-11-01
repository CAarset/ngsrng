"""Precisions given only as an action ("apply-only").

For an SPD precision Q (coefficients -> load vectors, Sigma = Q^-1) and A = M^-1 Q (M-self-adjoint),
c = A^-1/2 M^-1 z with z ~ N(0, M) has Cov(c) = Q^-1 exactly. With the rational approximation
r(lambda) ~ lambda^-1/2 of ``ngsrng.rational`` this is

    c ~ c0 M^-1 z + sum_l c_l (Q + t_l M)^-1 z,

and all shifted systems share one Krylov space for the M^-1-preconditioned operator A + t I. So a single multi-shift
preconditioned CG run (one application of Q and of M^-1 per iteration) solves all of them.
"""

import warnings

import numpy as np
import torch


class MultiShiftPCG:
    """Solve (Q + t_l M) x_l = b for all shifts t_l >= 0 at once, with preconditioner M^-1.

    ``apply_q`` and ``apply_minv`` map torch tensors (n x k) to (n x k). Columns of b are independent systems.
    Returns sum_l w_l x_l (the weighted combination is all the rational sampler needs) and the iteration count.
    Convergence: the residual of every shifted system is collinear with the seed residual (seed shift 0), with factor
    zeta_l; iteration stops when max_l |zeta_l| |r| <= tol |b| (in the M^-1 norm) for every column.
    """

    def __init__(self, apply_q, apply_minv, shifts, weights, tol=1e-8, maxiter=20000):
        self.apply_q, self.apply_minv = apply_q, apply_minv
        self.shifts = np.asarray(shifts, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.tol, self.maxiter = tol, maxiter
        self.iterations = []

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        squeeze = b.ndim == 1
        b = b[:, None] if squeeze else b
        n, k = b.shape
        s = torch.as_tensor(self.shifts, dtype=b.dtype, device=b.device)[:, None]  # (S, 1)
        w = torch.as_tensor(self.weights, dtype=b.dtype, device=b.device)
        x = torch.zeros((len(self.shifts), n, k), dtype=b.dtype, device=b.device)
        r = b.clone()
        z = self.apply_minv(r)
        p_seed = z.clone()
        ps = z.unsqueeze(0).repeat(len(self.shifts), 1, 1)  # shifted search directions
        rz = (r * z).sum(0)
        bnorm = torch.sqrt(rz).clamp_min(torch.finfo(b.dtype).tiny)
        zeta_old = torch.ones((len(self.shifts), k), dtype=b.dtype, device=b.device)
        zeta = torch.ones_like(zeta_old)
        alpha_old = torch.ones(k, dtype=b.dtype, device=b.device)
        beta_old = torch.zeros(k, dtype=b.dtype, device=b.device)
        active = torch.ones((len(self.shifts), k), dtype=torch.bool, device=b.device)
        it = 0
        for it in range(1, self.maxiter + 1):
            qp = self.apply_q(p_seed)
            pqp = (p_seed * qp).sum(0)
            # columns with r = 0 (zero right-hand side, or converged exactly) would divide 0 / 0: no step
            alpha = torch.where(pqp > 0, rz / pqp, torch.zeros_like(rz))
            # zeta recurrence for (A + sigma): r_sigma = zeta_sigma r (Frommer 2003, preconditioned form). Converged
            # shifted systems are frozen: their zeta would otherwise underflow (0/0) for large shifts.
            denom = alpha * beta_old * (zeta_old - zeta) + zeta_old * alpha_old * (1 + s * alpha)
            zeta_new = torch.where(active, zeta * zeta_old * alpha_old / denom, zeta)
            ratio = torch.where(active, zeta_new / zeta, torch.zeros_like(zeta))
            x += (alpha * ratio).unsqueeze(1) * ps
            r = r - alpha * qp
            z = self.apply_minv(r)
            rz_new = (r * z).sum(0)
            beta = torch.where(rz > 0, rz_new / rz, torch.zeros_like(rz))
            ps = torch.where(
                active.unsqueeze(1), zeta_new.unsqueeze(1) * z.unsqueeze(0) + (beta * ratio**2).unsqueeze(1) * ps, ps
            )
            p_seed = z + beta * p_seed
            shifted_res = torch.sqrt(rz_new.clamp_min(0)) * zeta_new.abs() / bnorm  # (S, k)
            active = active & (shifted_res > 1e-3 * self.tol)
            zeta_old, zeta, alpha_old, beta_old, rz = zeta, zeta_new, alpha, beta, rz_new
            if shifted_res.max() <= self.tol:
                break
        else:
            warnings.warn(
                f"multi-shift PCG did not converge in {self.maxiter} iterations "
                f"(residual {float(shifted_res.max()):.1e} > tol {self.tol:.0e}). Apply-only sampling scales with "
                "sqrt(cond(M^-1 Q)); very ill-conditioned precisions need a preconditioner, a solve, or an "
                "assembled/factorised form",
                stacklevel=2,
            )
        self.iterations.append(it)
        out = torch.einsum("s,snk->nk", w, x)
        return out[:, 0] if squeeze else out


def lanczos_interval(apply_q, apply_minv, n, iters=200, seed=0, dtype=torch.float64, device="cpu"):
    """Extreme eigenvalues of A = M^-1 Q by Lanczos in the M inner product (full reorthogonalisation).

    Works without products with M: each basis vector v is kept together with M v (start: v = M^-1 x, M v = x).
    Returns (lmin, lmax, residual bound of the lmax Ritz value). Reliable for lmax; use ``inverse_iteration_min`` for
    lmin.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, 1, generator=gen, dtype=dtype, device=device)
    v = apply_minv(x)
    nrm = torch.sqrt((v * x).sum())
    v, mv = v / nrm, x / nrm
    mv_prev = torch.zeros_like(mv)
    beta = torch.zeros((), dtype=dtype, device=device)
    vs, mvs, alphas, betas = [], [], [], []
    for _ in range(min(iters, n)):
        vs.append(v)
        mvs.append(mv)
        qv = apply_q(v)  # = M A v
        alpha = (v * qv).sum()
        wq = qv - alpha * mv - beta * mv_prev  # M w
        w = apply_minv(wq)
        for vj, mvj in zip(vs, mvs):  # full reorthogonalisation in the M inner product
            c = (w * mvj).sum()
            w, wq = w - c * vj, wq - c * mvj
        beta_new = torch.sqrt((w * wq).sum().clamp_min(0))
        alphas.append(float(alpha))
        betas.append(float(beta_new))
        if beta_new <= 1e-14 * abs(alpha):
            break
        mv_prev, beta = mv, beta_new
        v, mv = w / beta_new, wq / beta_new
    k = len(alphas)
    t = np.diag(alphas) + np.diag(betas[: k - 1], 1) + np.diag(betas[: k - 1], -1)
    lam, y = np.linalg.eigh(t)
    return float(lam[0]), float(lam[-1]), float(abs(betas[k - 1] * y[-1, -1]))


def inverse_iteration_min(
    apply_q, apply_m, apply_minv, n, tol=1e-6, iters=20, seed=0, dtype=torch.float64, device="cpu"
):
    """Smallest eigenvalue of A = M^-1 Q by inverse iteration v <- Q^-1 M v, solving with M^-1-preconditioned CG.

    Lanczos converges far too slowly at the bottom of PDE spectra (in a test, 150 iterations overestimated lambda_min by
    a factor of 22); inverse iteration converges geometrically with ratio lambda_1 / lambda_2. The Rayleigh quotient
    approaches lambda_min from above, so the caller applies a safety factor.
    """
    solver = MultiShiftPCG(apply_q, apply_minv, [0.0], [1.0], tol=tol)
    gen = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(n, 1, generator=gen, dtype=dtype, device=device)
    lam = None
    for _ in range(iters):
        v = solver.solve(apply_m(v))
        v = v / torch.sqrt((v * apply_m(v)).sum())
        lam_new = float((v * apply_q(v)).sum())
        if lam is not None and abs(lam_new - lam) <= 1e-4 * lam_new:
            return lam_new
        lam = lam_new
    return lam
