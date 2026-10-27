"""Rational approximation of lambda^-theta on a spectral interval, for fractional powers of SPD operators.

r(lambda) = c0 + sum_l c_l / (lambda + t_l)  ~  lambda^-theta  on [lmin, lmax], 0 < theta < 1,

with *relative* accuracy, so that r(M^-1 K) M^-1 = c0 M^-1 + sum_l c_l (K + t_l M)^-1 approximates (M^-1 K)^-theta M^-1
mode by mode. Relative (not uniform absolute) accuracy is what keeps the trace-norm covariance error independent of the
mesh: every eigenmode is approximated to the same relative tolerance.

The approximation comes from an AAA-type greedy barycentric fit (Nakatsukasa, Sete, Trefethen 2018) with relative
weighting on a logarithmic grid. lambda^-theta is a Stieltjes function, so good rational approximants have real
negative poles and positive residues. This is checked: then every shifted system K + t_l M is SPD and the sum is a
positive combination.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, nnls


@dataclass
class RationalPower:
    theta: float
    lmin: float
    lmax: float
    c0: float
    coeffs: np.ndarray  # c_l > 0
    shifts: np.ndarray  # t_l > 0
    rel_error: float  # max relative error on a fine grid over [lmin, lmax]

    def __call__(self, lam):
        lam = np.asarray(lam, dtype=float)
        return self.c0 + np.sum(self.coeffs[:, None] / (lam[None, :] + self.shifts[:, None]), axis=0)

    @property
    def degree(self) -> int:
        return len(self.shifts)


def fractional_power(theta, lmin, lmax, rtol=1e-7, max_degree=60, npts=4000) -> RationalPower:
    """Rational approximation of lambda^-theta on [lmin, lmax] with max relative error <= rtol (checked on a fine grid).

    1. Relative AAA fit; its poles are found as the real roots of the barycentric denominator on the negative axis
       (sign-change scan plus Brent in log(-x), which gives full relative precision over many decades), and the
       residues follow exactly.
    2. If double-precision AAA stagnates (very large spans), the AAA poles are augmented by geometrically spaced poles
       of increasing density and the residues refitted by nonnegative least squares (keeping the Stieltjes sign
       structure).
    """
    if not 0.0 < theta < 1.0:
        raise ValueError("theta must lie in (0, 1)")
    if not 0.0 < lmin < lmax:
        raise ValueError("need 0 < lmin < lmax")
    span = lmax / lmin  # work in x = lambda / lmin and scale back at the end
    x = np.geomspace(1.0, span, npts)
    f = x**-theta
    fine = np.geomspace(1.0, span, 20 * npts)

    def rel_err(c0, res, poles):
        approx = c0 + np.sum(res[:, None] / (fine[None, :] + poles[:, None]), axis=0)
        return float(np.max(np.abs(approx * fine**theta - 1.0)))

    z, fz, w = _aaa_relative(x, f, rtol / 10, max_degree)
    poles = _negative_roots(z, w, span)  # as t > 0: denominator roots at -t
    res = np.array([_bary_residue(-t, z, fz, w) for t in poles])
    c0 = float(np.sum(w * fz) / np.sum(w))
    ok = poles.size and np.all(res > 0) and c0 >= 0
    err = rel_err(c0, res, poles) if ok else np.inf
    density, refitted = 4, False
    while err > rtol and density <= 64:
        refitted = True
        cand = np.unique(np.r_[poles, np.geomspace(1e-4, span * 1e4, int(density * np.log10(span * 1e8)))])
        c0, res, poles = _nnls_fit(x, f, cand)
        err = rel_err(c0, res, poles)
        density *= 2
    if err > rtol:
        raise RuntimeError(f"rational approximation of lambda^-{theta} (span {span:.1e}) reached {err:.1e} > {rtol}")
    if refitted:  # NNLS path: prune poles greedily while the tolerance holds
        c0, res, poles, err = _prune(x, f, c0, res, poles, rtol, rel_err)
    order = np.argsort(poles)
    scale = lmin**-theta
    return RationalPower(
        theta=theta,
        lmin=lmin,
        lmax=lmax,
        c0=float(scale * c0),
        coeffs=scale * lmin * res[order],
        shifts=poles[order] * lmin,
        rel_error=err,
    )


def _aaa_relative(x, f, tol, mmax):
    """Greedy AAA with relative error: support points where |r - f|/|f| is largest; least squares weighted by 1/f."""
    mask = np.ones(len(x), dtype=bool)
    z, fz = [], []
    r = np.full_like(f, np.mean(f))
    w = None
    for _ in range(mmax):
        rel = np.where(mask, np.abs(r - f) / f, -1.0)
        j = int(np.argmax(rel))
        if rel[j] <= tol and z:
            break
        z.append(x[j])
        fz.append(f[j])
        mask[j] = False
        zz, ff = np.array(z), np.array(fz)
        cauchy = 1.0 / (x[mask, None] - zz[None, :])
        loewner = (f[mask, None] - ff[None, :]) * cauchy / f[mask, None]
        _, _, vh = np.linalg.svd(loewner, full_matrices=False)
        w = vh[-1].conj()
        r = f.copy()
        r[mask] = (cauchy @ (w * ff)) / (cauchy @ w)
    return np.array(z), np.array(fz), w


def _negative_roots(z, w, span):
    """t > 0 with sum_j w_j / (-t - z_j) = 0: real negative poles of the barycentric approximant."""
    d = lambda s: np.sum(w / (-np.exp(s) - z))  # noqa: E731
    s = np.linspace(np.log(1e-12), np.log(span * 1e12), 40000)
    vals = np.array([d(v) for v in s])
    idx = np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))
    return np.array([np.exp(brentq(d, s[i], s[i + 1], xtol=1e-15, rtol=1e-15)) for i in idx])


def _bary_residue(p, z, fz, w):
    return np.sum(w * fz / (p - z)) / (-np.sum(w / (p - z) ** 2))


def _prune(x, f, c0, res, poles, rtol, rel_err):
    """Greedily remove poles (least relative contribution first), refitting each time; poles whose removal breaks the
    tolerance are kept."""
    err = rel_err(c0, res, poles)
    essential = set()
    while True:
        importance = np.max(res[:, None] / ((x[None, :] + poles[:, None]) * f[None, :]), axis=1)
        candidates = [i for i in np.argsort(importance) if poles[i] not in essential]
        if not candidates or poles.size <= 1:
            return c0, res, poles, err
        i = candidates[0]
        c0_t, res_t, poles_t = _nnls_fit(x, f, np.delete(poles, i))
        err_t = rel_err(c0_t, res_t, poles_t)
        if err_t <= rtol:
            c0, res, poles, err = c0_t, res_t, poles_t, err_t
        else:
            essential.add(poles[i])


def _nnls_fit(x, f, poles):
    """c0, c_l >= 0 minimising the relative error of c0 + sum c_l / (x + t_l) against f; drops zero terms."""
    a = np.hstack([np.ones((len(x), 1)), 1.0 / (x[:, None] + poles[None, :])]) / f[:, None]
    sc = np.linalg.norm(a, axis=0)
    coef, _ = nnls(a / sc, np.ones(len(x)), maxiter=50 * a.shape[1])
    coef /= sc
    keep = coef[1:] > 0
    return float(coef[0]), coef[1:][keep], poles[keep]


def spectral_interval(apply_k_inv, apply_m_inv, k, m, iters=40, margin=2.0, seed=0):
    """Estimate [lmin, lmax] of M^-1 K by inverse and direct power iteration, widened by ``margin`` on both sides.

    Rayleigh quotients overestimate lmin and underestimate lmax; widening the interval is always safe (the rational
    approximation just gets slightly longer). ``apply_k_inv``/``apply_m_inv`` map (n,) numpy arrays to (n,) arrays.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(k.shape[0])
    for _ in range(iters):
        v = apply_k_inv(m @ v)
        v /= np.linalg.norm(v)
    lmin = (v @ (k @ v)) / (v @ (m @ v))
    v = rng.standard_normal(k.shape[0])
    for _ in range(iters):
        v = apply_m_inv(k @ v)
        v /= np.linalg.norm(v)
    lmax = (v @ (k @ v)) / (v @ (m @ v))
    return lmin / margin, lmax * margin
