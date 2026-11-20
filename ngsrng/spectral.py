"""Matérn fields with random correlation parameters, exactly, through one spectral basis.

The Matérn Gaussians of ngsrng are functions of one fixed operator pair. For the SPDE on a space (the natural or a
Robin condition with a given coefficient, Dirichlet DoFs, its interior-penalty form on L2 spaces) and for the torus
construction of :class:`ngsrng.StationaryMatern` (the SPDE on a periodic box), with K0 the form without its kappa^2
term and M the mass matrix on the free DoFs, and the generalised eigenpairs K0 Phi = M Phi Lambda (Phi^T M Phi = I),
the coefficient covariance with exact white noise is

    Sigma(length, nu) = c^2 (A^-1 M)^alpha M^-1 = Phi diag(c^2 (kappa^2 + lambda)^-alpha) Phi^T,
    A = K0 + kappa^2 M,  kappa = sqrt(8 nu) / length,  alpha = nu + d/2,  c^2 = 1 / var_{R^d}(length, nu),

the covariance ngsrng's pipeline samples (exactly, or to the accuracy of its rational approximation where it uses one:
fractional alpha, odd alpha on forms that couple elements). Phi and Lambda do not depend on the parameters. With P the
linear map from the operator's coefficients to the field's latent values (the projection from the torus to the
domain's FE space, then the evaluation at the points of a pointwise family, or the selected coefficients of a Gaussian
field) and B = P Phi (|S| x r), a draw with parameters theta is

    w = B (s_theta * z),  s_theta = c (kappa^2 + lambda)^(-alpha/2),  z ~ N(0, I_r),

and, for pointwise families, its exact standardisation w / sigma_theta with sigma_theta^2 = (B * B) s_theta^2: one
product for the values and one for the variances per draw. The torus box is the one of the largest padding over the
parameters' range (images farther away than that only bring the field closer to the Matérn covariance of R^d).

Truncation per draw: the modes are ordered by lambda, the same order of decreasing weight for every theta. A draw
keeps the leading r(theta) modes: those carrying all but ``tol`` of the latent variance (trace), or, without
``tol``, the modes whose share is at least ``rtol`` of the largest. r(theta) comes from a table over a grid of the
parameters' range (the largest over the grid cell around theta), rounded up to a ladder r_max 2^(-k/2), so that a
batch's draws fall into few groups, one product each. Pointwise families are standardised by the kept modes' variance,
so their marginals stay exact.
"""

import math

import numpy as np
from scipy.special import gammaln

__all__ = ["SpectralModes", "matern_spectrum"]

GRID = 33  # grid points per random parameter of the rank table (a truncation detail, not the law)
LADDER = 2 ** -0.5  # ratio of consecutive ranks a draw's rank is rounded up to (performance, not the law)


class SpectralModes:
    """The generalised eigenpairs of an operator pair as a field's free-DoF coefficients: ``lam`` (r,), ascending,
    and ``modes`` (nfree x r), the coefficients of the field for each unit mode (P Phi)."""

    def __init__(self, lam, modes, dim, source):
        order = np.argsort(lam, kind="stable")
        self.lam = np.clip(np.asarray(lam, dtype=np.float64)[order], 0.0, None)
        self.modes = np.ascontiguousarray(np.asarray(modes, dtype=np.float64)[:, order])
        self.dim, self.source = int(dim), source

    def __repr__(self):
        return f"SpectralModes({self.lam.size} modes of {self.source}, {self.modes.shape[0]} free DoFs)"


def matern_spectrum(length, nu, dim):
    """Per draw (arrays of lengths and smoothnesses): kappa^2, alpha / 2 and log c of the Matérn weights
    s = c (kappa^2 + lambda)^(-alpha/2) (unit variance on R^dim)."""
    length, nu = np.broadcast_arrays(np.asarray(length, dtype=float), np.asarray(nu, dtype=float))
    kappa2 = 8.0 * nu / length**2
    log_var = gammaln(nu) - gammaln(nu + dim / 2) - dim / 2 * math.log(4 * math.pi) - nu * np.log(kappa2)
    return kappa2, 0.5 * (nu + dim / 2), -0.5 * log_var


def log_weights(lam, length, nu, dim):
    """log s (n x r) for draws with the given parameters."""
    kappa2, half_alpha, logc = matern_spectrum(length, nu, dim)
    return logc[:, None] - half_alpha[:, None] * np.log(kappa2[:, None] + lam[None, :])


def rank_table(lam, colnorm2, grids, dim, fixed, tol, rtol, rank):
    """Kept modes at every grid point of the random parameters: (table, flat parameter values). ``grids``: {name:
    values} of the random parameters; ``fixed``: the others' values; ``colnorm2``: |B_j|^2."""
    names = list(grids)
    mesh = np.meshgrid(*[grids[k] for k in names], indexing="ij")
    vals = {k: m.ravel() for k, m in zip(names, mesh)}
    length = vals.get("length", np.full(mesh[0].size, fixed.get("length", np.nan)))
    nu = vals.get("nu", np.full(mesh[0].size, fixed.get("nu", np.nan)))
    lw = log_weights(lam, length, nu, dim)
    share = np.exp(2 * (lw - lw.max(axis=1, keepdims=True))) * colnorm2[None, :]  # relative variance per mode
    total = share.sum(axis=1, keepdims=True)
    if tol is not None:
        keep = (np.cumsum(share, axis=1) / total < 1.0 - tol).sum(axis=1) + 1
    elif rtol:
        top = share.max(axis=1, keepdims=True)
        above = share > rtol * top
        keep = np.where(above.any(axis=1), lam.size - np.argmax(above[:, ::-1], axis=1), 1)
    else:
        keep = np.full(share.shape[0], lam.size)
    keep = np.minimum(keep, lam.size if rank is None else int(rank))
    return keep.reshape(mesh[0].shape).astype(np.int64)


def ladder(rmax):
    """The ranks a draw's rank is rounded up to: r_max, r_max 2^-1/2, ..., down to 1 (descending)."""
    out, r = [], float(rmax)
    while r >= 1.0:
        out.append(int(math.ceil(r)))
        r *= LADDER
    return np.unique(np.asarray(out + [1], dtype=np.int64))[::-1]


def draw_ranks(spectral, theta):
    """The rank of each draw (n,): the table's largest over the grid cell around its parameters, rounded up to the
    ladder. ``theta``: {name: (n,) values} of the random parameters."""
    table = np.asarray(spectral["table"])
    names = list(spectral["grids"])
    n = len(next(iter(theta.values()))) if theta else 0
    r = np.zeros(n, dtype=np.int64)
    lo_hi = []
    for k in names:
        g = np.asarray(spectral["grids"][k])
        i = np.clip(np.searchsorted(g, theta[k], side="right") - 1, 0, g.size - 2)
        lo_hi.append((i, i + 1))
    for corner in range(2 ** len(names)):
        idx = tuple(lo_hi[j][(corner >> j) & 1] for j in range(len(names)))
        r = np.maximum(r, table[idx])
    steps = np.asarray(spectral["ladder"])  # descending
    pos = np.searchsorted(-steps, -r, side="right") - 1  # the smallest ladder rank >= r
    return steps[np.clip(pos, 0, steps.size - 1)]
