"""Chi-square, (inverse-)Wishart and matrix log-normal SPD fields.

Laws ("constructions") and algorithms are kept apart: a construction fixes the joint law of the field, and the
algorithm used for it is chosen automatically and never changes the law.

- chi-square process: sum_{j<=k} z_j^2 of k independent standardised Gaussian fields (integer k);
- Wishart process: W = A (sum_{j<=df} z_j z_j^T) A^T with z_j in R^p independent standardised vector fields
  (Gelfand et al. 2004; integer df >= p), the canonical spatial Wishart field: exactly Wishart_p(A A^T, df) at every
  point, and its joint law is equivariant under rotations of the matrix frame;
- Bartlett field: W = A O T T^T O^T A^T with T lower triangular, T_ii = sqrt(chi^2_{df-i}) copula fields and
  T_ij (i > j) standardised Gaussian fields (Bartlett decomposition), and O a Haar-random rotation per sample (which
  removes the dependence on the coordinate order that a plain Bartlett factor would have). Exactly Wishart_p at every
  point for any real df > p - 1, with p(p+1)/2 Gaussian fields instead of df * p, but a different spatial
  dependence than the Wishart process.

Inverse-Wishart fields invert pointwise (scipy convention: X ~ IW(df, Psi) iff X^-1 ~ W(df, Psi^-1)); with the
Bartlett construction X = (L O) T^-T T^-1 (L O)^T (Psi = L L^T) needs only a triangular inverse. Small matrices use
explicit formulas (p <= 3) and broadcasting, so everything runs in NumPy and on a CUDA device.
"""

import numpy as np
import scipy.stats as st

from .inversion import Inversion
from .pointwise import XP, GaussianAtPoints, PointwiseField, is_conditioned, warn_copies_of_posterior


def _spd_inside(values, margin):
    """Add margin * I to symmetric matrices (npoints, p, p, n): all eigenvalues rise by the margin."""
    out = np.array(values, dtype=float, copy=True)
    idx = np.arange(out.shape[1])
    out[:, idx, idx, :] += margin
    return out


class _Copies(PointwiseField):
    """k independent copies of a standardised Gaussian field at the points: (npoints, k, n)."""

    torch_capable = True

    def __init__(self, points: GaussianAtPoints, copies, representation):
        super().__init__(points, representation)
        self.copies = copies
        if is_conditioned(points.gauss):
            warn_copies_of_posterior()
        self.law = {}

    @property
    def info(self) -> dict:
        return {**self.law, "gaussian_fields": self.copies, **self._common_info()}

    def _standard(self, n, center=True):
        p, dev = self.points, self.on_device
        if center:
            z = p.standardized(n * self.copies, dev)
        else:  # keep the mean: z = g / std
            g = p.draw(n * self.copies, dev)
            std = p._torch_ops()[2] if dev else p.std
            z = g / std[:, None]
        return z.reshape(p.npoints, self.copies, n), XP(dev)


class ChiSquareField(_Copies):
    """Chi-square process sum_{j<=k} z_j^2: chi^2_k marginals (noncentral, lambda = k (mean/std)^2, if
    ``noncentral``)."""

    def __init__(self, points, df, noncentral=False, representation="quadrature"):
        if int(df) != df or df < 1:
            raise ValueError("the chi-square process needs an integer df >= 1 (use the copula construction otherwise)")
        super().__init__(points, int(df), representation)
        self.noncentral = noncentral
        self.law = {"law": f"chi-square process ({int(df)} squared fields)"}

    def _values(self, n):
        z, _ = self._standard(n, center=not self.noncentral)
        return (z**2).sum(1)

    def _pull_inside(self, values, margin):
        return np.maximum(values, margin)


def haar_rotations(p, n, rng) -> np.ndarray:
    """n Haar-distributed orthogonal p x p matrices (p, p, n): QR of Gaussian matrices with the sign fix of
    Mezzadri (2007)."""
    g = rng.standard_normal((n, p, p))
    q, r = np.linalg.qr(g)
    q = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    return np.moveaxis(q, 0, -1)


def _spd_inverse(w, xp):
    """Pointwise inverse of symmetric matrices w (npoints, p, p, n)."""
    p = w.shape[1]
    if p == 1:
        return 1.0 / w
    if p == 2:
        a, b, d = w[:, 0, 0], w[:, 0, 1], w[:, 1, 1]
        det = a * d - b * b
        rows = [xp.m.stack([d, -b], 1), xp.m.stack([-b, a], 1)]
        return xp.m.stack(rows, 1) / det[:, None, None]
    if p == 3:
        m = [[w[:, i, j] for j in range(3)] for i in range(3)]
        cof = [[m[(j + 1) % 3][(i + 1) % 3] * m[(j + 2) % 3][(i + 2) % 3]
                - m[(j + 1) % 3][(i + 2) % 3] * m[(j + 2) % 3][(i + 1) % 3] for j in range(3)] for i in range(3)]
        det = m[0][0] * cof[0][0] + m[0][1] * cof[1][0] + m[0][2] * cof[2][0]
        return xp.m.stack([xp.m.stack(r, 1) for r in cof], 1) / det[:, None, None]
    if xp.torch:
        return xp.m.linalg.inv(w.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return np.moveaxis(np.linalg.inv(np.moveaxis(w, -1, 1)), 1, -1)


def _gram(x, xp):
    """sum over axis 1 of outer products: x (npoints, k, p, n) -> (npoints, p, p, n), one reduction per entry."""
    p = x.shape[2]
    out = [[None] * p for _ in range(p)]
    for a in range(p):
        for b in range(a + 1):
            out[a][b] = out[b][a] = (x[:, :, a, :] * x[:, :, b, :]).sum(1)
    return xp.m.stack([xp.m.stack(row, 1) for row in out], 1)


class WishartField(_Copies):
    """Wishart_p(scale, df) (or inverse-Wishart_p(df, scale) with ``inverse``) at every point; samples
    (npoints, p, p, n). ``construction``: "process" (default; integer df >= p) or "bartlett" (any real df > p - 1);
    see the module docstring."""

    _pull_inside = staticmethod(_spd_inside)

    def __init__(self, points, df, scale, inverse=False, construction="process", representation="quadrature",
                 rng=None, method="auto", tol=1e-10):
        scale = np.atleast_2d(np.asarray(scale, dtype=float))
        p = scale.shape[0]
        if construction == "process":
            if int(df) != df or df < p:
                raise ValueError("the Wishart process needs an integer df >= p (use construction='bartlett')")
            copies = int(df) * p
        elif construction == "bartlett":
            if df <= p - 1:
                raise ValueError("Wishart fields need df > p - 1")
            copies = p * (p + 1) // 2
        else:
            raise ValueError("construction must be 'process' or 'bartlett'")
        super().__init__(points, copies, representation)
        self.p, self.df, self.inverse, self.construction = p, df, inverse, construction
        # inverse: X^-1 ~ W(df, scale^-1). Process: factor of scale^-1, invert W. Bartlett: X = (L O) T^-T T^-1 (.)^T.
        if inverse and construction == "process":
            self.chol = np.linalg.cholesky(np.linalg.inv(scale))
        else:
            self.chol = np.linalg.cholesky(scale)
        self.rng = np.random.default_rng() if rng is None else rng
        if construction == "bartlett":
            self.chi = [Inversion(st.chi2(df - i), method=method, tol=tol) for i in range(p)]
            self.torch_capable = all(c.torch_capable for c in self.chi)
        self.law = {"law": ("inverse-" if inverse else "") + f"Wishart {construction} (p={p}, df={df})"}

    @property
    def info(self) -> dict:
        out = super().info
        if self.construction == "bartlett":
            out["chi_inversion"] = self.chi[0].info["inversion"]
        return out

    def _values(self, n):
        z, xp = self._standard(n)
        p = self.p
        tensor = (lambda a: xp.m.as_tensor(a, dtype=z.dtype, device=z.device)) if xp.torch else (lambda a: a)
        if self.construction == "process":
            z = z.reshape(-1, int(self.df), p, n)  # (npoints, df, p, n)
            chol = tensor(self.chol)
            x = xp.m.stack([sum(chol[a, c] * z[:, :, c, :] for c in range(a + 1)) for a in range(p)], 2)
            w = _gram(x, xp)
            return _spd_inverse(w, xp) if self.inverse else w
        # Bartlett factor T (npoints, p, p, n), lower triangular
        zero = z[:, 0, :] * 0
        t = [[zero] * p for _ in range(p)]
        k = 0
        for i in range(p):
            t[i][i] = xp.m.sqrt(self.chi[i](z[:, k, :], xp))
            k += 1
        for i in range(p):
            for j in range(i):
                t[i][j] = z[:, k, :]
                k += 1
        if self.inverse:  # U = T^-T: invert the triangular factor entrywise (forward substitution)
            ti = [[zero] * p for _ in range(p)]
            for i in range(p):
                ti[i][i] = 1.0 / t[i][i]
                for j in range(i):
                    ti[i][j] = -sum(t[i][m] * ti[m][j] for m in range(j, i)) / t[i][i]
            u = [[ti[j][i] for j in range(p)] for i in range(p)]  # transpose
        else:
            u = t
        c = np.einsum("ab,bcn->acn", self.chol, haar_rotations(p, n, self.rng))  # (p, p, n): per-sample L O
        c = tensor(c)
        # B = C U, then W = B B^T (sum over the inner index)
        b = [[sum(c[a, m][None, :] * u[m][j] for m in range(p)) for j in range(p)] for a in range(p)]
        x = xp.m.stack([xp.m.stack(row, 1) for row in b], 1)  # (npoints, p(a), p(j), n)
        return _gram(x.transpose(1, 2) if xp.torch else np.swapaxes(x, 1, 2), xp)


class MatrixLogNormalField(_Copies):
    """S = expm(mean + sigma G), G symmetric with independent standardised entries (diagonal N(0,1), off-diagonal
    N(0,1/2)): an SPD matrix log-normal field. Samples (npoints, p, p, n). p = 2 uses the closed-form exponential of
    symmetric 2 x 2 matrices; larger p a batched symmetric eigendecomposition."""

    _pull_inside = staticmethod(_spd_inside)

    def __init__(self, points, p, sigma=1.0, mean=None, representation="quadrature"):
        self.p, self.sigma = int(p), float(sigma)
        self.mean_matrix = np.zeros((self.p, self.p)) if mean is None else np.asarray(mean, dtype=float)
        if not np.allclose(self.mean_matrix, self.mean_matrix.T):
            raise ValueError("mean must be a symmetric matrix")
        super().__init__(points, self.p * (self.p + 1) // 2, representation)
        self.law = {"law": f"matrix log-normal (p={self.p})", "algorithm": "closed form" if self.p == 2 else "eigh"}

    def _values(self, n):
        z, xp = self._standard(n)  # (npoints, p(p+1)/2, n)
        p, s = self.p, self.sigma
        mean = self.mean_matrix
        iu = list(zip(*np.triu_indices(p)))
        g = [[None] * p for _ in range(p)]
        for k, (i, j) in enumerate(iu):
            g[i][j] = g[j][i] = mean[i, j] + s * (z[:, k, :] if i == j else np.sqrt(0.5) * z[:, k, :])
        if p == 2:
            a, b, d = g[0][0], g[0][1], g[1][1]
            m, h = 0.5 * (a + d), 0.5 * (a - d)
            r = xp.m.sqrt(h * h + b * b)
            em = xp.m.exp(m)
            safe = xp.where(r > 1e-8, r, 1.0 + 0 * r)
            shc = xp.where(r > 1e-8, xp.m.sinh(safe) / safe, 1.0 + r * r / 6)
            ch = xp.m.cosh(r)
            off = em * shc * b
            return xp.m.stack([xp.m.stack([em * (ch + shc * h), off], 1), xp.m.stack([off, em * (ch - shc * h)], 1)], 1)
        sym = xp.m.stack([xp.m.stack(row, -1) for row in g], -2)  # (npoints, n, p, p)
        lam, vec = xp.m.linalg.eigh(sym)
        e = xp.m.exp(lam)
        out = sum(e[..., k, None, None] * vec[..., :, k, None] * vec[..., None, :, k] for k in range(p))  # V e^L V^T
        return out.permute(0, 2, 3, 1) if xp.torch else np.moveaxis(out, 1, -1)
