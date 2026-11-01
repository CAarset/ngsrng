"""Karhunen-Loeve expansions with independent, non-Gaussian coefficients: random functions with a prescribed
covariance (exactly) and non-Gaussian laws.

    f = m + sum_k sqrt(lambda_k) xi_k phi_k,   xi_k iid, E xi = 0, Var xi = 1,

with (lambda_k, phi_k) the eigenpairs of the covariance operator C (L2-orthonormal phi_k). For any such law
E f = m and Cov f = sum_k lambda_k phi_k (x) phi_k = C (C_r for a rank-r truncation): C is the covariance of f
itself, not of a hidden Gaussian. Gaussian xi give N(m, C_r); bounded xi give bounded fields,
|f - m|(x) <= sup|xi| sum_k sqrt(lambda_k) |phi_k(x)| (the affine-parametric model of uncertainty quantification,
e.g. Cohen, DeVore & Schwab 2010; Schwab & Gittelson 2011). The law depends on the basis (only Gaussian laws are
rotation invariant), which is why the KL basis is used. The sample is an FE function exactly.
"""

import numpy as np
import torch

from .mixtures import _FEField

STANDARD = {
    "normal": lambda rng, shape: rng.standard_normal(shape),
    "uniform": lambda rng, shape: rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), shape),
    "rademacher": lambda rng, shape: 2.0 * rng.integers(0, 2, shape) - 1.0,
    "laplace": lambda rng, shape: rng.laplace(0.0, 1.0 / np.sqrt(2.0), shape),
}


def coefficient_law(coefficients):
    """A sampler (rng, shape) -> iid standardised values, and a description."""
    if isinstance(coefficients, str):
        if coefficients not in STANDARD:
            raise ValueError(f"coefficients must be one of {sorted(STANDARD)}, a frozen scipy.stats distribution "
                             "with finite variance, or a callable (rng, shape) -> standardised values")
        return STANDARD[coefficients], coefficients
    if hasattr(coefficients, "rvs") and hasattr(coefficients, "std"):
        mean, std = float(coefficients.mean()), float(coefficients.std())
        if not np.isfinite(std) or std <= 0:
            raise ValueError("the coefficient distribution needs a finite, positive variance")
        name = getattr(getattr(coefficients, "dist", None), "name", "scipy")
        return (lambda rng, shape: (coefficients.rvs(size=shape, random_state=rng) - mean) / std), f"{name} (scipy)"
    if callable(coefficients):
        return coefficients, getattr(coefficients, "__name__", "custom")
    raise TypeError("unsupported coefficients")


class KLField(_FEField):
    """f = m + sum_k sqrt(lambda_k) xi_k phi_k with iid standardised xi_k, from a KL expansion (``kl``: a
    LowRankGaussian with ``eigenvalues`` and ``modes``, see :func:`ngsrng.karhunen_loeve`)."""

    def __init__(self, gauss, kl, coefficients="normal", seed=None, truncated=True):
        self.gauss = gauss  # mean and space of the output
        self.space = gauss.space
        self.law, name = coefficient_law(coefficients)
        self.rng = np.random.default_rng(seed)
        self.scaled_modes = kl.modes[self.space.free] * np.sqrt(kl.eigenvalues)[None, :]
        self.rank = len(kl.eigenvalues)
        cov = f"the rank-{self.rank} truncation of the Gaussian's" if truncated else "the Gaussian's"
        self.info = {"law": f"Karhunen-Loeve expansion, rank {self.rank}, iid {name} coefficients "
                            f"(covariance exactly {cov})",
                     "gaussian": gauss.plan, "kl": kl.plan.get("structure", "")}

    def _coefficients(self, ns):
        xi = np.asarray(self.law(self.rng, (ns, self.rank)), dtype=float).T  # sample by sample
        return torch.from_numpy(self.scaled_modes @ xi)

