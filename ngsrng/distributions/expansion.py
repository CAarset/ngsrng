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

from ..backend import DeviceCache
from ..seeding import as_source
from .mixtures import _FEField


def _normal(rng, shape):
    return rng.standard_normal(shape)


def _uniform(rng, shape):
    return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), shape)


def _rademacher(rng, shape):
    return 2.0 * rng.integers(0, 2, shape) - 1.0


def _laplace(rng, shape):
    return rng.laplace(0.0, 1.0 / np.sqrt(2.0), shape)


# standardised (mean 0, variance 1) coefficient laws; named functions, since fields store them and must pickle
STANDARD = {"normal": _normal, "uniform": _uniform, "rademacher": _rademacher, "laplace": _laplace}


class ScipyLaw:
    """Standardised draws from a frozen scipy.stats distribution."""

    def __init__(self, dist):
        self.dist, self.mean, self.std = dist, float(dist.mean()), float(dist.std())

    def __call__(self, rng, shape):
        return (self.dist.rvs(size=shape, random_state=rng) - self.mean) / self.std


def coefficient_law(coefficients):
    """A sampler (rng, shape) -> iid standardised values, and a description."""
    if isinstance(coefficients, str):
        if coefficients not in STANDARD:
            raise ValueError(f"coefficients must be one of {sorted(STANDARD)}, a frozen scipy.stats distribution "
                             "with finite variance, or a callable (rng, shape) -> standardised values")
        return STANDARD[coefficients], coefficients
    if hasattr(coefficients, "rvs") and hasattr(coefficients, "std"):
        law = ScipyLaw(coefficients)
        if not np.isfinite(law.std) or law.std <= 0:
            raise ValueError("the coefficient distribution needs a finite, positive variance")
        name = getattr(getattr(coefficients, "dist", None), "name", "scipy")
        return law, f"{name} (scipy)"
    if callable(coefficients):
        return coefficients, getattr(coefficients, "__name__", "custom")
    raise TypeError("unsupported coefficients")


class KLField(_FEField):
    """f = m + sum_k sqrt(lambda_k) xi_k phi_k with iid standardised xi_k, from a KL expansion (``kl``: a
    LowRankGaussian with ``eigenvalues`` and ``modes``, see :func:`ngsrng.karhunen_loeve`). The sum is computed
    where ``kl`` computes (its GPU, or NumPy); the coefficients come from NumPy either way."""

    def __init__(self, gauss, kl, coefficients="normal", seed=None, truncated=True):
        self.gauss = gauss  # mean and space of the output
        self.space = gauss.space
        self.law, name = coefficient_law(coefficients)
        self.source = as_source(seed)  # ngsrng.seeding
        self.scaled_modes = kl.modes[self.space.free] * np.sqrt(kl.eigenvalues)[None, :]
        self.rank = len(kl.eigenvalues)
        self.torch_device = getattr(kl, "torch_device", None)
        self._device = DeviceCache()
        cov = f"the rank-{self.rank} truncation of the Gaussian's" if truncated else "the Gaussian's"
        self.info = {"law": f"Karhunen-Loeve expansion, rank {self.rank}, iid {name} coefficients "
                            f"(covariance exactly {cov})",
                     "gaussian": gauss.plan, "kl": kl.plan.get("structure", "")}

    def _coefficients(self, ns):
        xi = np.asarray(self.law(self.source.streams().numpy(), (ns, self.rank)), dtype=float).T  # sample by sample
        dev = self.torch_device
        if dev is not None:
            if "modes" not in self._device:
                self._device["modes"] = torch.from_numpy(np.ascontiguousarray(self.scaled_modes)).to(dev)
            return self._device["modes"] @ torch.from_numpy(np.ascontiguousarray(xi)).to(dev)
        return torch.from_numpy(self.scaled_modes @ xi)

