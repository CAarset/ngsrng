"""Distribution families of random functions on FE spaces (the machinery behind ``ngsrng.Generator``).

- prescribed pointwise marginals by Gaussian copula: closed-form quantiles and numerical inversion of any
  scipy.stats distribution (``transforms``, ``inversion``), Dirichlet fields;
- quadratic forms: chi-square processes, Wishart processes and Bartlett fields, matrix log-normal (``quadratic``);
- scale mixtures that stay FE functions exactly: Student-t processes, type-G (NIG / variance-gamma) Levy-noise
  fields (``mixtures``).

The Gaussian random functions themselves are ``ngsrng.Gaussian`` and ``ngsrng.LowRankGaussian``. Pointwise families
are functions of the Gaussian's point values, so they need a Gaussian with point values (continuous samples) and a
representation in which "pointwise" is meaningful: values at the quadrature points of an IntegrationRuleSpace
(default; marginals exact at every point, which is what assembly sees), values at any given points, or their L2
projection into the FE space (``representation="projection"``; an FE function with smoothed marginals that may
leave the support). On posteriors they standardise with the prior.
"""

from .inversion import Inversion
from .mixtures import StudentTField, TypeGField
from .pointwise import GaussianAtPoints, PointwiseField
from .quadratic import ChiSquareField, MatrixLogNormalField, WishartField
from .transforms import CLOSED, SCIPY, DirichletField, MarginalField, indicator, marginal, named_marginal, transformed

__all__ = [
    "GaussianAtPoints",
    "PointwiseField",
    "Inversion",
    "MarginalField",
    "CLOSED",
    "SCIPY",
    "named_marginal",
    "marginal",
    "indicator",
    "transformed",
    "DirichletField",
    "ChiSquareField",
    "WishartField",
    "MatrixLogNormalField",
    "StudentTField",
    "TypeGField",
]
