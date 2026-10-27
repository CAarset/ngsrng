"""Inverse-transform maps and device-native iid samplers."""

import numpy as np
import pytest
import scipy.special as sps
import scipy.stats as st
import torch

from ngsrng.distributions.inversion import TABLE_MIN_VALUES, Inversion, _q_exact, _q_newton
from ngsrng.distributions.mixtures import torch_gamma, torch_inverse_gaussian
from ngsrng.distributions.pointwise import XP

Z = np.concatenate([np.linspace(-9.0, 9.0, 3001), np.random.default_rng(0).standard_normal(20000)])


def _reference(dist, z):
    with np.errstate(all="ignore"):
        return np.where(z <= 0, dist.ppf(sps.ndtr(z)), dist.isf(sps.ndtr(-z)))


@pytest.mark.parametrize(
    "dist",
    [st.gamma(2.0), st.gamma(0.2), st.beta(0.5, 0.5), st.beta(2, 5), st.t(1.5), st.t(30), st.f(5, 10),
     st.invgauss(0.5), st.chi2(3.5), st.vonmises(2.0), st.genextreme(0.3)],
    ids=lambda d: f"{d.dist.name}{d.args}",
)
def test_table_accuracy(dist):
    inv = Inversion(dist, method="table", tol=1e-10)
    assert inv.method == "table"
    q = inv(Z, XP(False))
    ref = _reference(dist, Z)
    ok = np.isfinite(ref)
    assert np.isfinite(q).all()
    assert np.max(np.abs(q[ok] - ref[ok]) / (np.abs(ref[ok]) + inv.s)) < 4e-10
    qt = inv(torch.from_numpy(Z), XP(True)).numpy()
    np.testing.assert_allclose(qt, q, rtol=1e-13, atol=1e-300)


@pytest.mark.parametrize("dist", [st.poisson(3.0), st.poisson(1e4), st.binom(10, 0.3), st.nbinom(5, 0.4),
                                  st.geom(0.2), st.bernoulli(0.1)], ids=lambda d: f"{d.dist.name}{d.args}")
def test_discrete_table_is_exact(dist):
    """The defining property: k = min{k : F(k) >= Phi(z)}, from the survival side S(k) <= Phi(-z) for z > 0 (scipy's
    generic isf computes ppf(1 - q) and is off by one deep in the upper tail, so it is not the reference)."""
    inv = Inversion(dist, method="discrete-table")
    z = Z[np.abs(Z) <= 8.0]
    k = inv(z, XP(False))
    lower = z <= 0
    u, s = sps.ndtr(z), sps.ndtr(-z)
    assert np.all(dist.cdf(k[lower]) >= u[lower]) and np.all(dist.cdf(k[lower] - 1) < u[lower])
    assert np.all(dist.sf(k[~lower]) <= s[~lower]) and np.all(dist.sf(k[~lower] - 1) > s[~lower])
    kt = inv(torch.from_numpy(z), XP(True)).numpy()  # torch's Phi may differ by an ulp at exact CDF ties
    assert np.mean(kt != k) < 1e-3 and np.abs(kt - k).max() <= 1
    assert np.mean(k == _reference(dist, z)) > 0.99


@pytest.mark.parametrize("dist", [st.gamma(0.7), st.beta(2, 3), st.t(1.2), st.invgauss(0.5), st.cauchy()],
                         ids=lambda d: f"{d.dist.name}{d.args}")
def test_newton_quantile(dist):
    """Relative accuracy ~1e-13 against the defining equation, also where scipy's own ppf is inaccurate."""
    z = Z[np.abs(Z) <= 8.5]
    q = _q_newton(dist, z)
    lower = z <= 0
    logp = np.where(lower, dist.logcdf(q), dist.logsf(q))
    np.testing.assert_allclose(logp, sps.log_ndtr(-np.abs(z)), rtol=1e-11, atol=1e-13)


def test_exact_quantile_beyond_scipy():
    q = _q_exact(st.f(5, 10), np.array([8.5, 10.0]))  # scipy's isf returns nan this far out
    assert np.isfinite(q).all() and q[1] > q[0]
    v = _q_exact(st.vonmises(2.0), np.array([-9.0, 0.0, 9.0]))  # scipy's von Mises CDF leaves [0, 1] past +-pi
    np.testing.assert_allclose(v, [-np.pi, 0.0, np.pi], atol=1e-7)


def test_auto_method_selection():
    inv = Inversion(st.gamma(2.0))
    inv(np.zeros(10), XP(False))
    assert inv.method is None  # small calls use ppf without building a table
    inv(np.zeros(TABLE_MIN_VALUES), XP(False))
    assert inv.method == "table"
    assert Inversion(st.gamma(np.full((5, 1), 2.0))).info["inversion"] == "ppf"  # spatially varying shape
    assert not Inversion(st.gamma(2.0), method="ppf").torch_capable


@pytest.mark.parametrize("a", [0.3, 1.0, 7.5])
def test_torch_gamma(a):
    gen = torch.Generator().manual_seed(1)
    x = torch_gamma(torch.full((50000,), a, dtype=torch.float64), gen).numpy()
    assert st.kstest(x, st.gamma(a).cdf).pvalue > 1e-3


@pytest.mark.parametrize("shape", [0.2, 3.0])
def test_torch_inverse_gaussian(shape):
    gen = torch.Generator().manual_seed(2)
    x = torch_inverse_gaussian(torch.ones(50000, dtype=torch.float64), torch.full((50000,), shape, dtype=torch.float64),
                               gen).numpy()
    assert st.kstest(x, st.invgauss(1 / shape, scale=shape).cdf).pvalue > 1e-3
