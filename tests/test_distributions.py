"""The laws of the distribution families: exact laws checked on i.i.d. values at a false-failure rate of 1e-12
(:mod:`stat_checks`), and exact constructions."""

import ngsolve as ngs
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.stats as st
import torch
from stat_checks import assert_bounded_mean, assert_law, assert_normal, assert_uniform_pit

import ngsrng
from ngsrng.distributions import (
    ChiSquareField,
    DirichletField,
    GaussianAtPoints,
    MatrixLogNormalField,
    StudentTField,
    TypeGField,
    WishartField,
    indicator,
    marginal,
    named_marginal,
    transformed,
)
from ngsrng.distributions.pointwise import XP
from ngsrng.reference import exact_covariance

N = 10000


@pytest.fixture(scope="module")
def prior():
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25))
    return ngsrng.Gaussian(ngs.H1(mesh, order=2), ngsrng.matern(4.0, 2, scale=10.0), dtype="float64", seed=0, mean=0.3)


@pytest.fixture(scope="module")
def points(prior):
    return GaussianAtPoints(prior, std="exact")


def _probe(v):
    """Three well-separated points (first, middle, last)."""
    return (0, v.shape[0] // 2, v.shape[0] - 1)


def test_exact_pointwise_std(prior, points):
    e = points.rule.evaluation.toarray()
    ref = np.sqrt(np.einsum("ij,jk,ik->i", e, exact_covariance(prior), e))
    np.testing.assert_allclose(points.std, ref, rtol=1e-12)
    np.testing.assert_allclose(points.mean, 0.3, rtol=1e-12)
    assert GaussianAtPoints(prior).std_method == "exact"  # "auto" on a small problem


@pytest.mark.parametrize(
    "name, args, ref",
    [
        ("uniform", (), st.uniform()),
        ("laplace", (0.5, 2.0), st.laplace(0.5, 2.0)),
        ("gumbel", (), st.gumbel_r()),
        ("weibull", (1.5,), st.weibull_min(1.5)),
        ("standard_cauchy", (), st.cauchy()),
        ("beta", (2.0, 5.0), st.beta(2, 5)),
        ("gamma", (3.0, 2.0), st.gamma(3.0, scale=2.0)),
        ("standard_t", (3.0,), st.t(3.0)),
        ("vonmises", (0.5, 2.0), st.vonmises(2.0, loc=0.5)),
    ],
)
def test_copula_marginals(points, name, args, ref):
    v = named_marginal(points, name, *args).sample(N)
    for q in _probe(v):
        assert_law(v[q], ref.cdf, what=f"{name} at point {q}")


def test_scipy_marginal_and_discrete(points):
    v = marginal(points, st.genextreme(0.2)).sample(N)
    assert_law(v[0], st.genextreme(0.2).cdf)
    k = named_marginal(points, "poisson", 3.0).sample(N)
    poisson = st.poisson(3.0)
    for q in _probe(k):
        assert np.array_equal(k[q], np.round(k[q]))
        assert_law(k[q], poisson.cdf, left_cdf=lambda t: poisson.cdf(t - 1), what=f"poisson at point {q}")


def test_chi_square_and_indicator(points):
    c = ChiSquareField(points, 3).sample(N)
    ind = indicator(points, 1.0).sample(N)
    assert set(np.unique(ind)) <= {0.0, 1.0}
    for q in _probe(c):
        assert_law(c[q], st.chi2(3).cdf, what=f"chi-square at point {q}")
        assert_bounded_mean(ind[q], st.norm.sf(1.0), 0.0, 1.0, what=f"indicator at point {q}")


@pytest.mark.parametrize("construction, df", [("process", 4), ("bartlett", 4), ("bartlett", 2.5)])
@pytest.mark.parametrize("inverse", [False, True])
def test_wishart_marginal_law(points, construction, df, inverse):
    """a^T W a / a^T S a ~ chi^2_df for W ~ W_p(S, df); for X ~ IW(df, Psi), X^-1 ~ W_p(Psi^-1, df)."""
    if inverse and df < 3:
        df = 3.5  # IW needs df > p - 1 as well; keep a finite mean
    scale = np.array([[2.0, 0.5], [0.5, 1.0]])
    w = WishartField(points, df, scale, inverse=inverse, construction=construction, seed=5).sample(2000)
    mats = np.moveaxis(w, -1, 1)  # (npoints, n, 2, 2)
    assert np.abs(mats - np.swapaxes(mats, -1, -2)).max() < 1e-10 and np.linalg.eigvalsh(mats).min() > 0
    s = np.linalg.inv(scale) if inverse else scale
    m = np.linalg.inv(mats) if inverse else mats
    for a in (np.array([1.0, 0.0]), np.array([0.6, -0.8])):
        for q in _probe(w):
            x = np.einsum("i,nij,j->n", a, m[q], a) / (a @ s @ a)
            assert_law(x, st.chi2(df).cdf, what=f"a^T W a at point {q}")


def test_wishart_process_needs_integer_df(points):
    with pytest.raises(ValueError):
        WishartField(points, 2.5, np.eye(2))
    with pytest.raises(ValueError):
        WishartField(points, 0.5, np.eye(2), construction="bartlett")


def test_matrix_lognormal_closed_form_and_eigh(points):
    for p in (2, 3):
        field = MatrixLogNormalField(points, p, 0.7, mean=0.1 * np.eye(p))
        z = np.random.default_rng(p).standard_normal((points.npoints, p * (p + 1) // 2, 3))
        field._standard = lambda n, z=z: (z, XP(False))
        s = field._values(3)
        iu = np.triu_indices(p)
        for q in (0, 7):
            for k in range(3):
                g = np.zeros((p, p))
                g[iu] = z[q, :, k] * np.where(iu[0] == iu[1], 1.0, np.sqrt(0.5))
                g = g + np.triu(g, 1).T
                np.testing.assert_allclose(s[q, :, :, k], sla.expm(0.1 * np.eye(p) + 0.7 * g), rtol=1e-12)


def test_dirichlet(points):
    alpha = np.array([0.5, 2.0, 3.0])
    x = DirichletField(points, alpha).sample(N)
    assert x.shape == (points.npoints, 3, N)
    np.testing.assert_allclose(x.sum(1), 1.0, atol=1e-12)
    for k in range(3):
        assert_law(x[0, k], st.beta(alpha[k], alpha.sum() - alpha[k]).cdf, what=f"component {k}")


def _functional(prior, x):
    """The functional f = int (x - mean) dx of draws x (ndof x n), and its weights w = M 1 (all dofs)."""
    w = np.ones(prior.space.ndof) @ prior.space.mass
    return w @ (np.asarray(x) - prior.mean[:, None]), w


def test_student_t_covariance(prior):
    """The t-process's finite-dimensional laws are multivariate t: a functional is sqrt(its Gaussian variance) times
    a t_8 variable (hence the covariance 8/6 of the Gaussian's)."""
    f, w = _functional(prior, StudentTField(prior, 8, seed=7).sample(N))
    wf = w[prior.space.free]
    assert_law(f / np.sqrt(wf @ exact_covariance(prior) @ wf), st.t(8).cdf)


@pytest.mark.parametrize("mixing", ["nig", "gamma"])
def test_type_g_keeps_covariance_and_has_heavy_tails(prior, mixing):
    """The exact law: the mixing variables V_e (mean 1, variance nu / |e|: inverse Gaussian or gamma), and, given them,
    a Gaussian field whose noise on element e is scaled by sqrt(V_e), so that a functional f = w^T (x - mean) is
    N(0, sum_k (L^T w)_k^2 V_e(k)). Hence E V = 1 keeps the Gaussian's covariance, and Var V > 0 gives heavy tails."""
    field = TypeGField(prior, mixing, nu=0.05, seed=8)
    shape = field.volumes[:, None] / field.nu
    law = st.invgauss(1 / shape, scale=shape) if mixing == "nig" else st.gamma(shape, scale=1 / shape)
    assert_uniform_pit(field.mixing_variables(4000), law.cdf, what="mixing variables")
    x = TypeGField(prior, mixing, nu=0.05, seed=9).sample(N)
    v = TypeGField(prior, mixing, nu=0.05, seed=9).mixing_variables(N)  # the same streams: the draws' V_e
    f, w = _functional(prior, x)
    a = prior.apply_t(torch.from_numpy(w[prior.space.free])).double().numpy() ** 2
    assert_normal(f / np.sqrt(a @ v[field.element_of_column]), 0.0, 1.0)


def test_type_g_needs_white_noise():
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
    odd = ngsrng.Gaussian(ngs.H1(mesh, order=1), ngsrng.matern(4.0, 3))
    with pytest.raises(NotImplementedError):
        TypeGField(odd)


def test_representations(prior, points):
    field = named_marginal(points, "lognormal", 0.0, 0.5, representation="projection")
    assert field.sample(4).shape == (prior.space.ndof, 4)
    gf = named_marginal(points, "lognormal").sample()
    assert gf.space.type == "irspace"
    gf2 = field.sample()
    assert isinstance(gf2, ngs.GridFunction) and gf2.space is prior.space.fes
    raw = transformed(points, np.exp).sample(N)  # exp(g) without standardisation: lognormal at every point
    for q in _probe(raw):
        assert_law(raw[q], st.lognorm(points.std[q], scale=np.exp(points.mean[q])).cdf, what=f"point {q}")


def test_torch_path_matches_numpy(prior):
    """The device code path (forced on CPU tensors) gives the same values as the NumPy path for the same draws."""
    g = ngsrng.Gaussian(prior.space.fes, ngsrng.matern(4.0, 2, scale=10.0), dtype="float64", seed=0,
                        method={"engine": "torch"})
    for make in (lambda p: named_marginal(p, "gamma", 2.0, method="table"),
                 lambda p: named_marginal(p, "laplace"),
                 lambda p: WishartField(p, 3, np.eye(2), construction="bartlett", seed=0, method="table"),
                 lambda p: MatrixLogNormalField(p, 2, 0.5)):
        outs = []
        for device_path in (False, True):
            g.generator.manual_seed(11)
            pts = GaussianAtPoints(g, std=GaussianAtPoints(prior).std)
            pts.device_capable = device_path
            field = make(pts)  # a fresh own stream (seed=0) where the field has randomness of its own
            v = field.sample(5)
            outs.append(v.numpy() if hasattr(v, "numpy") and not isinstance(v, np.ndarray) else v)
        np.testing.assert_allclose(outs[1], outs[0], rtol=1e-8, atol=1e-10)
