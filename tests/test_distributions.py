"""Statistical checks (fixed seeds, 5-sigma-type tolerances) of the distribution families."""

import ngsolve as ngs
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.stats as st

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

N = 3000
P_MIN = 1e-5  # KS p-value threshold: ~100 KS tests per run, so 1e-3 would fail spuriously about every tenth run


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
        assert st.kstest(v[q], ref.cdf).pvalue > P_MIN


def test_scipy_marginal_and_discrete(points):
    v = marginal(points, st.genextreme(0.2)).sample(N)
    assert st.kstest(v[0], st.genextreme(0.2).cdf).pvalue > P_MIN
    k = named_marginal(points, "poisson", 3.0).sample(N)
    for q in _probe(k):
        counts = np.bincount(k[q].astype(int), minlength=12)[:12]
        expect = st.poisson(3.0).pmf(np.arange(12)) * N
        assert st.chisquare(counts[:9], expect[:9] * counts[:9].sum() / expect[:9].sum()).pvalue > P_MIN


def test_chi_square_and_indicator(points):
    c = ChiSquareField(points, 3).sample(N)
    assert abs(c.mean() - 3) < 0.05 and abs(c.var() - 6) < 0.3
    ind = indicator(points, 1.0).sample(N)
    assert abs(ind.mean() - st.norm.sf(1.0)) < 0.01


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
            assert st.kstest(x, st.chi2(df).cdf).pvalue > P_MIN


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
        assert st.kstest(x[0, k], st.beta(alpha[k], alpha.sum() - alpha[k]).cdf).pvalue > P_MIN


def _functional_variance(prior, x):
    m = prior.space.mass
    one = np.ones(prior.space.ndof)
    return (one @ m @ (np.asarray(x) - prior.mean[:, None])).var(), one @ m


def test_student_t_covariance(prior):
    x = StudentTField(prior, 8, seed=7).sample(6000)
    var, w = _functional_variance(prior, x)
    wf = w[prior.space.free]
    gauss_var = wf @ exact_covariance(prior) @ wf
    assert abs(var / gauss_var - 8 / 6) < 0.15


@pytest.mark.parametrize("mixing", ["nig", "gamma"])
def test_type_g_keeps_covariance_and_has_heavy_tails(prior, mixing):
    field = TypeGField(prior, mixing, nu=0.05, seed=8)
    v = field.mixing_variables(4000)
    assert abs(v.mean() - 1) < 0.01
    x = field.sample(N)
    var, w = _functional_variance(prior, x)
    wf = w[prior.space.free]
    assert abs(var / (wf @ exact_covariance(prior) @ wf) - 1) < 5 * np.sqrt(2 / N)
    pts = GaussianAtPoints(prior, std=None)
    # heavy tails: the median over points of the excess kurtosis (nig over 12 seeds: 0.79 +- 0.04; the Gaussian
    # prior: 0.00 +- 0.03). A single point's estimate (sd ~0.15) failed for a third of the seeds at threshold 0.3.
    assert np.median(st.kurtosis(pts.rule.values(np.asarray(x)), axis=1)) > 0.4


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
    raw = transformed(points, np.exp).sample(N)  # exp(g) without standardisation
    expect = np.exp(points.mean + points.std**2 / 2)
    assert np.all(np.abs(raw.mean(axis=1) / expect - 1) < 5 * np.sqrt((np.exp(points.std**2) - 1) / N))


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
