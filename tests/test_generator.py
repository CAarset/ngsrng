"""The numpy-style Generator API."""

import ngsolve as ngs
import numpy as np
import pytest
import torch

import ngsrng
from ngsrng.reference import exact_covariance


@pytest.fixture(scope="module")
def fes():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=1)


@pytest.fixture(scope="module")
def rng(fes):
    return ngsrng.default_rng(fes, seed=3, correlation=0.3, dtype="float64")


def test_shapes_and_types(rng, fes):
    assert isinstance(rng.normal(), ngs.GridFunction)
    assert rng.normal(size=4).shape == (4, fes.ndof)
    assert rng.normal(0.0, 2.0, (2, 3)).shape == (2, 3, fes.ndof)  # positional size, as in numpy
    npts = rng.uniform(size=1).shape[1]
    assert npts == rng.points().shape[0]
    assert rng.gamma(2.0, size=(2, 2)).shape == (2, 2, npts)
    assert rng.dirichlet([1, 2, 3], size=5).shape == (5, npts, 3)
    assert rng.wishart(3, np.eye(2), size=2).shape == (2, npts, 2, 2)
    assert rng.wishart(3, np.eye(2)).shape == (npts, 2, 2)
    assert rng.poisson(2.0, size=3).dtype == np.int64
    assert rng.normal(size=2).dtype == np.float64
    assert isinstance(rng.normal(size=2, output="torch"), torch.Tensor)
    assert isinstance(rng.as_gridfunction(rng.uniform(size=1)[0]), ngs.GridFunction)
    assert isinstance(rng.as_gridfunction(rng.normal(size=1)[0]), ngs.GridFunction)


def test_aliases_and_names(rng):
    for name in ("normal", "gaussian", "Normal", "norm"):
        assert rng.sample(name, size=1).shape == rng.normal(size=1).shape
    assert rng.info("t", 4) == rng.info("StudentT", 4) == rng.info("student_t", 4)
    assert rng.info("chi2", 3)["law"].startswith("chi-square process")
    assert rng.info("chisquare", 2.5)["law"].startswith("Gaussian copula")
    assert rng.info("FisherSnedecor", 3, 4)["law"] == rng.info("f", 3, 4)["law"]
    assert rng.sample("genpareto", 0.1, size=2).shape[0] == 2  # scipy.stats fallback
    with pytest.raises(ValueError):
        rng.sample("no-such-distribution")
    with pytest.raises(TypeError):
        rng.gamma()


def test_reproducibility(fes):
    a = ngsrng.default_rng(fes, seed=5, correlation=0.3)
    b = ngsrng.default_rng(fes, seed=5, correlation=0.3)
    c = ngsrng.default_rng(fes, seed=6, correlation=0.3)
    for name, args in (("normal", ()), ("beta", (2.0, 3.0)), ("standard_t", (4,)), ("nig", (0.5,))):
        xa, xb, xc = (g.sample(name, *args, size=3) for g in (a, b, c))
        np.testing.assert_array_equal(xa, xb)
        assert not np.allclose(xa, xc)
    s1, s2 = a.spawn(2)
    assert not np.allclose(s1.normal(size=2), s2.normal(size=2))


def test_normal_location_and_scale(rng, fes):
    gauss = rng.gaussian()
    x = rng.normal(ngs.x, 3.0, size=4000)
    m = gauss.space.mass
    one = np.ones(fes.ndof)
    w = (one @ m)[gauss.space.free]
    expect_var = 9.0 * w @ exact_covariance(gauss) @ w
    f = x @ (m @ one)
    assert abs(f.mean() - 0.5) < 5 * np.sqrt(expect_var / 4000)
    assert abs(f.var() / expect_var - 1) < 5 * np.sqrt(2 / 4000)


def test_matern_correlation_has_unit_variance_inside():
    """Unit variance away from the boundary once the mesh resolves the correlation length (h ~ length / 8)."""
    fine = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.04)), order=2)
    rng = ngsrng.default_rng(fine, seed=1, correlation=ngsrng.matern_correlation(0.3, nu=1.0), dtype="float64")
    pts = rng._gaussian_at_points(rng.gaussian())
    inside = np.all(np.abs(rng.points() - 0.5) < 0.15, axis=1)
    assert np.all(np.abs(pts.std[inside] ** 2 - 1) < 0.1)


def test_spatially_varying_parameters(rng):
    x = rng.uniform(0.0, 1.0 + ngs.x, size=200)
    hi = 1.0 + rng.points()[:, 0]
    assert np.all(x >= 0) and np.all(x <= hi[None, :])
    assert np.all(x.max(0) > 0.7 * hi)


def test_white_noise_is_projected_white_noise(rng, fes):
    c = rng.white_noise(size=4000)
    m = rng.gaussian(precision=rng._mass_precision()).space.mass
    f = c @ (m @ np.ones(fes.ndof))  # <1, W_h> has variance |domain| = 1
    assert abs(f.var() - 1) < 5 * np.sqrt(2 / 4000)


def test_distribution_objects_are_cached(rng):
    d = rng.distribution("gamma", 2.0)
    assert d.field is rng.distribution("gamma", 2.0).field
    d.sample(50)
    assert d.info["inversion"] == "ppf"  # 50 samples x npoints < TABLE_MIN_VALUES: no table yet
    d.sample(500)
    assert d.info["inversion"] == "table"
    assert rng.distribution("wishart", 3, np.eye(2)).field is rng.distribution("wishart", 3, np.eye(2)).field


def test_other_spatial_structures(fes):
    prior = ngsrng.Gaussian(fes, ngsrng.matern(4.0, 2), dtype="float64", seed=0)
    post = prior.condition(points=[(0.5, 0.5)], noise=1e-6, data=[2.0])
    rng = ngsrng.default_rng(fes, seed=0, correlation=post, dtype="float64")
    obs = ngsrng.observation_operator(fes, points=[(0.5, 0.5)])
    assert abs((obs @ rng.normal(size=50).T).mean() - 2.0) < 0.05
    assert rng.lognormal(size=2).shape[0] == 2
    sigma = exact_covariance(prior)
    cov = ngsrng.OperatorCovariance(lambda y: torch.from_numpy(sigma @ y.numpy()), tol=1e-2)
    rng2 = ngsrng.default_rng(fes, seed=0, covariance=cov)
    assert "adaptive KL" in rng2.info("normal")["gaussian"]["structure"]
    with pytest.raises(ValueError):
        ngsrng.default_rng(fes).normal()


def test_pointwise_families_on_posteriors_use_the_prior(fes):
    """The posterior of the prior's uniform field: Phi((g_post - m0) / sigma0), so at an exactly observed point the
    value is Phi(y / sigma0), and far away the field stays (almost) uniform."""
    import scipy.stats as st

    prior = ngsrng.Gaussian(fes, ngsrng.matern_correlation(0.3).bind(fes), dtype="float64", seed=0)
    post = prior.condition(points=[(0.5, 0.5)], noise=1e-8, data=[1.5])
    rng = ngsrng.default_rng(fes, seed=1, correlation=post, dtype="float64")
    xy = np.array([[0.5, 0.5], [0.05, 0.05]])
    u = rng.uniform(size=1000, points=xy)
    sigma0 = rng._gaussian_at_points(post, points=xy).std[0]
    np.testing.assert_allclose(u[:, 0], st.norm.cdf(1.5 / sigma0), atol=1e-3)
    assert abs(u[:, 1].mean() - 0.5) < 0.05
    own = rng.uniform(size=1000, points=xy, standardize_by="own")
    assert abs(own[:, 0].mean() - 0.5) < 0.05  # the posterior's own copula: uniform everywhere
    with pytest.warns(UserWarning, match="independent copies"):
        rng.chisquare(3, size=2)
    with pytest.warns(UserWarning, match="elliptical"):
        rng.standard_t(4, size=2)


def test_values_at_arbitrary_points(rng, fes):
    import scipy.stats as st

    xy = np.random.default_rng(0).uniform(0.02, 0.98, (5, 2))
    pts = rng._gaussian_at_points(rng.gaussian(), points=xy)
    e = pts.eval_full[:, pts.gauss.space.free].toarray()
    np.testing.assert_allclose(pts.std, np.sqrt(np.einsum("ij,jk,ik->i", e, exact_covariance(pts.gauss), e)),
                               rtol=1e-10)
    g = rng.gamma(2.0, ngs.x + 1, size=2000, points=xy)  # spatially varying scale, evaluated at the points
    assert g.shape == (2000, 5)
    for i in range(5):
        assert st.kstest(g[:, i], st.gamma(2.0, scale=1 + xy[i, 0]).cdf).pvalue > 1e-4
    assert rng.normal(size=3, points=xy).shape == (3, 5)
    assert rng.uniform(points=xy).shape == (5,)


def test_bounded_values_stay_inside_in_the_output_precision(fes):
    for dtype in ("float32", "float64"):
        r = ngsrng.default_rng(fes, seed=3, correlation=0.3, dtype=dtype)
        for name, args in (("uniform", ()), ("beta", (2.0, 5.0)), ("power", (3.0,)), ("lognormal", (0.0, 3.0))):
            d = r.distribution(name, *args)
            n = d.field.points.npoints
            z = np.where(np.arange(n)[:, None] % 2 == 0, 40.0, -40.0)  # far beyond what Phi resolves
            d.field.points.standardized = lambda k, dev=False, z=z: np.repeat(z, k, axis=1)
            lo, hi = d.field.transform.support
            x = d.sample(2)
            # finite bounds are never reached; an unbounded side may still overflow to inf, as usual in floating point
            assert np.all(x > lo) and (np.isinf(hi) or np.all(x < hi)), (dtype, name)


def test_pointwise_families_refuse_gaussians_without_point_values():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=1)
    rng = ngsrng.default_rng(fes, correlation=ngsrng.matern(5.0, 1.0))  # nu = 0 in 2D
    with pytest.warns(UserWarning):
        rng.normal()  # the Gaussian itself is still available (with the trace-class warning)
    with pytest.raises(ValueError, match="no point values"):
        rng.uniform()


def test_dx_matches_the_quadrature_representation(rng, fes):
    """Quadrature-point fields must be integrated with their own rule; rng.dx() gives it (a plain dx segfaults)."""
    from ngsrng.quadrature import QuadratureRule

    a_q = rng.lognormal(0.0, 0.5)
    u, v = fes.TnT()
    a = ngs.BilinearForm(a_q * ngs.grad(u) * ngs.grad(v) * rng.dx()).Assemble()
    ref = ngs.BilinearForm(fes)
    ref += ngs.SymbolicBFI(a_q * ngs.grad(u) * ngs.grad(v), intrule=QuadratureRule(rng.gaussian().space).rule,
                           simd_evaluate=False)
    ref.Assemble()
    x = ngs.GridFunction(fes)
    x.Set(ngs.sin(3 * ngs.x) * ngs.y)
    y1, y2 = a.mat.CreateColVector(), ref.mat.CreateColVector()
    y1.data, y2.data = a.mat * x.vec, ref.mat * x.vec
    np.testing.assert_allclose(y1.FV().NumPy(), y2.FV().NumPy(), atol=1e-12)
    cube = ngs.H1(ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.6)), order=1)
    with pytest.warns(UserWarning, match="crashes in SIMD"):
        ngsrng.default_rng(cube, correlation=0.5).dx(order=6)
