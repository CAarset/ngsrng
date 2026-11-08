"""ObservedGaussian: exact draws of finitely many linear observations y = G f + e of a random function."""

import warnings

import ngsolve as ngs
import numpy as np
import pytest
import torch

import ngsrng
from ngsrng.reference import exact_covariance


@pytest.fixture(scope="module")
def fes():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2, dirichlet="left")


@pytest.fixture(scope="module")
def prior(fes):
    return ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", seed=0)


PTS = np.random.default_rng(0).uniform(0.05, 0.95, (12, 2))


def mc_close(y, cov, tol=0.08):
    """Empirical covariance of draws y (m x n) against cov, relative to its largest entry."""
    return np.abs(np.cov(y) - cov).max() < tol * np.abs(cov).max()


@pytest.mark.parametrize("method", ["factor", "pathwise"])
def test_law(prior, method):
    k = ngsrng.functional_covariance(prior, points=PTS)
    o = ngsrng.ObservedGaussian(prior, points=PTS, method=method, scale=2.0, loc=1.0)
    assert o.method == method and o.m == 12
    np.testing.assert_allclose(o.covariance(), 4 * k, rtol=1e-9, atol=1e-12 * np.abs(k).max())
    torch.manual_seed(0)
    y = o.sample(8000).numpy()
    assert mc_close(y, 4 * k) and np.abs(y.mean(1) - 1.0).max() < 0.1 * np.sqrt(4 * np.diag(k).max())


def test_pathwise_is_the_field_observed(prior):
    """Pathwise draws are G applied to the field's own draws (same randomness)."""
    g = ngsrng.Gaussian(prior.space.fes, ngsrng.matern(5.0, 2), dtype="float64")
    o = ngsrng.ObservedGaussian(g, points=PTS, method="pathwise")
    torch.manual_seed(1)
    a = o.sample(5).numpy()
    torch.manual_seed(1)
    c = g.sample(5).numpy()
    np.testing.assert_allclose(a, o.G @ c, rtol=0, atol=1e-12 * np.abs(a).max())


def test_factor_cutoff(prior):
    """||K - F F^T||_2 <= rtol max eigenvalue, and the dropped part is reported."""
    lam = np.linalg.eigvalsh(ngsrng.functional_covariance(prior, points=PTS))
    rtol = 3 * lam[1] / lam[-1]  # drops at least the two smallest directions
    o = ngsrng.ObservedGaussian(prior, points=PTS, rtol=rtol)
    k, f = o.covariance(), o._F64.cpu().numpy()
    assert o.rank == int((lam > rtol * lam.max()).sum()) <= 10
    assert np.linalg.norm(k - f @ f.T, 2) <= rtol * lam.max() * (1 + 1e-9)
    assert 0 < o.plan["dropped_trace"] < 12 * rtol


@pytest.mark.parametrize("method", ["factor", "pathwise"])
def test_noise(prior, method):
    """White, coloured and field-generated noise (an independent ObservedGaussian)."""
    fes = prior.space.fes
    k = ngsrng.functional_covariance(prior, points=PTS)
    gamma = 0.01 * (np.eye(12) + 0.5 * np.ones((12, 12)))
    other = ngsrng.ObservedGaussian(ngsrng.Gaussian(fes, ngsrng.matern(2.0, 2), dtype="float64"), points=PTS,
                                    scale=0.1)
    for noise, want in ((0.01, k + 0.01 * np.eye(12)), (gamma, k + gamma), (other, k + other.covariance())):
        o = ngsrng.ObservedGaussian(prior, points=PTS, noise=noise, method=method)
        np.testing.assert_allclose(o.covariance(), want, rtol=1e-9, atol=1e-12 * np.abs(want).max())
        torch.manual_seed(2)
        assert mc_close(o.sample(8000).numpy(), want)


def test_fields_and_operators(prior, fes):
    """Posteriors (and the predictive laws), low-rank and Matheron samplers, apply-only covariances (factor only);
    point values, local averages, LinearForms and dense matrices."""
    obs = ngsrng.LinearObservations(prior, points=PTS, noise=0.01)
    g = obs.G_free.toarray()
    k = g @ exact_covariance(prior) @ g.T
    np.testing.assert_allclose(obs.predictive().covariance(), k + 0.01 * np.eye(12), rtol=1e-9)
    post = obs.posterior(data=np.zeros(12))
    for field in (post, ngsrng.MatheronPosterior(obs, data=np.zeros(12)), ngsrng.karhunen_loeve(prior, 8)):
        want = g @ exact_covariance(field) @ g.T
        for method in ("factor", "pathwise"):
            o = ngsrng.ObservedGaussian(field, points=PTS, method=method)
            np.testing.assert_allclose(o.covariance(), want, rtol=1e-8, atol=1e-12 * np.abs(want).max())
            torch.manual_seed(3)
            assert mc_close(o.sample(8000).numpy(), want), (type(field).__name__, method)
    cov = ngsrng.OperatorCovariance(lambda y: torch.from_numpy(exact_covariance(prior) @ y.numpy()))
    oc = ngsrng.ObservedGaussian(cov, fes=fes, points=PTS)
    assert oc.method == "factor"
    np.testing.assert_allclose(oc.covariance(), k, rtol=1e-9)
    with pytest.raises(ValueError, match="sampler"):
        ngsrng.ObservedGaussian(cov, fes=fes, points=PTS, method="pathwise")
    u, v = fes.TnT()
    flux = ngs.LinearForm(v * ngs.ds("right")).Assemble()
    dense = np.random.default_rng(1).standard_normal((5, fes.ndof))
    for kw in ({"averages": (PTS[:4], 0.1)}, {"functionals": [flux]}, {"operator": dense}):
        gk = ngsrng.observation_operator(fes, **{("matrix" if key == "operator" else key): v for key, v in kw.items()})
        gk = gk[:, prior.space.free]
        gk = gk.toarray() if hasattr(gk, "toarray") else gk
        want = gk @ exact_covariance(prior) @ gk.T
        for method in ("factor", "pathwise"):
            o = ngsrng.ObservedGaussian(prior, method=method, **kw)
            np.testing.assert_allclose(o.covariance(), want, rtol=1e-8, atol=1e-12 * np.abs(want).max())
            torch.manual_seed(4)
            assert mc_close(o.sample(8000).numpy(), want), (list(kw), method)


def test_seeding_and_chunks(prior):
    """A torch.Generator reseeded per epoch reproduces the draws, without a new set-up; on the CPU the draws do not
    depend on how an in-place fill is split into chunks."""
    gen = torch.Generator().manual_seed(9)
    o = ngsrng.ObservedGaussian(prior, points=PTS, seed=gen)
    a = o.sample(4)
    gen.manual_seed(9)
    assert torch.equal(o.sample(4), a)
    for method in ("factor", "pathwise"):
        x = ngsrng.ObservedGaussian(prior, points=PTS, seed=5, method=method)
        y = ngsrng.ObservedGaussian(prior, points=PTS, seed=5, method=method)
        y.work_values = 30  # chunks of a few samples
        if method == "pathwise":
            prior.stream = ngsrng.backend.NormalStream(11)
        ref = x.sample(9).T.numpy().copy()
        if method == "pathwise":
            prior.stream = ngsrng.backend.NormalStream(11)
        buf = np.empty((9, 12))
        y.sample_into(buf)
        np.testing.assert_allclose(buf, ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def test_generator_family(fes):
    """rng.observed: (size, m) values in the generator's dtype, out= buffers, and the factor shared with spawned
    children (set up once)."""
    rng = ngsrng.default_rng(fes, seed=4, correlation=0.3, dtype="float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = rng.observed(0.0, 2.0, points=PTS, size=(3, 2))
        assert y.shape == (3, 2, 12) and y.dtype == np.float32
        buf = np.empty((7, 12), np.float32)
        assert rng.observed(0.0, 2.0, points=PTS, out=buf) is buf
        child = rng.spawn(1)[0]
        child.observed(0.0, 2.0, points=PTS, size=2)
    (_, factor), = rng._factors.values()
    (_, child_factor), = child._factors.values()
    assert child_factor is factor
    d = rng.distribution("observed", 0.0, 2.0, points=PTS, noise=0.1)
    assert d.sample(5).shape == (5, 12) and "y = G f + e" in d.info["law"]


def test_normal_at_points_on_the_torch_engine(fes):
    """normal(points=) with the torch engine evaluates on the Gaussian's device and is G applied to its draws."""
    rng = ngsrng.default_rng(fes, correlation=0.3, dtype="float64", method={"engine": "torch"})
    torch.manual_seed(6)
    a = rng.normal(0, 2.0, points=PTS, size=3)
    torch.manual_seed(6)
    c = rng.normal(0, 2.0, size=3)
    g = ngsrng.observation_operator(fes, points=PTS)
    np.testing.assert_allclose(np.asarray(a), (g @ np.asarray(c).T).T, rtol=0, atol=1e-10 * np.abs(a).max())
