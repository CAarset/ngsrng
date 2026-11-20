"""Random correlation parameters (ngsrng.parameters, ngsrng.spectral): the spectral modes are each construction's
exact covariance, the spectral law is the exact law at every parameter value (and standardises every draw exactly),
the parameters follow their law, both entry points give the same law, finite laws give the finite mixture, storage
round-trips (and the NumPy reference reader agrees), and what has no spectral law is refused clearly."""

import ngsolve as ngs
import numpy as np
import pytest
import scipy.stats as st
import torch
from conftest import cuda_usable
from stat_checks import assert_law
from test_dense import reference

import ngsrng
from ngsrng.spectral import log_weights

LAW = st.loguniform(0.3, 0.6)


@pytest.fixture(scope="module")
def space():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2)


def corr(boundary="torus", length=LAW, nu=2.0, **kw):
    return ngsrng.matern_correlation(length=length, nu=nu, boundary=boundary, **kw)


def affine_cov(sampler, length):
    """Covariance of a Gaussian (affine) dense sampler's outputs at one parameter value, from its map z -> y."""
    r = sampler.rank
    y = sampler.from_noise(np.vstack([np.zeros(r), np.eye(r)]), parameters={"length": np.full(r + 1, length)})
    d = y[1:] - y[0]
    return d.T @ d


@pytest.mark.parametrize("make", ["torus", "neumann", "dirichlet", "robin", "dg"])
def test_spectral_modes_are_the_exact_covariance(space, make):
    """Phi diag(c^2 (kappa^2 + lambda)^-alpha) Phi^T is the coefficient covariance the pipeline samples."""
    mesh = space.mesh
    if make == "torus":
        g = ngsrng.StationaryMatern(space, 0.4, 2.0, construction="torus", seed=0)
    else:
        fes = {"dirichlet": ngs.H1(mesh, order=2, dirichlet="left"),
               "dg": ngs.L2(mesh, order=1, dgjumps=True)}.get(make, space)
        kw = {"boundary": "robin", "beta": 3.0} if make == "robin" else {}
        g = ngsrng.default_rng(fes, correlation=ngsrng.matern_correlation(0.4, nu=2, **kw)).gaussian()
    modes = g.spectral_modes()
    eye = torch.eye(g.nrand, dtype=torch.float64)
    lf = g.apply(eye).numpy()
    s = np.exp(log_weights(modes.lam, np.array([0.4]), np.array([2.0]), 2))[0]
    rational = "rational" in str(g.plan)  # the pipeline's own approximation (odd power, coupled DG form): 1e-8
    np.testing.assert_allclose((modes.modes * s**2) @ modes.modes.T, lf @ lf.T, rtol=0,
                               atol=(1e-7 if rational else 1e-12) * np.abs(lf @ lf.T).max())


@pytest.mark.parametrize("boundary", ["torus", "neumann"])
@pytest.mark.parametrize("kw", [{}, {"representation": "quadrature"}])
def test_the_law_at_a_parameter_value_is_the_fields(space, boundary, kw):
    """At the largest length (whose torus box the spectral law uses), the spectral law's covariance is the field's
    exact one: a Gaussian field, and standardised point values (normal marginals)."""
    rng = ngsrng.default_rng(space, seed=0, dtype="float64")
    spec = corr(boundary)
    a = rng.distribution("normal", **kw, correlation=spec).dense(rtol=None)
    exact = rng.distribution("normal", **kw, correlation=spec.at(length=0.6)).dense(rtol=None)
    r = exact.rank
    x = exact.from_noise(np.vstack([np.zeros(r), np.eye(r)]))
    c = (x[1:] - x[0]).T @ (x[1:] - x[0])
    np.testing.assert_allclose(affine_cov(a, 0.6), c, rtol=0, atol=1e-12 * np.abs(c).max())


@pytest.mark.parametrize("tol", [None, 1e-3])
def test_every_draw_is_standardised_exactly(space, tol):
    """Standardised families have unit latent variance at every point for every parameter value, also truncated."""
    rng = ngsrng.default_rng(space, seed=0, dtype="float64")
    a = rng.distribution("normal", representation="quadrature", correlation=corr()).dense(tol=tol)
    for length in (0.3, 0.41, 0.6):
        np.testing.assert_allclose(np.diag(affine_cov(a, length)), 1.0, rtol=1e-10)


def test_parameters_follow_their_law_and_draws_split(space):
    """The drawn lengths follow the law (exact-bound KS); a batch split in two draws the same values and lengths;
    labels=True and parameters=True return them; the Bernstein range holds."""
    rng = ngsrng.default_rng(space, seed=3)
    pm = rng.distribution("uniform", -1.0, 2.0, representation="bernstein", correlation=corr())
    assert isinstance(pm, ngsrng.ParameterMixture)
    a = pm.dense(tol=1e-6)
    y, lab, th = a.sample(6000, labels=True, parameters=True)
    assert not lab.any() and th["length"].shape == (6000,)
    assert_law(th["length"], LAW.cdf, what="lengths")
    one, two = a.to(seed=9), a.to(seed=9)
    y1, t1 = one.sample(30, parameters=True)
    y2, t2 = two.sample(12, parameters=True)
    y3, t3 = two.sample(18, parameters=True)
    np.testing.assert_array_equal(np.concatenate([t2["length"], t3["length"]]), t1["length"])
    np.testing.assert_allclose(np.vstack([y2, y3]), y1, rtol=1e-12, atol=1e-12)
    from ngsrng.stationary import basis_at

    vals = basis_at(space, np.random.default_rng(1).uniform(0, 1, (500, 2))) @ y[:500].T
    assert vals.min() > -1.0 and vals.max() < 2.0
    x, th = pm.sample(4, parameters=True)
    assert x.shape == (4, space.ndof) and th["length"].shape == (4,)
    assert isinstance(pm.sample(), ngs.GridFunction)


def test_both_entry_points_give_the_same_law(space):
    """ngsrng.mixture(factory, over=law) is the factory with the law as its parameter: the same spectral law."""
    rng = ngsrng.default_rng(space, seed=4)

    def field(length):
        return rng.distribution("uniform", representation="bernstein", correlation=corr(length=length))

    a, b = field(LAW), ngsrng.mixture(field, over=LAW)
    assert type(a) is type(b) is ngsrng.ParameterMixture
    da, db = a.dense(), b.dense()
    np.testing.assert_array_equal(da.laws[0]["factor"], db.laws[0]["factor"])
    z = np.random.default_rng(0).standard_normal((5, da.rank))
    th = {"length": np.linspace(0.3, 0.6, 5)}
    np.testing.assert_array_equal(da.from_noise(z, parameters=th), db.from_noise(z, parameters=th))


def test_finite_laws_are_finite_mixtures(space):
    """A law with finitely many values: the finite mixture of the field at each value (any construction, also
    conditioned), weighted by the probabilities, through either entry point; parameters per draw; a continuous and a
    finite parameter: the finite mixture of continuous ones."""
    rng = ngsrng.default_rng(space, seed=5)
    values = st.rv_discrete(values=([0.3, 0.5], [0.25, 0.75]))
    spec = ngsrng.matern_correlation(length=values, nu=2, boundary="restrict")
    m = rng.distribution("uniform", representation="bernstein", correlation=spec)
    assert isinstance(m, ngsrng.Mixture) and len(m) == 2
    np.testing.assert_allclose(m.weights, [0.25, 0.75])
    x, lab, th = m.sample(40, labels=True, parameters=True)
    np.testing.assert_array_equal(th["length"], np.where(lab == 0, 0.3, 0.5))
    via = ngsrng.mixture(lambda length: rng.distribution("uniform", representation="bernstein",
                                                         correlation=spec.at(length=length)), over=values)
    assert [d.field for d in via.laws] == [d.field for d in m.laws] and via.values == m.values
    _, th = m.dense().sample(10, parameters=True)
    assert set(th["length"]) <= {0.3, 0.5}
    both = rng.distribution("uniform", representation="bernstein",
                            correlation=corr(nu=st.rv_discrete(values=([1.5, 2.5], [0.5, 0.5]))))
    assert isinstance(both, ngsrng.Mixture) and all(isinstance(d, ngsrng.ParameterMixture) for d in both.laws)
    _, th = both.dense().sample(20, parameters=True)
    assert set(th["nu"]) <= {1.5, 2.5} and (th["length"] >= 0.3).all() and (th["length"] <= 0.6).all()


def test_storage_and_reference_reader(space, tmp_path):
    """A spectral law round-trips bit for bit; the NumPy reference reader (every column) gives from_noise's values."""
    rng = ngsrng.default_rng(space, seed=6, dtype="float64")
    a = rng.distribution("lognormal", 0.0, 0.5, representation="bernstein", correlation=corr()).dense(rtol=None)
    path = tmp_path / "spectral.npz"
    a.save(path)
    b = ngsrng.DenseSampler.load(path, seed=2)
    np.testing.assert_array_equal(b.sample(10), a.to(seed=2).sample(10))
    z = np.random.default_rng(0).standard_normal((20, 1, a.rank))
    th = {"length": np.random.default_rng(1).uniform(0.3, 0.6, 20)}
    y = a.from_noise(z, parameters=th)
    np.testing.assert_allclose(reference(path, z, parameters=th)[0], y, rtol=0, atol=1e-9 * np.abs(y).max())


def test_refused_clearly(space):
    with pytest.raises(ValueError, match="one fixed"):
        corr("restrict")
    with pytest.raises(ValueError, match="bounded support"):
        corr(length=st.lognorm(0.5))
    with pytest.raises(ValueError, match="random parameters"):
        ngsrng.default_rng(space, correlation=corr()).gaussian()
    dirichlet = ngs.H1(space.mesh, order=2, dirichlet="left")
    with pytest.raises(NotImplementedError, match="spectral"):
        ngsrng.default_rng(dirichlet, seed=0).distribution("normal", correlation=corr()).dense()
    with pytest.raises(TypeError, match="may be random"):
        ngsrng.mixture(lambda length: ngsrng.default_rng(space, correlation=0.3).distribution("uniform"), over=LAW)


@pytest.mark.skipif(not cuda_usable(), reason="no usable CUDA device")
def test_gpu(space):
    """On a GPU: host noise gives the CPU's draws and parameters (to float32 rounding of the weights' products)."""
    rng = ngsrng.default_rng(space, seed=7, dtype="float64")
    a = rng.distribution("uniform", representation="bernstein", correlation=corr()).dense(tol=1e-6)
    y, th = a.to(device="cuda", seed=2, noise="host").sample(64, parameters=True)
    yc, tc = a.to(seed=2).sample(64, parameters=True)
    np.testing.assert_array_equal(th["length"].cpu().numpy(), tc["length"])
    np.testing.assert_allclose(y.cpu().numpy(), yc, rtol=0, atol=1e-10)
