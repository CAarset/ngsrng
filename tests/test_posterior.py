import ngsolve as ngs
import numpy as np
import pytest
import torch
from conftest import cuda_usable

import ngsrng
from ngsrng import LinearObservations, MatheronPosterior
from ngsrng.reference import covariance_errors, exact_covariance, sampler_covariance

ENGINES = [("ngsolve", "cpu", {"solver": "direct"})]
if cuda_usable():
    ENGINES.append(("torch", "cuda", {"solver": "torch-pcg"}))


def _prior(kind="H1", power=2, engine=ENGINES[0], order=2):
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25))
    if kind == "H1":
        fes = ngs.H1(mesh, order=order, dirichlet="left")
        spec = ngsrng.matern(4.0, power)
    else:
        fes = ngs.L2(mesh, order=order, dgjumps=True)
        spec = ngsrng.matern_dg(fes, 4.0, power)
    _, dev, method = engine
    return ngsrng.Gaussian(fes, spec, device=dev, dtype="float64", method=method, tol=1e-13, seed=0)


def _dense_posterior(prior, g, prec):
    s0 = exact_covariance(prior)
    return s0, np.linalg.inv(np.linalg.inv(s0) + g.T @ prec @ g)


def _obs(prior, m=10, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.95, (m, 2)), np.linspace(0.01, 0.1, m), rng.standard_normal(m)


@pytest.mark.parametrize("form", ["fe", "noise"])
@pytest.mark.parametrize("engine", ENGINES, ids=[e[0] for e in ENGINES])
@pytest.mark.parametrize("kind,power", [("H1", 2), ("H1", 3), ("L2", 2)])
def test_noise_space_posterior_is_exact(engine, kind, power, form):
    prior = _prior(kind, power, engine)
    pts, var, y = _obs(prior)
    obs = LinearObservations(prior, points=pts, noise=var)
    post = obs.posterior(data=y, form=form)
    g = obs.G[:, prior.space.free].toarray()
    prec = np.diag(1 / var)
    _, sp_ = _dense_posterior(prior, g, prec)
    tol = 1e-9 if engine[0] == "ngsolve" else 1e-6
    err = covariance_errors(sampler_covariance(post), sp_)
    assert err["rel_fro"] < tol, err
    mean_ref = sp_ @ (g.T @ prec @ (y - obs.G @ prior.mean))
    got = post.mean[prior.space.free] - prior.mean[prior.space.free]
    assert np.abs(got - mean_ref).max() < tol * np.abs(mean_ref).max()


def test_matheron_matches_dense():
    prior = _prior()
    pts, var, y = _obs(prior)
    obs = LinearObservations(prior, points=pts, noise=var)
    mat = MatheronPosterior(obs, data=y)
    g = obs.G[:, prior.space.free].toarray()
    _, sp_ = _dense_posterior(prior, g, np.diag(1 / var))
    assert covariance_errors(sampler_covariance(mat), sp_)["rel_fro"] < 1e-9
    post = obs.posterior(data=y)
    np.testing.assert_allclose(mat.mean, post.mean, atol=1e-10 * np.abs(post.mean).max())


def test_dense_noise_covariance():
    prior = _prior()
    pts, _, y = _obs(prior, m=6)
    a = np.random.default_rng(3).standard_normal((6, 6))
    gamma = 0.05 * (a @ a.T / 6 + np.eye(6))
    obs = LinearObservations(prior, points=pts, noise=gamma)
    g = obs.G[:, prior.space.free].toarray()
    _, sp_ = _dense_posterior(prior, g, np.linalg.inv(gamma))
    assert covariance_errors(sampler_covariance(obs.posterior()), sp_)["rel_fro"] < 1e-9


@pytest.mark.parametrize("forms", [("fe", "fe"), ("noise", "fe"), ("fe", "noise")])
def test_sequential_equals_joint_conditioning(forms):
    prior = _prior()
    pts, var, y = _obs(prior)
    joint = LinearObservations(prior, points=pts, noise=var).posterior(data=y)
    first = LinearObservations(prior, points=pts[:4], noise=var[:4]).posterior(data=y[:4], form=forms[0])
    second = LinearObservations(first, points=pts[4:], noise=var[4:]).posterior(data=y[4:], form=forms[1])
    assert covariance_errors(sampler_covariance(second), sampler_covariance(joint))["rel_fro"] < 1e-9
    np.testing.assert_allclose(second.mean, joint.mean, atol=1e-9 * np.abs(joint.mean).max())


def test_fe_and_noise_forms_give_identical_samples():
    prior = _prior()
    pts, var, y = _obs(prior)
    obs = LinearObservations(prior, points=pts, noise=var)
    a, b = obs.posterior(data=y, form="fe"), obs.posterior(data=y, form="noise")
    eta = torch.randn(prior.nrand, 5, dtype=torch.float64)
    torch.testing.assert_close(a.apply(eta), b.apply(eta), rtol=1e-9, atol=1e-12)
    y_ = torch.randn(prior.space.nfree, 3, dtype=torch.float64)
    torch.testing.assert_close(a.apply_t(y_), b.apply_t(y_), rtol=1e-9, atol=1e-12)


def test_downdate_reproduces_posterior():
    """C_post = C0 - B K^-1 B* with B = C0 G*, K = Gamma + G C0 G*, given directly, gives the same Gaussian."""
    prior = _prior()
    pts, var, _ = _obs(prior)
    obs = LinearObservations(prior, points=pts, noise=var)
    s0 = exact_covariance(prior)
    g = obs.G[:, prior.space.free].toarray()
    b = np.zeros((prior.space.ndof, len(var)))
    b[prior.space.free] = s0 @ g.T
    k = np.diag(var) + g @ s0 @ g.T
    post = prior.downdate(b, k)
    _, sp_ = _dense_posterior(prior, g, np.diag(1 / var))
    assert covariance_errors(sampler_covariance(post), sp_)["rel_fro"] < 1e-8


def test_downdate_rejects_indefinite():
    prior = _prior()
    b = np.zeros((prior.space.ndof, 1))
    b[prior.space.free[3], 0] = 1.0
    with pytest.raises(ValueError, match="not positive semidefinite"):
        prior.downdate(b, np.array([[1e-12]]))


def test_randomized_low_rank_converges():
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.12))
    prior = ngsrng.Gaussian(ngs.H1(mesh, order=1), ngsrng.matern(4.0), dtype="float64", method={"solver": "direct"})
    pts, var, _ = _obs(prior, m=40)
    var = 1e-4 * np.ones(40)
    ref = sampler_covariance(LinearObservations(prior, points=pts, noise=var).posterior())

    def err(r):
        obs = LinearObservations(prior, points=pts, noise=var, rank=r, oversampling=0, power_iterations=0)
        return covariance_errors(sampler_covariance(obs.posterior()), ref)["rel_fro"]

    errs = [err(r) for r in (4, 16, 40)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-9


def test_posterior_samples_monte_carlo():
    prior = _prior()
    pts, var, y = _obs(prior)
    obs = LinearObservations(prior, points=pts, noise=var)
    post = obs.posterior(data=y, seed=5)
    xs = post.sample(3000).double().numpy()
    g = obs.G.toarray()
    # observed values concentrate near the data, with the posterior predictive variance
    pred_var = np.einsum("ij,jk,ik->i", g[:, prior.space.free], sampler_covariance(post), g[:, prior.space.free])
    emp = (g @ xs).var(axis=1)
    assert np.all(np.abs(emp - pred_var) < 5 * pred_var * np.sqrt(2 / 2999))


def test_posterior_from_local_averages_is_exact():
    prior = _prior()
    centers = np.random.default_rng(6).uniform(0.15, 0.85, (8, 2))
    lf = ngs.LinearForm(prior.space.fes.TestFunction() * ngs.ds("top"))
    obs = LinearObservations(prior, averages=(centers, 0.1), functionals=[lf], noise=0.01)
    g = obs.G[:, prior.space.free].toarray()
    _, sp_ = _dense_posterior(prior, g, np.eye(obs.m) / 0.01)
    assert obs.m == 9
    assert covariance_errors(sampler_covariance(obs.posterior()), sp_)["rel_fro"] < 1e-9


@pytest.mark.parametrize("form", ["fe", "noise"])
def test_noise_scaling_equals_rescaled_noise(form):
    """posterior(noise_scaling=w) is the posterior with noise variances sigma^2 / w (w = 0: observation ignored)."""
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=1)
    prior = ngsrng.Gaussian(fes, ngsrng.matern(kappa=6.0, power=2), dtype=torch.float64, seed=0)
    pts = np.random.default_rng(0).uniform(0.1, 0.9, (8, 2))
    data = np.random.default_rng(1).standard_normal(8)
    w = np.array([1.0, 0.5, 2.0, 0.0, 1.0, 3.0, 0.1, 1.0])
    obs = ngsrng.LinearObservations(prior, points=pts, noise=0.04)
    scaled = obs.posterior(data=data, noise_scaling=w, form=form)
    keep = w > 0
    ref = ngsrng.LinearObservations(prior, points=pts[keep], noise=0.04 / w[keep]).posterior(data=data[keep])
    np.testing.assert_allclose(scaled.mean, ref.mean, atol=1e-10)
    np.testing.assert_allclose(sampler_covariance(scaled), sampler_covariance(ref), atol=1e-10)
    with pytest.raises(ValueError):
        obs.posterior(noise_scaling=-w)


def test_posterior_with_given_mean_keeps_the_covariance():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=1)
    prior = ngsrng.Gaussian(fes, ngsrng.matern(kappa=6.0, power=2), dtype=torch.float64, seed=0)
    obs = ngsrng.LinearObservations(prior, points=np.random.default_rng(0).uniform(0.1, 0.9, (6, 2)), noise=0.04)
    centre = np.random.default_rng(1).standard_normal(fes.ndof)
    post = obs.posterior(mean=centre, noise_scaling=np.full(6, 2.0))
    np.testing.assert_allclose(post.mean, centre)
    reference = obs.posterior(noise_scaling=np.full(6, 2.0))
    np.testing.assert_allclose(sampler_covariance(post), sampler_covariance(reference), atol=1e-12)
    with pytest.raises(ValueError):
        obs.posterior(data=np.zeros(6), mean=centre)


def test_exact_covariance_of_conditioned_gaussians():
    """reference.exact_covariance gives the exact covariance of the conditioned law (it used to return the prior's):
    for both update forms, with noise scaling, for a randomised factorisation (whose sampler only approximates it),
    downdates, Matheron and low-rank posteriors."""
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=2)
    prior = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", seed=0)
    s0 = exact_covariance(prior)
    noise = np.array([0.01, 0.02, 0.03])
    obs = ngsrng.LinearObservations(prior, points=[(0.3, 0.4), (0.6, 0.5), (0.5, 0.8)], noise=noise)
    g = obs.gf.toarray()

    def posterior_cov(weights):
        prec = np.linalg.inv(s0) + g.T @ ((weights / noise)[:, None] * g)
        return np.linalg.inv(prec)

    def close(a, b):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9 * np.abs(b).max())

    want = posterior_cov(np.ones(3))
    for post in (obs.posterior(form="fe"), obs.posterior(form="noise")):
        close(exact_covariance(post), want)
        close(sampler_covariance(post), want)
    w = np.array([1.0, 0.0, 2.0])  # observation 1 ignored, observation 2 twice as precise
    close(exact_covariance(obs.posterior(noise_scaling=w)), posterior_cov(w))
    rough = ngsrng.LinearObservations(prior, points=[(0.3, 0.4), (0.6, 0.5), (0.5, 0.8)], noise=noise, rank=1,
                                      oversampling=0)
    close(exact_covariance(rough.posterior()), want)  # the true posterior, not the rank-1 approximation
    assert np.abs(sampler_covariance(rough.posterior()) - want).max() > 1e-6 * np.abs(want).max()
    b = s0 @ g.T  # C0 - B K^-1 B* with these B, K is the posterior again
    close(exact_covariance(prior.downdate(b, np.diag(noise) + g @ s0 @ g.T)), want)
    close(exact_covariance(ngsrng.MatheronPosterior(obs)), want)
    kl = ngsrng.karhunen_loeve(prior, 8)
    w8 = kl.modes[prior.space.free] * np.sqrt(kl.eigenvalues)
    close(exact_covariance(kl), w8 @ w8.T)


@pytest.mark.parametrize("engine", ["ngsolve", "torch"])
def test_dense_observation_matrix_stays_dense(engine, monkeypatch):
    """A dense observation matrix (e.g. observations of a PDE solution) stays dense: it gives the posterior of the same
    matrix stored sparse, through both update forms and engines, and functional_covariance agrees; the in-place QR
    (used when memory is short) gives the same factorisation."""
    import scipy.sparse as sp

    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=2, dirichlet="left")
    prior = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", seed=0, method={"engine": engine})
    g = np.random.default_rng(0).standard_normal((7, fes.ndof))
    assert isinstance(ngsrng.observation_operator(fes, g), np.ndarray)
    with pytest.raises(ValueError, match="identically zero"):
        ngsrng.observation_operator(fes, np.vstack([g, np.zeros(fes.ndof)]))
    dense = LinearObservations(prior, g, noise=0.1)
    sparse = LinearObservations(prior, sp.csr_matrix(g) * 1.0, noise=0.1)
    assert isinstance(dense.gf, np.ndarray)  # (a full sparse matrix is densified too)
    want = exact_covariance(dense.posterior())
    np.testing.assert_allclose(exact_covariance(sparse.posterior()), want, atol=1e-12 * np.abs(want).max())
    for form in ("fe", "noise"):
        got = sampler_covariance(dense.posterior(form=form))
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9 * np.abs(want).max())
    if engine == "ngsolve":  # the NGSolve per-sample path of the FE-space update
        post = dense.posterior(data=np.zeros(7))
        x = post.sample(4000).numpy()
        free = prior.space.free
        emp = np.cov(x[free])
        assert np.abs(emp - want).max() < 0.15 * np.abs(want).max()
    fc = ngsrng.functional_covariance(prior, g[:3])
    np.testing.assert_allclose(fc, g[:3, prior.space.free] @ exact_covariance(prior) @ g[:3, prior.space.free].T,
                               rtol=1e-8)
    monkeypatch.setattr("ngsrng.posterior.available_memory", lambda: 1.0)  # forces the in-place QR ...
    monkeypatch.setattr("ngsrng.posterior.check_memory", lambda *args, **kwargs: None)  # ... past the memory check
    in_place = LinearObservations(prior, g, noise=0.1)
    np.testing.assert_allclose(in_place.sv, dense.sv, rtol=1e-10)
    np.testing.assert_allclose(exact_covariance(in_place.posterior()), want, atol=1e-12 * np.abs(want).max())


def test_factorisation_api():
    """The public face of the factorisation F^T = Q R (used by code built on it, e.g. experimental design)."""
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=2, dirichlet="left")
    prior = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", seed=0)
    pts = [(0.3, 0.4), (0.6, 0.5), (0.5, 0.8), (0.8, 0.2)]
    noise = np.array([0.01, 0.02, 0.03, 0.04])
    obs = LinearObservations(prior, points=pts, noise=noise)
    s0 = exact_covariance(prior)
    g = obs.G_free.toarray()
    gsg = g @ s0 @ g.T
    assert obs.exact and not obs.correlated_noise and obs.R.shape == (obs.rank, obs.m)
    np.testing.assert_allclose(obs.noise_std, np.sqrt(noise))
    np.testing.assert_allclose(obs.data_variance(), np.diag(gsg), rtol=1e-10)
    np.testing.assert_allclose(obs.R.T @ obs.R, gsg / np.sqrt(np.outer(noise, noise)), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(obs.singular_values, np.linalg.norm(obs.R, axis=1), rtol=1e-12)
    lq = obs.lq()  # Sigma_post = C0 - LQ (I - (I + R R^T)^-1) LQ^T
    mid = np.eye(obs.rank) - np.linalg.inv(np.eye(obs.rank) + obs.R @ obs.R.T)
    np.testing.assert_allclose(s0 - lq @ mid @ lq.T, exact_covariance(obs.posterior()), atol=1e-10 * np.abs(s0).max())
    rough = LinearObservations(prior, points=pts, noise=noise, rank=2, oversampling=0)
    assert not rough.exact
    np.testing.assert_allclose(rough.data_variance(), np.diag(gsg), rtol=1e-10)  # exact also then
    corr = 0.01 * (np.eye(4) + 0.3 * np.ones((4, 4)))
    obs_c = LinearObservations(prior, points=pts, noise=corr)
    assert obs_c.correlated_noise and obs_c.noise_std is None
    np.testing.assert_allclose(obs_c.data_variance(), np.diag(gsg), rtol=1e-10)
    est, err = ngsrng.xnystrace(lambda x: s0 @ x, s0.shape[0], 40, np.random.default_rng(0))
    assert abs(est - np.trace(s0)) < 5 * err + 1e-8 * np.trace(s0)
