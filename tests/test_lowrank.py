import ngsolve as ngs
import numpy as np
import pytest
import torch

import ngsrng
from ngsrng import LowRankGaussian
from ngsrng.reference import covariance_errors, exact_covariance, sampler_covariance


def _prior():
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25))
    fes = ngs.H1(mesh, order=2, dirichlet="left")
    return ngsrng.Gaussian(fes, ngsrng.matern(4.0), dtype="float64", method={"solver": "direct"}, seed=0)


def test_low_rank_gaussian():
    prior = _prior()
    fes = prior.space.fes
    rng = np.random.default_rng(4)
    b = np.zeros((fes.ndof, 5))
    b[prior.space.free] = rng.standard_normal((prior.space.nfree, 5))
    a = rng.standard_normal((5, 5))
    k = a @ a.T + 5 * np.eye(5)
    lr = LowRankGaussian(fes, b, k, d=(prior.space.mass @ b).T)
    bf = b[prior.space.free]
    assert covariance_errors(sampler_covariance(lr), bf @ np.linalg.solve(k, bf.T))["rel_fro"] < 1e-12
    assert lr.sample(3).shape == (fes.ndof, 3)
    with pytest.raises(ValueError):
        LowRankGaussian(fes, b, k, d=2 * b.T)


def test_karhunen_loeve_matches_exact_spectrum():
    import scipy.linalg as sla

    from ngsrng.lowrank import karhunen_loeve
    from ngsrng.reference import exact_covariance

    g = _prior()
    s = exact_covariance(g)
    m = g.space.mass_free.toarray()
    lam_exact = np.sort(sla.eigh(m @ s @ m, m, eigvals_only=True))[::-1]
    kl = karhunen_loeve(g, 10, power_iterations=5)
    phi = kl.modes[g.space.free]
    np.testing.assert_allclose(phi.T @ m @ phi, np.eye(kl.nrand), atol=1e-12)
    np.testing.assert_allclose(kl.eigenvalues, lam_exact[:10], rtol=1e-6)
    # truncation error in trace norm equals the discarded eigenvalues
    approx = sampler_covariance(kl)
    np.testing.assert_allclose(np.trace((s - approx) @ m), lam_exact[10:].sum(), rtol=1e-6)


def test_low_rank_conditioning_is_exact():
    from ngsrng.lowrank import karhunen_loeve

    g = _prior()
    kl = karhunen_loeve(g, 12)
    pts = np.random.default_rng(2).uniform(0.1, 0.9, (5, 2))
    y = np.random.default_rng(3).standard_normal(5)
    post = kl.condition(points=pts, noise=0.05, data=y)
    s0 = sampler_covariance(kl)
    gmat = ngsrng.observation_operator(g.space.fes, points=pts)[:, g.space.free].toarray()
    gs = gmat @ s0
    kd = 0.05 * np.eye(5) + gs @ gmat.T
    sp_ = s0 - gs.T @ np.linalg.solve(kd, gs)
    assert covariance_errors(sampler_covariance(post), sp_)["rel_fro"] < 1e-10
    mean_ref = gs.T @ np.linalg.solve(kd, y - gmat @ kl.mean[g.space.free])
    np.testing.assert_allclose(post.mean[g.space.free] - kl.mean[g.space.free], mean_ref, atol=1e-10)
    eta = np.random.default_rng(4).standard_normal((post.nrand, 3))
    lmat = post.apply(torch.from_numpy(np.eye(post.nrand))).numpy()
    np.testing.assert_allclose(post.apply_t(torch.from_numpy(lmat @ eta)).numpy(), lmat.T @ lmat @ eta)


@pytest.mark.parametrize("convention", ["dual->primal", "primal->primal"])
def test_operator_kl_reaches_trace_tolerance(convention):
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.15)), order=1)
    prior = ngsrng.Gaussian(fes, ngsrng.matern(4.0), dtype="float64", method={"solver": "direct"}, seed=0)
    sigma = exact_covariance(prior)
    m = prior.space.mass_free.toarray()
    op = sigma if convention == "dual->primal" else sigma @ m
    for tol in (1e-1, 1e-3):
        cov = ngsrng.OperatorCovariance(lambda y: torch.from_numpy(op @ y.numpy()), convention=convention, tol=tol)
        lr = ngsrng.operator_kl(fes, cov, seed=1)
        approx = lr.modes[prior.space.free] @ np.diag(lr.eigenvalues) @ lr.modes[prior.space.free].T
        true = np.trace((sigma - approx) @ m) / np.trace(sigma @ m)
        assert true < 2 * tol and abs(lr.trace_residual - true) < 0.5 * true + 1e-6
