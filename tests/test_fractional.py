import ngsolve as ngs
import numpy as np
import pytest
import torch
from conftest import cuda_usable
from ngsolve.meshes import Make1DMesh

import ngsrng
from ngsrng import LinearObservations
from ngsrng.rational import fractional_power
from ngsrng.reference import covariance_errors, exact_covariance, sampler_covariance


@pytest.mark.parametrize("theta", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("span", [1e2, 1e6, 1e10])
def test_rational_power_structure_and_accuracy(theta, span):
    r = fractional_power(theta, 3.0, 3.0 * span, rtol=1e-7)
    assert r.rel_error <= 1e-7 and np.all(r.coeffs > 0) and np.all(r.shifts > 0) and r.c0 >= 0
    lam = np.geomspace(3.0, 3.0 * span, 5000)
    assert np.max(np.abs(r(lam) * lam**theta - 1)) <= 1e-7


def _space(kind):
    if kind == "1D":
        return ngs.H1(Make1DMesh(25), order=3)
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25))
    return ngs.H1(mesh, order=2, dirichlet="left") if kind == "H1" else ngs.L2(mesh, order=1, dgjumps=True)


def _spec(fes, beta):
    return ngsrng.matern_dg(fes, 3.0, beta) if fes.type == "l2ho" else ngsrng.matern(3.0, beta)


ENGINES = [("ngsolve", "cpu", {})]
if cuda_usable():
    ENGINES += [("torch", "cuda", {"solver": "cudss"})]


@pytest.mark.parametrize("engine", ENGINES, ids=[e[0] for e in ENGINES])
@pytest.mark.parametrize("kind", ["1D", "H1", "L2"])
@pytest.mark.parametrize("beta", [1.3, 2.5, 3.7])
def test_fractional_prior_is_exact(kind, beta, engine):
    fes = _space(kind)
    _, dev, method = engine
    g = ngsrng.Gaussian(fes, _spec(fes, beta), device=dev, dtype="float64", method=method)
    assert "rational" in g.plan["structure"]
    err = covariance_errors(sampler_covariance(g), exact_covariance(g))
    assert err["rel_fro"] < 1e-6 and err["max_rel_var"] < 1e-6, err


def test_fractional_adjoint_is_exact_transpose():
    fes = _space("H1")
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0, 2.6), dtype="float64")
    lmat = g.apply(torch.eye(g.nrand, dtype=torch.float64)).numpy()
    ltmat = g.apply_t(torch.eye(g.space.nfree, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(ltmat, lmat.T, atol=1e-12 * np.abs(lmat).max())


def test_integer_limit_matches_integer_path():
    fes = _space("H1")
    near = ngsrng.Gaussian(fes, ngsrng.matern(3.0, 2.0 + 1e-6), dtype="float64")
    exact = ngsrng.Gaussian(fes, ngsrng.matern(3.0, 2), dtype="float64")
    err = covariance_errors(sampler_covariance(near), exact_covariance(exact))
    assert err["rel_fro"] < 1e-4


def test_conditioning_a_fractional_prior():
    fes = _space("H1")
    prior = ngsrng.Gaussian(fes, ngsrng.matern(3.0, 1.7), dtype="float64")
    pts = np.random.default_rng(0).uniform(0.1, 0.9, (8, 2))
    obs = LinearObservations(prior, points=pts, noise=0.01)
    s0 = exact_covariance(prior)
    gmat = obs.G[:, prior.space.free].toarray()
    sp = s0 - (gmat @ s0).T @ np.linalg.solve(0.01 * np.eye(8) + gmat @ s0 @ gmat.T, gmat @ s0)
    assert covariance_errors(sampler_covariance(obs.posterior()), sp)["rel_fro"] < 1e-6
