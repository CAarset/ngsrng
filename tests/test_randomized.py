"""Randomised subspace iteration (ngsrng.randomized) and its uses in the KL expansion and in posteriors."""

import ngsolve as ngs
import numpy as np
import pytest
import torch

import ngsrng
from ngsrng.randomized import gram_orth, randomized_qr, randomized_svd
from ngsrng.reference import exact_covariance


def _operator(n, m, s, seed=0):
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.standard_normal((n, len(s))))
    v, _ = np.linalg.qr(rng.standard_normal((m, len(s))))
    return (u * s) @ v.T


def test_svd_recovers_a_decaying_spectrum():
    s = 0.7 ** np.arange(60)
    a = _operator(300, 200, s)
    u, sv, v = randomized_svd(lambda x: a @ x, lambda y: a.T @ y, 200, 20, power_iterations=2, oversampling=10)
    np.testing.assert_allclose(sv, s[:20], rtol=1e-6)
    np.testing.assert_allclose(u.T @ u, np.eye(20), atol=1e-12)
    np.testing.assert_allclose(v.T @ v, np.eye(20), atol=1e-12)


def test_subspace_iteration_keeps_small_singular_values():
    """Over 12 orders of magnitude: re-orthonormalising after every application keeps the small singular values;
    applying (A A^T)^q before orthonormalising loses everything below ~eps^(1/(2q+1)) to rounding."""
    s = np.logspace(0, -12, 40)
    a = _operator(400, 300, s, seed=1)
    errors = {}
    for method in ("subspace", "power"):
        _, sv, _ = randomized_svd(lambda x: a @ x, lambda y: a.T @ y, 300, 30, power_iterations=4, oversampling=10,
                                  method=method)
        errors[method] = np.max(np.abs(sv[:30] - s[:30]) / s[:30])
    assert errors["subspace"] < 1e-6 < errors["power"]


def test_weighted_output_space_and_qr_form():
    rng = np.random.default_rng(2)
    n, m = 150, 90
    g = rng.standard_normal((n, n))
    gram = g @ g.T / n + np.eye(n)  # an SPD Gram matrix (like a mass matrix)
    a = _operator(n, m, 0.8 ** np.arange(40), seed=3)
    # A: R^m -> (R^n, <.,.>_G); adjoint A* = A^T G
    u, sv, v = randomized_svd(lambda x: a @ x, lambda y: a.T @ (gram @ y), m, 15, power_iterations=2,
                              orth=gram_orth(gram))
    np.testing.assert_allclose(u.T @ gram @ u, np.eye(15), atol=1e-10)
    # singular values of A as a map into the weighted space = those of G^1/2 A
    w = np.linalg.cholesky(gram).T
    np.testing.assert_allclose(sv, np.linalg.svd(w @ a, compute_uv=False)[:15], rtol=1e-6)
    q, r = randomized_qr(lambda x: a @ x, lambda y: a.T @ y, m, 40, power_iterations=1)
    np.testing.assert_allclose(q @ r, a, atol=1e-10)
    np.testing.assert_allclose(r, np.triu(r), atol=1e-12)


@pytest.fixture(scope="module")
def prior():
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.12))
    fes = ngs.H1(mesh, order=2)
    return ngsrng.Gaussian(fes, ngsrng.matern(8.0, 2), dtype="float64", method={"solver": "direct"}, seed=0)


def test_karhunen_loeve_methods_agree_with_the_exact_spectrum(prior):
    s = prior.space
    m = s.mass_free.toarray()
    w = np.linalg.cholesky(m)
    exact = np.sort(np.linalg.eigvalsh(w.T @ exact_covariance(prior) @ w))[::-1]
    for method in ("subspace", "covariance"):
        kl = ngsrng.karhunen_loeve(prior, rank=25, power_iterations=3, method=method)
        np.testing.assert_allclose(kl.eigenvalues[:20], exact[:20], rtol=1e-3)
        modes = kl.modes[s.free]
        np.testing.assert_allclose(modes.T @ m @ modes, np.eye(kl.nrand), atol=1e-9)


def test_karhunen_loeve_application_count(prior):
    """Subspace iteration costs 2q + 2 block applications of L or L^T."""
    calls = {"apply": 0, "apply_t": 0}

    class Counting:
        def __init__(self, g):
            self.g, self.space, self.nrand, self.mean = g, g.space, g.nrand, g.mean

        def apply(self, x):
            calls["apply"] += 1
            return self.g.apply(x)

        def apply_t(self, y):
            calls["apply_t"] += 1
            return self.g.apply_t(y)

    ngsrng.karhunen_loeve(Counting(prior), rank=10, power_iterations=3)
    assert calls == {"apply": 4, "apply_t": 4}


def test_randomized_posterior_methods(prior):
    pts = np.random.default_rng(4).uniform(0.1, 0.9, (40, 2))
    exact = ngsrng.LinearObservations(prior, points=pts, noise=1e-3)
    for method in ("subspace", "power"):
        approx = ngsrng.LinearObservations(prior, points=pts, noise=1e-3, rank=30, power_iterations=1,
                                           low_rank_method=method)
        np.testing.assert_allclose(approx.sv[:10], exact.sv[:10], rtol=1e-6)
    post = approx.posterior(data=np.zeros(40), seed=1)
    assert torch.isfinite(post.sample(3)).all()
