"""Correlated random fields (ngsrng.correlated): every component keeps its law (marginals, conditioning, the Bernstein
range, bnd=), the latents are correlated by R, and draws follow the sampler conventions."""

import math

import ngsolve as ngs
import numpy as np
import pytest
import scipy.stats as st
import torch

import ngsrng
from ngsrng.boundary import boundary_dofs
from ngsrng.stationary import basis_at

CORR = ngsrng.matern_correlation(0.3, nu=2)
LEFT = np.column_stack([np.full(41, 1e-9), np.linspace(1e-9, 1 - 1e-9, 41)])  # the left edge, just inside


@pytest.fixture(scope="module")
def space():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.15)), order=2)


def dist(space, seed, name="uniform", *args, correlation=CORR, **kw):
    rng = ngsrng.default_rng(space, seed=seed, dtype="float64", correlation=correlation)
    args = args or ((-1, 20) if name == "uniform" else ())
    kw = {"representation": "bernstein"} | kw  # representation=None: the family's default (FE coefficients)
    return rng.distribution(name, *args, **{k: v for k, v in kw.items() if v is not None})


def centre_vertex(space):
    xyz = np.array([v.point for v in space.mesh.vertices])
    return int(np.argmin(np.linalg.norm(xyz - [0.5, 0.5], axis=1)))  # H1: vertex dof = vertex number


def test_identity_gives_the_components_own_draws(space):
    """R = I: each component is its own draw (built twice with the same seeds: equal to rounding)."""
    own = [dist(space, s).sample(5) for s in (1, 2)]
    joint = ngsrng.correlated([dist(space, 1), dist(space, 2)], 0.0).sample(5)
    for a, b in zip(joint, own):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-10 * np.abs(b).max())


@pytest.mark.parametrize("rho", [0.8, -0.5])
def test_components_keep_their_laws_and_correlate(space, rho):
    """Exact marginals (uniform and lognormal components; a vertex value of a Bernstein field has the exact
    marginal), latent correlation rho, and Spearman's (6 / pi) arcsin(rho / 2) for the values."""
    pair = ngsrng.correlated([dist(space, 3), dist(space, 4, "lognormal", 0.0, 0.5)], rho)
    g1, g2 = pair.sample(4000)
    i = centre_vertex(space)
    u, ln = g1[:, i], g2[:, i]
    assert st.kstest(u, st.uniform(-1, 21).cdf).pvalue > 1e-3
    assert st.kstest(ln, st.lognorm(0.5).cdf).pvalue > 1e-3
    z1, z2 = st.norm.ppf((u + 1) / 21), np.log(ln) / 0.5  # the latents, back from the values
    assert np.corrcoef(z1, z2)[0, 1] == pytest.approx(rho, abs=0.03)
    assert st.spearmanr(u, ln).statistic == pytest.approx(6 / math.pi * math.asin(rho / 2), abs=0.03)
    assert pair.info["cross_correlation"] == [[1.0, rho], [rho, 1.0]]


def test_conditioning_holds_per_component(space):
    """Posteriors of one prior at the same point and noise, conditioned on different data: each component keeps its
    own data (a Gaussian FE field equals it; a uniform field is its transform after standardising by the prior)."""
    prior = ngsrng.default_rng(space, seed=5, dtype="float64", correlation=CORR).gaussian()
    x = np.array([space.mesh.vertices[centre_vertex(space)].point])  # a vertex: Bernstein values are exact there
    posts = [prior.condition(points=x, noise=1e-10, data=[t]) for t in (-1.0, 1.5)]
    normals = ngsrng.correlated([dist(space, 6 + k, "normal", correlation=p, representation=None)
                                 for k, p in enumerate(posts)], 0.9).sample(50)
    for g, t in zip(normals, (-1.0, 1.5)):
        np.testing.assert_allclose(basis_at(space, x) @ g.T, t, atol=1e-3)
    sd = math.sqrt(ngsrng.pointwise_variance(prior)(space.mesh(*x[0])))
    uniforms = ngsrng.correlated([dist(space, 8 + k, correlation=p) for k, p in enumerate(posts)], 0.9).sample(50)
    for g, t in zip(uniforms, (-1.0, 1.5)):
        np.testing.assert_allclose(basis_at(space, x) @ g.T, -1 + 21 * st.norm.cdf(t / sd), atol=1e-3)


def test_range_and_boundary_restriction(space):
    """The Bernstein range guarantee and bnd= per component."""
    d = dist(space, 9, bnd="left")
    trace = boundary_dofs(space, "left", free_only=False)
    off = np.setdiff1d(np.arange(space.ndof), trace)
    for g in ngsrng.correlated([d, d], 0.9).sample(200):
        assert not g[:, off].any()
        vals = basis_at(space, LEFT) @ g.T
        assert -1.0 <= vals.min() and vals.max() <= 20.0
    inside = np.random.default_rng(0).uniform(0.02, 0.98, (200, 2))
    for g in ngsrng.correlated([dist(space, 10), dist(space, 11)], -0.7).sample(100):
        vals = basis_at(space, inside) @ g.T
        assert -1.0 <= vals.min() and vals.max() <= 20.0


def test_draw_conventions(space):
    """Shapes, single samples, torch output, in-place fills and chunking (seed None: torch's global state)."""
    pair = ngsrng.correlated([dist(space, None), dist(space, None)], 0.5)
    torch.manual_seed(1)
    ref = pair.sample((2, 3))
    assert all(r.shape == (2, 3, space.ndof) for r in ref)
    bufs = [np.empty((2, 3, space.ndof)) for _ in range(2)]
    torch.manual_seed(1)
    assert pair.sample((2, 3), out=bufs)[0] is bufs[0]
    for b, r in zip(bufs, ref):
        np.testing.assert_array_equal(b, r)
    for part in pair._parts:
        part.chunk = 1  # one sample per chunk: the same draw
    torch.manual_seed(1)
    for a, r in zip(pair.sample((2, 3)), ref):
        np.testing.assert_array_equal(a, r)
    assert all(isinstance(g, ngs.GridFunction) for g in pair.sample())
    assert all(isinstance(t, torch.Tensor) for t in pair.sample(2, output="torch"))
    same = ngsrng.correlated([dist(space, 12), dist(space, 12)], 1.0).sample(3)  # rho = 1: one field twice
    np.testing.assert_allclose(same[0], same[1], atol=1e-12)


def test_gaussian_fields(space):
    """Gaussian FE fields (FE coefficients) with their own loc and scale: correlation rho, laws unchanged."""
    a, b = ngsrng.correlated([dist(space, 13, "normal", representation=None),
                              dist(space, 14, "normal", 2.0, 3.0, representation=None)], 0.9).sample(3000)
    i = centre_vertex(space)
    assert a[:, i].std() == pytest.approx(1.0, rel=0.06) and b[:, i].std() == pytest.approx(3.0, rel=0.06)
    assert b[:, i].mean() == pytest.approx(2.0, abs=0.15)
    assert np.corrcoef(a[:, i], b[:, i])[0, 1] == pytest.approx(0.9, abs=0.02)


def test_errors(space):
    d = dist(space, 15)
    other = dist(space, 16, correlation=ngsrng.matern_correlation(0.15, nu=2))
    with pytest.raises(ValueError, match="differ in covariance"):
        ngsrng.correlated([d, other], 0.5)
    with pytest.raises(ValueError, match="different places"):
        ngsrng.correlated([dist(space, 17, "normal", representation=None), d], 0.5)
    with pytest.raises(NotImplementedError, match="several latents"):
        ngsrng.correlated([dist(space, 18, "chisquare", 3), d], 0.5)
    with pytest.raises(ValueError, match="same points"):
        ngsrng.correlated([dist(space, 19, representation="projection"), d], 0.5)
    with pytest.raises(ValueError, match="symmetric"):
        ngsrng.correlated([d, d], [[1, 0.5], [0.4, 1]])
    with pytest.raises(ValueError, match="semidefinite"):
        ngsrng.correlated([d, d, d], [[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
    with pytest.raises(ValueError, match="matrix"):
        ngsrng.correlated([d, d, d], 0.5)
    with pytest.raises(TypeError, match="rng.distribution"):
        ngsrng.correlated([d, "uniform"], 0.5)
