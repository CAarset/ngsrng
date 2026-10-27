"""Covariance operators as data: local covariances (pointwise variance k(x, x)), traces, covariance functions,
sample (Wishart) covariances, operator covariances and Karhunen-Loeve fields with non-Gaussian coefficients."""

import warnings

import ngsolve as ngs
import numpy as np
import pytest
import torch
from ngsolve.meshes import Make1DMesh, MakeStructured2DMesh, MakeStructured3DMesh

import ngsrng
from ngsrng.covariance import as_covariance, local_covariance, xnystrace


def _dense(g):
    """Sigma (free dofs) of a Gaussian-like object, densely."""
    l = g.apply(torch.eye(g.nrand, dtype=torch.float64)).double().numpy()
    return l @ l.T


def _setup(fes, length=0.3, seed=1):
    rng = ngsrng.default_rng(fes, seed=seed, correlation=ngsrng.matern_correlation(length, nu=1.0), dtype="float64")
    return rng, rng.gaussian()


def _points(fes, k=40, seed=0):
    d = fes.mesh.dim
    return np.random.default_rng(seed).uniform(0.03, 0.97, (k, d))


def _variance(fes, sigma, pts):
    s = ngsrng.SpaceInfo(fes)
    e = ngsrng.observation_operator(fes, points=pts)[:, s.free].toarray()
    return np.einsum("ij,jk,ik->i", e, sigma, e)


SPACES = {
    "1D-H1-p3": lambda: ngs.H1(Make1DMesh(15), order=3),
    "2D-H1-p2-dirichlet": lambda: ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2, dirichlet="left"),
    "2D-L2-p1": lambda: ngs.L2(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=1, dgjumps=True),
    "2D-H1-quads-p2": lambda: ngs.H1(MakeStructured2DMesh(quads=True, nx=6, ny=6), order=2),
    "3D-H1-hexes-p1": lambda: ngs.H1(MakeStructured3DMesh(hexes=True, nx=4, ny=4, nz=4), order=1),
}


@pytest.mark.parametrize("name", SPACES)
def test_exact_variance_and_trace(name):
    fes = SPACES[name]()
    if fes.type == "l2":
        rng = ngsrng.default_rng(fes, seed=1, precision=ngsrng.matern_dg(fes, kappa=8.0), dtype="float64")
        g = rng.gaussian()
    else:
        _, g = _setup(fes)
    sigma = _dense(g)
    pts = _points(fes)
    lc = local_covariance(g, "exact")
    e = ngsrng.observation_operator(fes, points=pts)[:, g.space.free]
    np.testing.assert_allclose(lc.values(e), _variance(fes, sigma, pts), rtol=1e-10)
    # the variance is an L2(order 2p) FE function exactly
    gf = lc.variance()
    vals = np.array([gf(fes.mesh(*p)) for p in pts])
    np.testing.assert_allclose(vals, _variance(fes, sigma, pts), rtol=1e-9)
    # tr(C) = tr(Sigma M) = int k(x, x) dx
    m = g.space.mass_free.toarray()
    assert abs(lc.trace()[0] / np.trace(sigma @ m) - 1) < 1e-12
    assert abs(ngs.Integrate(gf, fes.mesh, order=4 * fes.globalorder + 2) / np.trace(sigma @ m) - 1) < 1e-10


def test_posterior_variance_is_prior_plus_exact_update():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2)
    _, prior = _setup(fes)
    pts = _points(fes)
    obs_pts = _points(fes, 8, seed=3)
    e = ngsrng.observation_operator(fes, points=pts)[:, prior.space.free]
    posts = {
        "fe": prior.condition(points=obs_pts, noise=0.01),
        "noise": prior.condition(points=obs_pts, noise=0.01, form="noise"),
        "randomized": prior.condition(points=obs_pts, noise=0.01, rank=4),
        "chain": prior.condition(points=obs_pts[:4], noise=0.01).condition(points=obs_pts[4:], noise=0.02),
    }
    obs = ngsrng.LinearObservations(prior, points=obs_pts, noise=0.01, factorize=False)
    posts["matheron"] = ngsrng.MatheronPosterior(obs)
    for name, post in posts.items():
        lc = local_covariance(post)
        assert "update" in lc.method, name
        ref = _variance(fes, _dense(post), pts)
        np.testing.assert_allclose(lc.values(e), ref, rtol=1e-8, atol=1e-12 * ref.max(), err_msg=name)
    # the prior's local covariance is computed once and shared (the posteriors' own caches are separate)
    assert local_covariance(prior) is local_covariance(prior, "exact")
    assert local_covariance(posts["fe"]) is not local_covariance(prior)


def test_lowrank_and_monte_carlo_estimates_are_unbiased_with_valid_errors():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.12)), order=1)
    _, g = _setup(fes, length=0.4)
    pts = _points(fes)
    e = ngsrng.observation_operator(fes, points=pts)[:, g.space.free]
    ref = local_covariance(g, "exact").values(e)
    for method, kw in (("lowrank", dict(rank=40, samples=48)), ("mc", dict(samples=400))):
        v, se = local_covariance(g, method, **kw).values(e, stderr=True)
        z = (v - ref) / se
        assert np.abs(z).max() < 5 and abs(z.mean()) < 1.0, method
    # with a basis spanning everything, the low-rank part is exact and the residual vanishes
    small = ngs.H1(Make1DMesh(10), order=1)
    _, g1 = _setup(small)
    lc = local_covariance(g1, "lowrank", rank=g1.space.nfree, samples=8)
    e1 = ngsrng.observation_operator(small, points=_points(small))[:, g1.space.free]
    np.testing.assert_allclose(lc.values(e1), local_covariance(g1, "exact").values(e1), rtol=1e-7)


def test_operator_covariances_conventions_and_forms():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2, dirichlet="bottom")
    u, v = fes.TnT()
    a = ngs.BilinearForm(ngs.grad(u) * ngs.grad(v) * ngs.dx + 10 * u * v * ngs.dx).Assemble()
    mass = ngs.BilinearForm(u * v * ngs.dx).Assemble()
    ainv = a.mat.Inverse(fes.FreeDofs())
    op = ainv @ mass.mat @ ainv  # C = A^-2: Matern-type, as an NGSolve operator expression
    src = as_covariance(op, fes=fes)
    n = src.space.nfree
    sigma = src.sigma(np.eye(n))
    m = src.space.mass_free.toarray()
    y = np.random.default_rng(0).standard_normal((n, 3))
    for conv, mat in (("primal->primal", sigma @ m), ("primal->dual", m @ sigma @ m)):
        spec = ngsrng.OperatorCovariance(lambda x, mat=mat: torch.from_numpy(mat @ x.numpy()), convention=conv)
        np.testing.assert_allclose(as_covariance(spec, fes=fes).sigma(y), sigma @ y, rtol=1e-8, atol=1e-14)
    with pytest.warns(UserWarning, match="not trace class|no local operator"):
        ngsrng.OperatorCovariance(mass)
    pts = _points(fes)
    e = ngsrng.observation_operator(fes, points=pts)[:, src.space.free]
    np.testing.assert_allclose(local_covariance(op, "exact", fes=fes).values(e), _variance(fes, sigma, pts),
                               rtol=1e-10)
    # apply-only low-rank estimate: exact part plus a white-noise cross estimator of the residual
    v_lr, se = local_covariance(op, "lowrank", fes=fes, rank=40, samples=64).values(e, stderr=True)
    assert np.abs((v_lr - _variance(fes, sigma, pts)) / se).max() < 5
    with pytest.raises(ValueError, match="sampler"):
        local_covariance(op, "mc", fes=fes)
    # the Generator accepts the operator directly (and samples its adaptive KL expansion)
    rng = ngsrng.default_rng(fes, seed=2, covariance=op)
    assert "apply-only covariance" in rng.info("normal")["gaussian"]["structure"]
    assert rng.normal(size=2).shape == (2, fes.ndof)


def test_trace_estimators():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.12)), order=1)
    _, g = _setup(fes, length=0.4)
    exact = ngsrng.covariance_trace(g, "exact")
    assert exact.stderr == 0 and exact.applications == 2 * g.space.nfree
    for method in ("hutchinson", "hutch++", "xnystrace"):
        est = ngsrng.covariance_trace(g, method, probes=60, seed=1)
        assert abs(est.value - exact.value) < 5 * est.stderr + 1e-3 * exact.value, method
    est = ngsrng.covariance_trace(g, "variance", variance_method="lowrank", rank=30, samples=30)
    assert abs(est.value - exact.value) < 5 * est.stderr
    # finite rank: exact and free
    kl = ngsrng.karhunen_loeve(g, 20)
    assert abs(ngsrng.covariance_trace(kl).value / kl.eigenvalues.sum() - 1) < 1e-12


def test_xnystrace_is_the_leave_one_out_average():
    a = np.random.default_rng(3).standard_normal((50, 50))
    a = a @ a.T / 50 + np.diag(np.linspace(0, 3, 50))
    est, _ = xnystrace(lambda x: a @ x, 50, 12, np.random.default_rng(5))
    om = np.random.default_rng(5).standard_normal((50, 12))
    loo = []
    for i in range(12):
        o = np.delete(om, i, axis=1)
        y = a @ o
        nys = y @ np.linalg.solve(o.T @ y, y.T)
        w = om[:, i]
        loo.append(np.trace(nys) + w @ (a - nys) @ w)
    assert abs(est / np.mean(loo) - 1) < 1e-8


def test_sample_covariance_is_an_ensemble_gaussian():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=1, dirichlet="left")
    rng, g = _setup(fes)
    ens = ngsrng.sample_covariance(g, 25, seed=0)
    assert ens.df == 25 and ens.nrand == 25
    assert np.abs(ens.b[np.flatnonzero(~np.array(fes.FreeDofs()))]).max() == 0  # Dirichlet values stay fixed
    x = rng.normal(size=30)
    ens2 = ngsrng.sample_covariance(x, fes=fes)  # given samples: sample mean, 1/(n - 1)
    xc = x - x.mean(0)
    np.testing.assert_allclose(ens2.b @ ens2.b.T, xc.T @ xc / 29, atol=1e-12)
    assert ens2.df == 29
    # exact KL of the ensemble, exact variance and trace
    kl = ngsrng.karhunen_loeve(ens2)
    assert kl.nrand == 29  # centred samples span 29 dimensions
    assert abs(ngsrng.covariance_trace(ens2).value / kl.eigenvalues.sum() - 1) < 1e-10
    # unbiased: E C_n = C (trace check over replicates)
    t = np.mean([ngsrng.covariance_trace(ngsrng.sample_covariance(g, 10, seed=s)).value for s in range(30)])
    assert abs(t / ngsrng.covariance_trace(g).value - 1) < 0.1


def test_karhunen_loeve_fields_have_exactly_the_covariance():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=1)
    _, g = _setup(fes)
    ens = ngsrng.sample_covariance(g, 15, seed=1)
    rng = ngsrng.default_rng(fes, seed=4, covariance=ens, dtype="float64")
    cov = ens.b @ ens.b.T
    for coefficients in ("uniform", "rademacher", "laplace"):
        k = rng.karhunen_loeve(coefficients, size=40000)
        assert np.abs(np.cov(k.T) - cov).max() < 0.05 * np.abs(cov).max(), coefficients
    # bounded coefficients: |f - m| <= sqrt(3) sum_k sqrt(lambda_k) |phi_k| at every dof (vertex values for P1)
    kl = ngsrng.karhunen_loeve(ens)
    bound = np.sqrt(3) * np.abs(kl.modes) @ np.sqrt(kl.eigenvalues)
    k = rng.karhunen_loeve("uniform", size=2000)
    assert (np.abs(k - ens.mean) <= bound + 1e-12).all()
    with pytest.raises(ValueError, match="rank"):
        ngsrng.default_rng(fes, seed=1, correlation=0.3).karhunen_loeve("uniform")
    assert ngsrng.default_rng(fes, seed=1, correlation=0.3).karhunen_loeve("uniform", rank=10, size=3).shape == (
        3, fes.ndof)


def test_covariance_function():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2)
    _, g = _setup(fes)
    sigma = _dense(g)
    x0, pts = (0.3, 0.6), _points(fes)
    k = ngsrng.covariance_function(g, x0)
    s = g.space
    e0 = ngsrng.observation_operator(fes, points=[x0])[:, s.free].toarray()
    e = ngsrng.observation_operator(fes, points=pts)[:, s.free].toarray()
    np.testing.assert_allclose([k(fes.mesh(*p)) for p in pts], (e @ sigma @ e0.T)[:, 0], rtol=1e-10)
    rho = ngsrng.covariance_function(g, x0, correlation=True)
    assert abs(rho(fes.mesh(*x0)) - 1) < 1e-8


def test_pointwise_families_use_the_local_covariance():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=1)
    _, g = _setup(fes)
    kl = ngsrng.karhunen_loeve(g, 20)
    rng = ngsrng.default_rng(fes, seed=1, correlation=kl)
    assert rng.info("uniform")["standardisation"] == "exact"
    # a finite-rank Gaussian's exact std costs no applications: the cache holds the factor-based local covariance
    assert local_covariance(kl, "exact").applications == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no Monte Carlo warning
        rng.uniform(size=2)


def test_explicit_exact_posterior_stays_exact_beyond_the_auto_limit(monkeypatch):
    import ngsrng.covariance as cv

    monkeypatch.setattr(cv, "EXACT_LIMIT", 10)  # every space is "large" now
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=1)
    _, prior = _setup(fes)
    post = prior.condition(points=_points(fes, 4), noise=0.01)
    auto = local_covariance(post)
    assert not auto.exact and "low rank" in auto.method  # auto: the prior part is estimated
    exact = local_covariance(post, "exact")
    assert exact.exact and exact is not auto
    e = ngsrng.observation_operator(fes, points=_points(fes))[:, prior.space.free]
    np.testing.assert_allclose(exact.values(e), _variance(fes, _dense(post), _points(fes)), rtol=1e-9)


def test_functional_covariance_is_g_sigma_gt():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2, dirichlet="left")
    _, g = _setup(fes)
    sigma = _dense(g)
    pts = _points(fes, 12)
    gmat = ngsrng.observation_operator(fes, points=pts)[:, g.space.free].toarray()
    np.testing.assert_allclose(ngsrng.functional_covariance(g, points=pts), gmat @ sigma @ gmat.T, rtol=1e-10,
                               atol=1e-14)
    post = g.condition(points=pts[:5], noise=0.01)
    lf = ngs.LinearForm(ngs.x * fes.TestFunction() * ngs.dx).Assemble()
    c = ngsrng.functional_covariance(post, functionals=[lf], averages=(pts[5:8], 0.1))
    gp = ngsrng.observation_operator(fes, functionals=[lf], averages=(pts[5:8], 0.1))[:, g.space.free].toarray()
    np.testing.assert_allclose(c, gp @ _dense(post) @ gp.T, rtol=1e-9)


@pytest.mark.parametrize("kind", ["H1", "L2"])
def test_robin_boundary_flattens_the_variance(kind):
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.05))
    fes = ngs.H1(mesh, order=1) if kind == "H1" else ngs.L2(mesh, order=1, dgjumps=True)
    pts = [(0.5, 0.5), (0.0, 0.5), (0.5, 0.0), (0.0, 0.0), (1.0, 1.0)] if kind == "H1" else \
        [(0.5, 0.5), (0.01, 0.5), (0.5, 0.01), (0.01, 0.01), (0.99, 0.99)]
    v = {}
    for boundary in ("neumann", "robin"):
        corr = ngsrng.matern_correlation(0.3, nu=1.0, boundary=boundary)
        g = ngsrng.default_rng(fes, seed=0, correlation=corr, dtype="float64").gaussian()
        v[boundary] = np.diag(ngsrng.functional_covariance(g, points=pts))
    assert v["neumann"][1:].min() > 1.7 * v["neumann"][0]  # reflection: about 2x at edges, more in corners
    assert np.abs(v["robin"][1:] / v["robin"][0] - 1).max() < 0.2  # Robin, beta = kappa / 1.42: nearly flat
    with pytest.raises(ValueError):
        ngsrng.matern_correlation(0.3, boundary="dirichlet")
    spec = ngsrng.matern(5.0, robin="auto")
    a = ngsrng.SpaceInfo(fes if kind == "H1" else ngs.H1(mesh, order=1)).assemble_free(spec.form)
    b = ngsrng.SpaceInfo(fes if kind == "H1" else ngs.H1(mesh, order=1)).assemble_free(ngsrng.matern(5.0).form)
    one = np.ones(a.shape[0])
    assert abs((one @ (a - b) @ one) / (5.0 / 1.42) / 4.0 - 1) < 1e-10  # beta * |boundary| = (kappa / 1.42) * 4
