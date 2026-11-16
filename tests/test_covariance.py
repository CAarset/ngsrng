"""Covariance operators as data: local covariances (pointwise variance k(x, x)), traces, covariance functions,
sample (Wishart) covariances, operator covariances and Karhunen-Loeve fields with non-Gaussian coefficients."""

import dataclasses
import warnings

import ngsolve as ngs
import numpy as np
import pytest
import scipy.stats as st
import torch
from ngsolve.meshes import Make1DMesh, MakeStructured2DMesh, MakeStructured3DMesh
from stat_checks import assert_bounded_mean, assert_law, assert_quadratic_form, assert_scaled_chi2

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
    """Exact laws: a Monte Carlo variance from s Gaussian samples is the exact variance times chi^2_s / s; "lowrank"
    adds to its exact low-rank part a Monte Carlo variance of the residual (its samples independent of the basis), so
    (estimate - low-rank part) is (exact - low-rank part) times chi^2_s / s. The standard errors are the samples'
    standard deviation over sqrt(s)."""
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.12)), order=1)
    _, g = _setup(fes, length=0.4)
    pts = _points(fes)
    e = ngsrng.observation_operator(fes, points=pts)[:, g.space.free]
    ref = local_covariance(g, "exact").values(e)
    for method, kw in (("lowrank", dict(rank=40, samples=48)), ("mc", dict(samples=400))):
        lc = local_covariance(g, method, **kw)
        v, se = lc.values(e, stderr=True)
        base = dataclasses.replace(lc, mc=None).values(e)  # the exact part (none for "mc")
        a, b = lc.mc
        t = (e @ a) * (e @ b)
        np.testing.assert_allclose(v - base, t.mean(axis=1), rtol=1e-10)
        np.testing.assert_allclose(se, t.std(axis=1, ddof=1) / np.sqrt(kw["samples"]), rtol=1e-10)
        resid = ref - base
        assert resid.min() > 1e-10 * ref.max(), method  # a residual to estimate at every point
        assert_scaled_chi2(v - base, resid, kw["samples"], what=method)
    # with a basis spanning everything, the low-rank part is exact and the residual vanishes
    small = ngs.H1(Make1DMesh(10), order=1)
    _, g1 = _setup(small)
    lc = local_covariance(g1, "lowrank", rank=g1.space.nfree, samples=8)
    e1 = ngsrng.observation_operator(small, points=_points(small))[:, g1.space.free]
    np.testing.assert_allclose(lc.values(e1), local_covariance(g1, "exact").values(e1), rtol=1e-7)


def test_fe_space_basis_spans_the_noise_space_range():
    """With more noise than free dofs the "lowrank" basis is computed in FE space. Subspace iteration on Sigma M from
    L Omega spans the range of the noise-space randomised SVD of L from the same Omega, (Sigma M)^q L Omega; the
    estimator starts from Sigma D^(1/2) Omega instead (D = diag M) and is unbiased."""
    import ngsrng.covariance as cov
    from ngsrng.randomized import gaussian_block, gram_orth, randomized_svd

    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.15)), order=2)
    _, g = _setup(fes, length=0.4)
    src = as_covariance(g, compute="host")
    mass, k = g.space.mass_free, 20
    assert src.nrand > g.space.nfree
    omega = gaussian_block(src.nrand, k, 7)
    q_noise, _, _ = randomized_svd(src.l, lambda y: src.lt(mass @ y), src.nrand, k, oversampling=0,
                                   power_iterations=2, seed=7, orth=gram_orth(mass))
    q_fe = cov._subspace_iteration(src.l(omega), src.sigma, mass, cov._stable_orth(mass), 2)
    np.testing.assert_allclose(q_fe.T @ (mass @ q_fe), np.eye(k), atol=1e-12)
    cosines = np.linalg.svd(q_noise.T @ (mass @ q_fe), compute_uv=False)  # of the principal angles
    np.testing.assert_allclose(cosines, 1.0, atol=1e-9)
    # the estimator: FE-space basis, unbiased (within its standard errors of the exact variance)
    lc = local_covariance(g, "lowrank", rank=k, samples=64)
    assert "FE space" in lc.method and lc.applications == 8 * k + 64  # 2 (q + 2) k + samples
    e = ngsrng.observation_operator(fes, points=_points(fes))[:, g.space.free]
    ref = local_covariance(g, "exact").values(e)
    v, se = lc.values(e, stderr=True)
    assert np.abs((v - ref) / se).max() < 5
    q, _, apps, where = cov._basis(src, k, 2, np.random.default_rng(0))
    assert where == "FE space" and apps == 8 * k and q.shape == (g.space.nfree, k)
    np.testing.assert_allclose(q.T @ (mass @ q), np.eye(k), atol=1e-12)


def test_lowrank_basis_in_noise_space_when_it_is_the_smaller_one():
    fes = SPACES["2D-L2-p1"]()  # diagonal mass: as much noise as free dofs
    g = ngsrng.default_rng(fes, seed=1, precision=ngsrng.matern_dg(fes, kappa=8.0), dtype="float64").gaussian()
    lc = local_covariance(g, "lowrank", rank=10, samples=16)
    assert g.nrand <= g.space.nfree and "noise space" in lc.method and lc.applications == 7 * 10 + 16


def test_samples_do_not_depend_on_the_block_size(monkeypatch):
    import ngsrng.covariance as cov

    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2)
    _, g = _setup(fes)
    src = as_covariance(g, compute="host")
    a = src.samples(40, np.random.default_rng(3))
    monkeypatch.setattr(cov, "block_columns", lambda *rows: 7)
    b = src.samples(40, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (g.space.nfree, 40)


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
    """Hutchinson with p Gaussian probes is a Gaussian quadratic form with weights lambda_k / p (lambda: the
    eigenvalues of C = Sigma M); Hutch++'s Hutchinson part (k = p / 3 probes, independent of its subspace) and the
    "lowrank" variance's Monte Carlo residual are such forms too, with eigenvalues of a compression of C, bounded by
    C's (Laurent-Massart with those bounds). XNysTrace has no tractable law: a deterministic check (fixed seed on a
    structured mesh, so the same realisation on every machine)."""
    fes = ngs.H1(MakeStructured2DMesh(quads=False, nx=10, ny=10), order=1)
    _, g = _setup(fes, length=0.4)
    exact = ngsrng.covariance_trace(g, "exact")
    assert exact.stderr == 0 and exact.applications == 2 * g.space.nfree
    r = np.linalg.cholesky(g.space.mass_free.toarray())
    lam = np.clip(np.linalg.eigvalsh(r.T @ _dense(g) @ r), 0.0, None)  # the eigenvalues of Sigma M
    np.testing.assert_allclose(lam.sum(), exact.value, rtol=1e-10)
    for method, p in (("hutchinson", 60), ("hutch++", 60 // 3)):
        est = ngsrng.covariance_trace(g, method, probes=60, seed=1)
        assert_quadratic_form(est.value, exact.value, (lam**2).sum() / p, lam.max() / p, what=method)
    est = ngsrng.covariance_trace(g, "xnystrace", probes=60, seed=1)
    assert abs(est.value - exact.value) < 5 * est.stderr + 1e-3 * exact.value
    est = ngsrng.covariance_trace(g, "variance", variance_method="lowrank", rank=30, samples=30)
    assert_quadratic_form(est.value, exact.value, (lam**2).sum() / 30, lam.max() / 30, what="lowrank variance")
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
    # unbiased, with the exact law: an ensemble of n draws about the known mean has n tr(C_n M) = sum_k lambda_k
    # chi^2_n (lambda: the eigenvalues of Sigma M), so the mean trace of 30 independent ensembles is a Gaussian
    # quadratic form with weights lambda_k / (30 n), each 30 n times
    n = ngsrng.sample_covariance(g, 10, seed=0).df
    t = np.mean([ngsrng.covariance_trace(ngsrng.sample_covariance(g, 10, seed=s)).value for s in range(30)])
    r = np.linalg.cholesky(g.space.mass_free.toarray())
    lam = np.clip(np.linalg.eigvalsh(r.T @ _dense(g) @ r), 0.0, None)
    assert_quadratic_form(t, lam.sum(), (lam**2).sum() / (30 * n), lam.max() / (30 * n))


def test_karhunen_loeve_fields_have_exactly_the_covariance():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=1)
    _, g = _setup(fes)
    ens = ngsrng.sample_covariance(g, 15, seed=1)
    rng = ngsrng.default_rng(fes, seed=4, covariance=ens, dtype="float64")
    cov = ens.b @ ens.b.T
    kl = ngsrng.karhunen_loeve(ens)  # the expansion the generator samples (the same, exact construction)
    mass = ngsrng.SpaceInfo(fes).mass
    laws = {"uniform": st.uniform(-np.sqrt(3), 2 * np.sqrt(3)).cdf, "laplace": st.laplace(scale=np.sqrt(0.5)).cdf}
    for coefficients in ("uniform", "rademacher", "laplace"):
        k = rng.karhunen_loeve(coefficients, size=20000)
        # the coefficients back from the draws: exactly in the expansion's span, i.i.d. with the coefficient law,
        # hence the covariance exactly cov
        xi = ((k - ens.mean) @ (mass @ kl.modes)) / np.sqrt(kl.eigenvalues)
        np.testing.assert_allclose(ens.mean + (xi * np.sqrt(kl.eigenvalues)) @ kl.modes.T, k,
                                   atol=1e-10 * np.abs(k).max())
        if coefficients == "rademacher":
            np.testing.assert_allclose(np.abs(xi), 1.0, atol=1e-8)
            assert_bounded_mean(xi > 0, 0.5, 0.0, 1.0, what=coefficients)
        else:
            assert_law(xi, laws[coefficients], what=coefficients)
    np.testing.assert_allclose(kl.modes @ np.diag(kl.eigenvalues) @ kl.modes.T, cov, atol=1e-12 * np.abs(cov).max())
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


def test_variance_at_points_costs_the_points(monkeypatch):
    """(E Sigma E^T)_ii at m points: one application of L^T per point (factor) or of Sigma (operator), not one per
    free dof (the full local covariance is never computed); from a cached exact local covariance or a finite-rank
    factor at no cost; all routes exact against dense Sigma."""
    import ngsrng.covariance as cv
    from ngsrng.covariance import variance_at_points

    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2, dirichlet="bottom")
    _, prior = _setup(fes)
    post = prior.condition(points=_points(fes, 4), noise=0.01)
    pts = _points(fes, 7)
    e = ngsrng.observation_operator(fes, points=pts)[:, prior.space.free]
    u, v = fes.TnT()
    a = ngs.BilinearForm(ngs.grad(u) * ngs.grad(v) * ngs.dx + 10 * u * v * ngs.dx).Assemble()
    mass = ngs.BilinearForm(u * v * ngs.dx).Assemble()
    ainv = a.mat.Inverse(fes.FreeDofs())
    op = ainv @ mass.mat @ ainv
    sigma_op = as_covariance(op, fes=fes).sigma(np.eye(prior.space.nfree))
    real_columns = cv._columns
    monkeypatch.setattr(cv, "_columns", lambda src: pytest.fail("the full local covariance was computed"))
    for source, sigma, kw, how in ((prior, _dense(prior), {}, "L^T"), (post, _dense(post), {}, "L^T"),
                                   (op, sigma_op, {"fes": fes}, "Sigma")):
        r = variance_at_points(source, e, **kw)
        np.testing.assert_allclose(r.values, _variance(fes, sigma, pts), rtol=1e-10)
        assert r.applications == len(pts) and how in r.method
        if source is not op:  # cached on the Gaussian per evaluation matrix (a bare operator has nowhere to keep it)
            assert variance_at_points(source, e.copy(), **kw) is r
    kl = ngsrng.karhunen_loeve(prior, 20)
    r = variance_at_points(kl, e)
    np.testing.assert_allclose(r.values, _variance(fes, _dense(kl), pts), rtol=1e-10)
    assert r.applications == 0
    # as many points as free dofs: the full local covariance (cached, shared by every point set of the source)
    monkeypatch.setattr(cv, "_columns", real_columns)
    many = _points(fes, prior.space.nfree, seed=3)
    em = ngsrng.observation_operator(fes, points=many)[:, prior.space.free]
    r = variance_at_points(prior, em)
    np.testing.assert_allclose(r.values, _variance(fes, _dense(prior), many), rtol=1e-10)
    assert "all columns" in r.method
    other = _points(fes, 5, seed=4)
    eo = ngsrng.observation_operator(fes, points=other)[:, prior.space.free]
    assert variance_at_points(prior, eo).applications == 0  # from the now cached local covariance


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
        beta = {"beta": ngsrng.matern_correlation(0.3).kappa() / 1.42} if boundary == "robin" else {}
        corr = ngsrng.matern_correlation(0.3, nu=1.0, boundary=boundary, **beta)
        g = ngsrng.default_rng(fes, seed=0, correlation=corr, dtype="float64").gaussian()
        v[boundary] = np.diag(ngsrng.functional_covariance(g, points=pts))
    assert v["neumann"][1:].min() > 1.7 * v["neumann"][0]  # reflection: about 2x at edges, more in corners
    assert np.abs(v["robin"][1:] / v["robin"][0] - 1).max() < 0.2  # Robin, beta = kappa / 1.42: nearly flat
    with pytest.raises(ValueError):
        ngsrng.matern_correlation(0.3, boundary="dirichlet")
    spec = ngsrng.matern(5.0, robin=5.0 / 1.42)
    a = ngsrng.SpaceInfo(fes if kind == "H1" else ngs.H1(mesh, order=1)).assemble_free(spec.form)
    b = ngsrng.SpaceInfo(fes if kind == "H1" else ngs.H1(mesh, order=1)).assemble_free(ngsrng.matern(5.0).form)
    one = np.ones(a.shape[0])
    assert abs((one @ (a - b) @ one) / (5.0 / 1.42) / 4.0 - 1) < 1e-10  # beta * |boundary| = (kappa / 1.42) * 4


def test_space_passed_as_method_is_a_clear_error():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=1)
    _, g = _setup(fes)
    with pytest.raises(TypeError, match="pass a space as fes"):
        ngsrng.covariance_trace(g, fes)
    with pytest.raises(TypeError, match="pass a space as fes"):
        ngsrng.pointwise_variance(g, fes)


def test_operator_kl_says_when_it_stops_short():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=1)
    _, g = _setup(fes, length=0.05)  # rough: slow spectral decay
    sigma = _dense(g)
    cov = ngsrng.OperatorCovariance(lambda y: torch.from_numpy(sigma @ y.numpy()), tol=1e-6, max_rank=8)
    with pytest.warns(UserWarning, match="stopped at rank 8 \\(max_rank=8\\)"):
        kl = cov.bind(fes, seed=0)
    assert kl.trace_residual > 1e-6


def test_weighted_trace():
    """tr(Sigma W) = E ||f - m||_W^2 for a weight W (a matrix or a function), e.g. the energy of a linear image."""
    from ngsrng.reference import exact_covariance

    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=2)
    g = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", seed=0)
    from ngsrng.space import assemble

    s0 = exact_covariance(g)
    free = g.space.free
    w = assemble(fes, lambda u, v: ngs.grad(u) * ngs.grad(v) * ngs.dx)[free][:, free]  # H1 seminorm energy
    want = float(np.sum(s0 * w.toarray()))
    for weight in (w, lambda y: w @ y):
        est = ngsrng.covariance_trace(g, weight=weight, probes=40)
        assert est.method == "xnystrace" and abs(est.value - want) < 5 * est.stderr + 1e-8 * want
    assert abs(ngsrng.covariance_trace(g, "hutchinson", weight=w, probes=400).value - want) < 0.2 * want
    with pytest.raises(ValueError, match="randomised method"):
        ngsrng.covariance_trace(g, "exact", weight=w)
