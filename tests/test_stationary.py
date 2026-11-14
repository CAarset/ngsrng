"""Matérn fields with constant pointwise variance on bounded domains (ngsrng.stationary): exactness of the sources,
constant variance, Matérn correlations, the Dirichlet boundary update, and the sampler contract."""

import pickle

import ngsolve as ngs
import numpy as np
import pytest
import torch

import ngsrng
import ngsrng.stationary as st
from ngsrng.stationary import _DenseSource, _GridSource, basis_at, matern_function, padding_for

LENGTH = 0.3


@pytest.fixture(scope="module")
def square():
    return ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.1))


@pytest.fixture(scope="module")
def coarse():
    return ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2))


def vertices(mesh):
    xyz = np.array([v.point for v in mesh.vertices], dtype=float)[:, :mesh.dim]
    return xyz + 1e-9 * (xyz.mean(0) - xyz)  # just inside, for the point locator


def factor_rows(g, pts):
    """L^T e_x for the points (nrand x npoints): Var u(x) = |L^T e_x|^2, Cov = (L^T e_x).(L^T e_y)."""
    e = basis_at(g.space.fes, pts)[:, g.space.free].toarray().T
    return g.apply_t(torch.from_numpy(e)).numpy()


def variance(g, pts):
    return (factor_rows(g, pts) ** 2).sum(0)


def test_matern_function_and_padding():
    for nu in (0.5, 1.0, 2.0, 5.0):
        assert matern_function(0.0, LENGTH, nu) == 1.0
        assert 0.12 < float(matern_function(LENGTH, LENGTH, nu)) < 0.15  # the practical range
        p = padding_for(LENGTH, nu, 1e-3)
        assert float(matern_function(p, LENGTH, nu)) == pytest.approx(1e-3, rel=1e-4)
    np.testing.assert_allclose(matern_function(np.array([0.1, 0.2]), LENGTH, 0.5),
                               np.exp(-np.sqrt(4.0) / LENGTH * np.array([0.1, 0.2])), rtol=1e-12)  # exponential


def test_circulant_embedding_is_exact_on_the_grid():
    """On the grid nodes the FFT source has exactly the Matérn covariance (the embedding is PSD here); its adjoint is
    the transpose."""
    x = np.linspace(0.0, 0.5, 6)
    nodes = np.array([(a, b) for a in x for b in x])
    src = _GridSource(nodes, nodes.min(0), nodes.max(0), LENGTH, 1.0, 0.1)
    assert src.info["exact_on_grid"] and src.n == [6, 6]
    v = src.values(np.eye(src.nrand))
    dist = np.linalg.norm(nodes[:, None] - nodes[None], axis=-1)
    np.testing.assert_allclose(v @ v.T, matern_function(dist, LENGTH, 1.0), atol=1e-12)
    rng = np.random.default_rng(0)
    eta, y = rng.standard_normal((src.nrand, 2)), rng.standard_normal((len(nodes), 2))
    assert (src.values(eta) * y).sum() == pytest.approx((eta * src.values_t(y)).sum(), rel=1e-12)


def test_dense_source_is_the_kernel():
    pts = np.random.default_rng(1).random((80, 2))
    src = _DenseSource(pts, LENGTH, 2.0, max_points=100)
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    np.testing.assert_allclose(src.r @ src.r.T, matern_function(dist, LENGTH, 2.0), atol=1e-10)
    with pytest.raises(MemoryError, match="max_dense"):
        _DenseSource(pts, LENGTH, 2.0, max_points=50)


@pytest.mark.parametrize("construction, method", [("torus", "auto"), ("restrict", "fft")])
def test_variance_is_constant_up_to_the_boundary(square, construction, method):
    fes = ngs.H1(square, order=2)
    g = ngsrng.StationaryMatern(fes, LENGTH, 2.0, construction=construction, method=method, seed=0)
    v = variance(g, vertices(square))
    assert 0.97 < v.min() and v.max() < 1.08, (v.min(), v.max())  # FE projection error only (h = 0.1)
    neumann = ngsrng.Gaussian(fes, ngsrng.matern_correlation(LENGTH, 2.0).bind(fes), dtype="float64", seed=0)
    vn = variance(neumann, vertices(square))
    assert vn.max() > 3.5  # the natural condition: about 4x in the corners
    rng = np.random.default_rng(2)
    eta, y = rng.standard_normal((g.nrand, 2)), rng.standard_normal((g.space.nfree, 2))
    lhs = (g.apply(torch.from_numpy(eta)).numpy() * y).sum()
    assert lhs == pytest.approx((eta * g.apply_t(torch.from_numpy(y)).numpy()).sum(), rel=1e-10)


def test_dense_and_fft_restrict_agree(square):
    """Both represent the restricted Matérn field (exact kernel vs. FFT on a grid): with a fine grid, the same
    covariance up to the multilinear interpolation (P1 here, so that the dense kernel stays small)."""
    fes = ngs.H1(square, order=1)
    dense = ngsrng.StationaryMatern(fes, LENGTH, 2.0, construction="restrict", method="dense")
    fft = ngsrng.StationaryMatern(fes, LENGTH, 2.0, construction="restrict", method="fft", resolution=0.01)
    assert dense.plan["method"] == "dense" and fft.plan["source"]["exact_on_grid"]
    pts = np.vstack([vertices(square)[::7], [[0.5, 0.5], [0.8, 0.5]]])
    ld, lf = factor_rows(dense, pts), factor_rows(fft, pts)
    np.testing.assert_allclose(lf.T @ lf, ld.T @ ld, atol=5e-3)
    vd = (ld**2).sum(0)
    assert 0.85 < vd.min() and vd.max() < 1.3  # P1 projection at kappa h = 1.3: overshoots at vertices


def test_correlations_are_matern(square):
    """Correlation at one length: Matérn, at the boundary as in the interior; the torus' images of points at opposite
    edges are at least the padding away, so their correlation is at most tol (plus discretisation)."""
    fes = ngs.H1(square, order=2)
    rho = float(matern_function(LENGTH, LENGTH, 2.0))
    pairs = np.array([[1e-9, 0.5], [LENGTH, 0.5], [0.35, 0.5], [0.35 + LENGTH, 0.5], [1e-9, 1e-9], [LENGTH, 1e-9]])
    for construction, method in (("torus", "auto"), ("restrict", "fft")):
        g = ngsrng.StationaryMatern(fes, LENGTH, 2.0, construction=construction, method=method)
        lt = factor_rows(g, pairs)
        var = (lt**2).sum(0)
        for a, b in ((0, 1), (2, 3), (4, 5)):
            assert lt[:, a] @ lt[:, b] / np.sqrt(var[a] * var[b]) == pytest.approx(rho, abs=0.02)
    torus = ngsrng.StationaryMatern(fes, LENGTH, 2.0, construction="torus", tol=1e-3)
    lt = factor_rows(torus, np.array([[1e-9, 0.5], [1 - 1e-9, 0.5]]))
    assert abs(lt[:, 0] @ lt[:, 1]) < 5e-3
    assert float(matern_function(torus.plan["source"]["padding"], LENGTH, 2.0)) == pytest.approx(1e-3, rel=1e-3)


@pytest.mark.parametrize("dense_limit", [st.DENSE_CONDITIONING_VALUES, 0])
def test_dirichlet_boundary_update_is_exact_conditioning(coarse, monkeypatch, dense_limit):
    """On a space with Dirichlet dofs the field is the stationary one conditioned on its trace there: the covariance
    is Sigma_FF - Sigma_FD Sigma_DD^-1 Sigma_DF of the unconditioned field (dense and matrix-free forms)."""
    monkeypatch.setattr(st, "DENSE_CONDITIONING_VALUES", dense_limit)
    free_space = ngs.H1(coarse, order=2)
    dir_space = ngs.H1(coarse, order=2, dirichlet="left")
    full = ngsrng.StationaryMatern(free_space, LENGTH, 2.0, construction="torus")
    cond = ngsrng.StationaryMatern(dir_space, LENGTH, 2.0, construction="torus")
    assert cond.plan["conditioning"]["mode"] == ("dense" if dense_limit else "matrix-free")
    lf = full.apply(torch.eye(full.nrand, dtype=torch.float64)).numpy()
    sigma = lf @ lf.T
    f = np.flatnonzero(np.array(dir_space.FreeDofs(), bool))
    d = np.setdiff1d(np.arange(dir_space.ndof), f)
    expected = sigma[np.ix_(f, f)] - sigma[np.ix_(f, d)] @ np.linalg.solve(sigma[np.ix_(d, d)], sigma[np.ix_(d, f)])
    lc = cond.apply(torch.eye(cond.nrand, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(lc @ lc.T, expected, atol=1e-9 * np.abs(expected).max())
    rng = np.random.default_rng(3)
    eta, y = rng.standard_normal((cond.nrand, 2)), rng.standard_normal((cond.space.nfree, 2))
    lhs = (cond.apply(torch.from_numpy(eta)).numpy() * y).sum()
    assert lhs == pytest.approx((eta * cond.apply_t(torch.from_numpy(y)).numpy()).sum(), rel=1e-9)
    v = variance(cond, np.array([[1e-9, 0.5], [0.6, 0.5]]))
    assert v[0] < 1e-12 and 0.8 < v[1] < 1.25  # zero on the Dirichlet edge; two lengths away as unconditioned
    np.testing.assert_allclose(v[1], variance(full, np.array([[0.6, 0.5]]))[0], rtol=0.05)


def test_boundary_update_is_insensitive_to_rounding_in_the_set_up():
    """The kept regression comes from a QR of L^T e_D: a rounding-size change of the set-up (as between multithreaded
    NGSolve set-ups) moves the conditioned samples by about kappa(Sigma_DD)^(1/2) eps, not kappa(Sigma_DD) eps as
    through a Cholesky factor of Sigma_DD (here kappa(Sigma_DD) ~ 1e8: smooth field, many Dirichlet dofs per length)."""
    import scipy.linalg as sla

    from ngsrng.randomized import GRAM_JITTER, gram_regression

    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.1)), order=2, dirichlet=".*")
    f = ngsrng.StationaryMatern(fes, 1.0, 5.0, construction="torus", seed=1)
    assert f.plan["conditioning"]["mode"] == "dense"
    nd, free = f._dir.size, f.space.free
    e = np.zeros((f.space.ndof, nd))
    e[f._dir, np.arange(nd)] = 1.0
    lt = f._project_t(e)
    sdd = lt.T @ lt
    assert np.linalg.cond(sdd) > 1e7

    def cholesky_route(lt):
        s = lt.T @ lt
        factor = sla.cho_factor(s + GRAM_JITTER * np.trace(s) / nd * np.eye(nd), lower=True)
        return sla.cho_solve(factor, f._project(lt)[free].T).T

    c = f._project(np.random.default_rng(0).standard_normal((f.nrand, 3)))
    moved = lt * (1 + 1e-14 * np.random.default_rng(1).standard_normal(lt.shape))
    change = {}
    for name, make in (("qr", lambda m: gram_regression(m, f._project, free)), ("cholesky", cholesky_route)):
        base, other = (c[free] - make(m) @ c[f._dir] for m in (lt, moved))
        change[name] = np.linalg.norm(other - base) / np.linalg.norm(base)
    assert change["qr"] < 1e-10 and change["cholesky"] > 20 * change["qr"]
    # the field uses that regression, and both routes agree (same law)
    np.testing.assert_allclose(f._cond[1], gram_regression(lt, f._project, free), rtol=0, atol=1e-10)
    np.testing.assert_allclose(f._cond[1] @ c[f._dir], cholesky_route(lt) @ c[f._dir], rtol=0,
                               atol=1e-6 * np.abs(c[free]).max())


def test_dirichlet_data_come_from_the_mean(coarse):
    dir_space = ngs.H1(coarse, order=2, dirichlet="left|bottom")
    g = ngsrng.StationaryMatern(dir_space, LENGTH, 1.0, construction="restrict", method="fft", mean=1 + ngs.y,
                                seed=4)
    s = g.sample(3).numpy()
    data = ngs.GridFunction(dir_space)
    data.Set(1 + ngs.y)
    d = ~np.array(dir_space.FreeDofs(), bool)
    np.testing.assert_allclose(s[d], np.repeat(data.vec.FV().NumPy()[d][:, None], 3, axis=1), atol=1e-12)


def test_sampler_contract(coarse):
    """In-place fills, seeding and pickling, as for every sampler."""
    fes = ngs.H1(coarse, order=2, dirichlet="left")
    make = lambda seed: ngsrng.StationaryMatern(fes, LENGTH, 1.0, construction="torus", seed=seed)  # noqa: E731
    ref = make(5).sample(6).numpy()
    sampler = make(5)
    sampler.work_values = 2 * (fes.ndof + sampler.nrand)  # several chunks
    buf = np.empty((fes.ndof, 6))
    assert sampler.sample(6, out=buf) is buf
    np.testing.assert_allclose(buf, ref, rtol=0, atol=1e-12 * np.abs(ref).max())
    assert np.abs(make(6).sample(6).numpy() - ref).max() > 0.1
    twin = pickle.loads(pickle.dumps(make(5)))
    np.testing.assert_allclose(twin.sample(6).numpy(), ref, atol=1e-12 * np.abs(ref).max())
    torch.manual_seed(1)
    a = make(None).sample(2).numpy()
    torch.manual_seed(1)
    b = make(None).sample(2).numpy()  # seed None follows torch's global state (another factorisation: rounding)
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-12 * np.abs(a).max())
    assert isinstance(make(5).sample(), ngs.GridFunction)


def test_generator_families_use_it(coarse):
    fes = ngs.H1(coarse, order=2)
    corr = ngsrng.StationaryMatern(fes, LENGTH, 2.0, construction="torus")
    rng = ngsrng.default_rng(fes, seed=7, correlation=corr)
    assert np.asarray(rng.normal(size=3)).shape == (3, fes.ndof)
    u = np.asarray(rng.uniform(size=2))
    assert 0 < u.min() and u.max() < 1
    a, b = rng.spawn(2)
    assert np.abs(np.asarray(a.normal(size=2)) - np.asarray(b.normal(size=2))).max() > 0.1
    np.testing.assert_allclose(ngsrng.pointwise_variance(corr)(coarse(0.5, 0.5)),
                               variance(corr, np.array([[0.5, 0.5]]))[0], rtol=1e-10)


def test_posteriors_and_observations(coarse):
    """Exact posteriors (LinearObservations / condition), Matheron sampling and observed data with a stationary
    prior; the posterior's covariance and mean against the dense formulas."""
    from ngsrng.observations import observation_operator

    fes = ngs.H1(coarse, order=2)
    prior = ngsrng.StationaryMatern(fes, LENGTH, 2.0, construction="torus", seed=1)
    pts, noise_var, y = [[0.02, 0.5], [0.5, 0.5]], 0.01, np.array([1.0, -1.0])  # noise= is the variance
    post = prior.condition(points=pts, noise=noise_var, data=y)
    assert isinstance(post, ngsrng.StationaryMatern) and post._prior is prior
    lp = prior.apply(torch.eye(prior.nrand, dtype=torch.float64)).numpy()
    sigma = lp @ lp.T
    g = observation_operator(fes, points=pts)[:, prior.space.free].toarray()
    gain = sigma @ g.T @ np.linalg.inv(g @ sigma @ g.T + noise_var * np.eye(2))
    lq = post.apply(torch.eye(post.nrand, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(lq @ lq.T, sigma - gain @ g @ sigma, atol=1e-9 * np.abs(sigma).max())
    np.testing.assert_allclose(post.mean[prior.space.free], gain @ y, atol=1e-9)
    assert post.sample(3).shape == (fes.ndof, 3)
    v = ngsrng.pointwise_variance(post)
    assert v(coarse(0.02, 0.5)) < 0.1 < v(coarse(0.9, 0.9))  # observed point: small posterior variance
    obs = ngsrng.LinearObservations(prior, points=pts, noise=noise_var, factorize=False)
    assert ngsrng.MatheronPosterior(obs, data=y, seed=2).sample(4).shape == (fes.ndof, 4)
    assert ngsrng.ObservedGaussian(prior, points=pts, scale=np.sqrt(noise_var)).sample(5).shape == (2, 5)


@pytest.mark.parametrize("dirichlet", [None, "interface"])
def test_subdomain_spaces(dirichlet):
    """A space on part of the mesh (definedon, with or without Compress, with or without a Dirichlet part): the field
    lives on the subdomain's elements only (their quadrature points and extent), and the compressed and uncompressed
    spaces give the same field, dof for dof."""
    from ngsrng.domains import NGSSquare
    from ngsrng.quadrature import QuadratureRule
    from ngsrng.space import SpaceInfo

    mesh = NGSSquare(maxh=0.1).mesh  # unit square; the subdomain "inner" is bounded by "interface"
    base = ngs.H1(mesh, order=2, definedon="inner", **({} if dirichlet is None else {"dirichlet": dirichlet}))
    comp = ngs.Compress(base)
    fields = [ngsrng.StationaryMatern(v, 0.1, 2.0, construction="torus", seed=1) for v in (base, comp)]
    per_element = QuadratureRule.count(SpaceInfo(ngs.H1(mesh, order=2)), 4) // mesh.ne
    inner = sum(el.mat == "inner" for el in mesh.Elements(ngs.VOL))
    for f in fields:
        assert f.plan["quadrature_points"] == inner * per_element
        assert max(f.plan["source"]["box"]) < 1.0  # the subdomain (0.3 wide) and its padding, not the whole mesh
        assert f.sample(3).shape == (f.space.ndof, 3) and isinstance(f.sample(), ngs.GridFunction)
    # compressed dof -> dof of the uncompressed space, from the element dof lists
    to_base = np.full(comp.ndof, -1)
    for el in mesh.Elements(ngs.VOL):
        for c, b in zip(comp.GetDofNrs(el), base.GetDofNrs(el)):
            if c >= 0:
                to_base[c] = b
    rows = np.searchsorted(fields[0].space.free, to_base[fields[1].space.free])
    eye = torch.eye(fields[0].nrand, dtype=torch.float64)
    lb, lc = (f.apply(eye).numpy() for f in fields)
    np.testing.assert_allclose(lc, lb[rows], atol=1e-10 * np.abs(lb).max())
    if dirichlet is not None:
        s = fields[1].sample(4).numpy()
        fixed = ~np.array(comp.FreeDofs(), bool)
        assert fixed.any() and np.abs(s[fixed]).max() == 0.0  # conditioned on zero data on the interface


def test_pointwise_families_on_a_dirichlet_field(coarse):
    """On a space with Dirichlet dofs the field is conditioned on boundary data, and its pointwise families are
    standardised by the field before that conditioning, as posteriors by their prior: the boundary values are the
    transform of the data (the median for zero data) and the spread grows back away from the boundary. bnd= keeps
    the trace dofs fixed by dirichlet=, so a Bernstein trace stays inside the range."""
    from ngsrng.boundary import boundary_dofs

    fes = ngs.H1(coarse, order=2, dirichlet="bottom")
    rng = ngsrng.default_rng(fes, seed=3, dtype="float64",
                             correlation=ngsrng.matern_correlation(LENGTH, nu=2, boundary="torus"))
    assert rng.info("uniform", -0.2, 1.0, representation="bernstein")["standardised_by"].startswith("prior")
    u = np.asarray(rng.uniform(-0.2, 1.0, size=300, representation="bernstein"))
    fixed = np.flatnonzero(~np.array(fes.FreeDofs(), bool))
    np.testing.assert_allclose(u[:, fixed[fixed < coarse.nv]], 0.4, atol=1e-12)  # vertex dofs: the median
    xyz = vertices(coarse)
    far = xyz[np.argmin(np.linalg.norm(xyz - [0.5, 0.7], axis=1))]  # a vertex: exact marginals there
    vals = basis_at(fes, np.array([[0.5, 0.02], far])) @ u.T
    assert vals.var(1)[0] < 0.25 * vals.var(1)[1] and vals.var(1)[1] == pytest.approx(1.2**2 / 12, rel=0.3)
    g = np.asarray(rng.uniform(-0.2, 1.0, size=20, representation="bernstein", bnd="left"))
    on_trace = np.setdiff1d(boundary_dofs(fes, "left", free_only=False), boundary_dofs(fes, "left"))
    assert on_trace.size and np.allclose(g[:, on_trace[on_trace < coarse.nv]], 0.4)
    trace = basis_at(fes, np.column_stack([np.full(41, 1e-9), np.linspace(1e-9, 1 - 1e-9, 41)])) @ g.T
    assert -0.2 <= trace.min() and trace.max() <= 1.0


def test_lshape_and_fractional_smoothness():
    """A non-convex domain with a re-entrant corner, and a fractional power (nu = 1.5 in 2D: the rational route on the
    torus)."""
    from netgen.occ import MoveTo, OCCGeometry, Rectangle

    shape = Rectangle(1, 1).Face() - MoveTo(0.5, 0.5).Rectangle(0.5, 0.5).Face()
    mesh = ngs.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.1))
    fes = ngs.H1(mesh, order=2)
    for nu in (2.0, 1.5):
        g = ngsrng.StationaryMatern(fes, LENGTH, nu, construction="torus")
        v = variance(g, vertices(mesh))
        assert 0.96 < v.min() and v.max() < 1.08, (nu, v.min(), v.max())


@pytest.mark.parametrize("dim", [1, 3])
def test_one_and_three_dimensions(dim):
    if dim == 1:
        from ngsolve.meshes import Make1DMesh
        mesh = Make1DMesh(40)
        nu, order = 1.5, 2
    else:
        from netgen.csg import unit_cube
        mesh = ngs.Mesh(unit_cube.GenerateMesh(maxh=0.25))
        nu, order = 1.5, 1
    fes = ngs.H1(mesh, order=order)
    length = LENGTH if dim == 1 else 1.0  # P1 at h = 0.25 resolves a length of 1 (kappa h = 0.9), not of 0.3
    for construction, method in (("torus", "auto"), ("restrict", "fft")):
        g = ngsrng.StationaryMatern(fes, length, nu, construction=construction, method=method, seed=0)
        v = variance(g, vertices(mesh))
        assert 0.8 < v.min() and v.max() < 1.2, (dim, construction, v.min(), v.max())
        assert g.sample(2).shape == (fes.ndof, 2)


def test_matern_correlation_boundary_options(coarse):
    """boundary="torus" / "restrict" in matern_correlation: bind returns the sampler; the generator builds it with
    its own randomness, caches it, spawns and pickles it."""
    fes = ngs.H1(coarse, order=2)
    for boundary, options in (("torus", {}), ("restrict", {"method": "fft"})):
        corr = ngsrng.matern_correlation(LENGTH, 2.0, boundary=boundary, **options)
        assert corr.stationary and corr.options == options
        g = corr.bind(fes, seed=3)
        assert isinstance(g, ngsrng.StationaryMatern) and g.construction == boundary
        rng = ngsrng.default_rng(fes, seed=1, correlation=corr, dtype="float64")
        gauss = rng.gaussian()
        assert isinstance(gauss, ngsrng.StationaryMatern) and rng.gaussian() is gauss
        assert 0 < np.asarray(rng.uniform(size=2)).min()
        twin = pickle.loads(pickle.dumps(rng))
        np.testing.assert_allclose(np.asarray(twin.normal(size=2)), np.asarray(rng.normal(size=2)), atol=1e-12)
        a, b = rng.spawn(2)
        assert np.abs(np.asarray(a.normal(size=1)) - np.asarray(b.normal(size=1))).max() > 0.1
    with pytest.raises(ValueError, match="pass boundary='robin' as well"):
        ngsrng.matern_correlation(LENGTH, 2.0, boundary="torus", beta=1.0)
    with pytest.raises(ValueError, match="options of boundary='torus' or 'restrict'"):
        ngsrng.matern_correlation(LENGTH, 2.0, method="fft")
    with pytest.raises(TypeError, match="unknown option"):
        ngsrng.matern_correlation(LENGTH, 2.0, boundary="torus", padding=1.0)
    with pytest.raises(ValueError, match="boundary must be"):
        ngsrng.matern_correlation(LENGTH, 2.0, boundary="periodic")


def test_errors(coarse):
    fes = ngs.H1(coarse, order=1)
    with pytest.raises(ValueError, match="construction has no default"):
        ngsrng.StationaryMatern(fes, LENGTH, 1.0)
    with pytest.raises(ValueError, match="not available for construction='torus'"):
        ngsrng.StationaryMatern(fes, LENGTH, 1.0, construction="torus", method="fft")
    with pytest.raises(ValueError, match="tol must lie"):
        ngsrng.StationaryMatern(fes, LENGTH, 1.0, construction="torus", tol=2.0)
    with pytest.raises(NotImplementedError, match="scalar"):
        ngsrng.StationaryMatern(ngs.VectorH1(coarse, order=1), LENGTH, 1.0, construction="torus")
