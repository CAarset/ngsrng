"""Dense samplers (ngsrng.dense): the law of d.sample() at chosen DoFs or under an observation, by dense products.

The checks follow the invariants: the latent covariance is the exact discrete one (conditioning included), the
support is exact (topology), a draw is the current route's draw for the same latent values (so the laws agree to
rounding), bounded Bernstein families stay in range, the two observation syntaxes agree and compose, Gaussian fields
fold, correlated components keep R, the stream conventions hold, stored samplers round-trip bit for bit (also
inversions and NumPy functions, and against a NumPy reference sampler), and set-ups survive pickling and spawn."""

import copy
import json
import pickle

import ngsolve as ngs
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.stats as st
import torch
from conftest import cuda_usable
from scipy.special import ndtr
from stat_checks import assert_gaussian, assert_law, assert_normal, assert_same_law

import ngsrng
import ngsrng.dense as dense_module
from ngsrng.backend import owned_cache
from ngsrng.boundary import boundary_dofs
from ngsrng.distributions.bernstein import BernsteinForm
from ngsrng.seeding import Source
from ngsrng.space import SpaceInfo
from ngsrng.stationary import basis_at

CORR = ngsrng.matern_correlation(0.3, nu=2)
LEFT = np.column_stack([np.zeros(41), np.linspace(0, 1, 41)])  # the left edge


@pytest.fixture(scope="module")
def space():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=3)


@pytest.fixture(scope="module")
def rng(space):
    return ngsrng.default_rng(space, seed=1, dtype="float64", correlation=CORR)


def sigma(gauss):
    """The coefficient covariance on the free dofs, dense (small spaces only)."""
    eye = torch.eye(gauss.space.nfree, dtype=torch.float64)
    return gauss.apply(gauss.apply_t(eye)).double().numpy()


def full_cov(gauss, ndof):
    cov = np.zeros((ndof, ndof))
    free = gauss.space.free
    cov[np.ix_(free, free)] = sigma(gauss)
    return cov


def route_draw(gen, field, n):
    """The current route's draw, split as ngsrng.correlated splits it: latent values, then the map, representation
    and bnd= (returns both)."""
    w = field._latent(n)
    w = w.cpu().numpy() if isinstance(w, torch.Tensor) else w
    vals = field._represent(field._from_latent(w), n)
    vals = (vals.cpu().numpy() if isinstance(vals, torch.Tensor) else vals).T.copy()
    if getattr(field, "bnd", None) is not None:
        gen._apply_bnd(field, vals)
    return w, vals


def noise_for(sampler, w, i=0):
    """The standard normals z with which the dense sampler's latent is closest to given latent values w (|S| x n),
    and the distance (zero up to rounding, or up to the cut when K is singular)."""
    law = sampler.law
    f, mu = law["factor"], law["components"][i]["mean"]
    ws = w[law["latent_index"]]
    z, *_ = np.linalg.lstsq(f, ws - mu[:, None], rcond=None)
    return z.T, float(np.abs(f @ z + mu[:, None] - ws).max())


def affine_law(sampler):
    """Covariance and mean of a Gaussian dense sampler's outputs, from its affine map z -> y."""
    r = sampler.rank
    y = sampler.from_noise(np.vstack([np.zeros(r), np.eye(r)]))
    d = y[1:] - y[0]
    return d.T @ d, y[0]


# ------------------------------------------------------------------ the law


@pytest.mark.parametrize("name, args, kw", [
    ("uniform", (-1, 20), {"representation": "bernstein", "bnd": "left"}),
    ("lognormal", (0.0, 0.5), {"representation": "bernstein"}),
    ("gumbel", (1.0, 2.0), {}),  # quadrature points (values), dofs= picks some
    ("uniform", (0, 1), {"representation": "projection", "bnd": "left"}),
])
def test_a_draw_is_the_routes_draw(space, rng, name, args, kw):
    """For the same latent values, the dense sampler gives the current route's outputs: same points, map,
    representation and bnd=. With the exact latent covariance (next test) the laws are the same."""
    d = rng.distribution(name, *args, **kw)
    dofs = None if kw else np.arange(0, d.field.points.npoints, 7)
    a = d.dense(dofs=dofs, max_points=4000)
    w, route = route_draw(rng, d.field, 6)
    z, dist = noise_for(a, w)
    assert dist < (1e-11 if kw.get("representation") == "bernstein" else 1e-5)  # rank-deficient K: the cut
    np.testing.assert_allclose(a.from_noise(z), route[:, a.dofs], rtol=0, atol=max(1e-10, 10 * dist) *
                               np.abs(route).max())
    assert a.info["exact"] and a.info["latent"] == "points"


@pytest.mark.parametrize("rtol", [1e-12, None])
def test_latent_covariance_is_exact(space, rng, rtol):
    """F F^T is the standardised latent covariance D E Sigma E^T D at the latent points (here of a posterior). With
    rtol=None nothing is cut: F = R^T, one triangular matrix of the latent's size."""
    post = rng.gaussian().condition(points=[(0.5, 0.5)], noise=1e-2, data=[0.3])
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left", correlation=post)
    a = d.dense(rtol=rtol)
    s, f = a.law["latent_index"], a.law["factor"]
    pts = d.field.points
    e = pts.eval_full.tocsr()[s][:, post.space.free]
    dd = 1 / pts.std[s]
    k = dd[:, None] * (e @ (e @ sigma(post)).T) * dd[None, :]
    np.testing.assert_allclose(f @ f.T, k, rtol=0, atol=1e-11 * np.abs(k).max())
    if rtol is None:
        assert f.shape == (len(s), len(s)) and np.allclose(f, np.tril(f)) and a.info["truncation"]["rtol"] is None


def test_pinned_and_deterministic_points(space, rng):
    """Points with exactly zero variance get exactly zero rows without any option; pin= also pins points whose
    variance is at most pin times the largest (here conditioned to the noise level): exactly their mean, as the
    route's values there."""
    corners = np.array([[0.0, 0.0], [0.0, 1.0]])
    post = rng.gaussian().condition(points=corners, noise=1e-24, data=[0.3, -0.2])
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left", correlation=post)
    assert d.dense().info["deterministic"] == 0  # 1e-24: tiny, not zero
    a = d.dense(pin=1e-12)
    assert a.info["deterministic"] == 2
    pinned = np.flatnonzero(~a.law["factor"].any(axis=1))
    np.testing.assert_allclose(a.law["latent_coordinates"][pinned], corners, atol=1e-12)
    verts = [v.nr for v in space.mesh.vertices if min(np.linalg.norm(np.subtract(v.point, c)) for c in corners) < 1e-12]
    cols = [list(a.dofs).index(v) for v in verts]
    y = a.sample(200)
    assert np.ptp(y[:, cols], axis=0).max() == 0.0
    np.testing.assert_allclose(y[0, cols], d.sample(2)[0, verts], rtol=1e-10)
    dirichlet = ngs.H1(space.mesh, order=2, dirichlet="bottom")
    r2 = ngsrng.default_rng(dirichlet, seed=2, dtype="float64", correlation=CORR)
    g = r2.distribution("normal", 1.0, 2.0).dense(dofs=np.arange(dirichlet.ndof))
    fixed = np.flatnonzero(~np.array(dirichlet.FreeDofs()))
    assert g.info["deterministic"] == fixed.size and not g.law["factor"][fixed].any()


@pytest.mark.parametrize("order, dim, kind", [(3, 2, "H1"), (2, 3, "H1"), (2, 2, "L2")])
def test_support_is_exact(order, dim, kind):
    """The topological dependence pattern of the Bernstein conversion contains every entry of the computed
    conversion that is not rounding, and outside it the conversion is rounding."""
    mesh = ngs.Mesh((ngs.unit_square if dim == 2 else ngs.unit_cube).GenerateMesh(maxh=0.5))
    fes = (ngs.H1 if kind == "H1" else ngs.L2)(mesh, order=order)
    bern = BernsteinForm(SpaceInfo(fes))
    c, _ = bern.to_coefficients(np.eye(bern.npoints))
    pattern = bern.dependence(np.arange(fes.ndof)).toarray() > 0
    big = np.abs(c) > 1e-9 * np.abs(c).max()
    assert not (big & ~pattern).any()
    assert np.abs(c[~pattern]).max(initial=0.0) < 1e-11 * np.abs(c).max()
    if kind == "H1":
        assert pattern.sum(axis=1).min() == 1  # vertex coefficients: the vertex's point alone


def test_gaussian_fields_and_folding(space, rng):
    """Gaussian FE fields: the covariance of the outputs is O Sigma O^T exactly, with loc and scale (dofs; few
    observations: the observations are the latent; a dense map of the outputs: folded into the factor)."""
    gauss = rng.gaussian()
    cov = full_cov(gauss, space.ndof)
    d = rng.distribution("normal", 2.0, 3.0)
    loc = d.field.loc
    e = ngsrng.observation_operator(space, points=[(0.3, 0.4), (0.7, 0.2), (0.5, 0.5)])
    dense_e = np.random.default_rng(0).standard_normal((5, space.ndof))
    eye = sp.eye(space.ndof, format="csr")
    for kw, o, latent in (({"dofs": [0, 5, 9]}, eye[[0, 5, 9]], "coefficients"),
                          ({"dofs": [9, 0, 5]}, eye[[9, 0, 5]], "outputs"),  # a permutation: cheaper folded
                          ({"observe": e}, e, "outputs"), ({"observe": dense_e}, dense_e, "outputs")):
        a = d.dense(**kw)
        k, mean = affine_law(a)
        want = 9 * o @ cov @ o.T
        np.testing.assert_allclose(k, want, rtol=0, atol=1e-10 * np.abs(want).max())
        np.testing.assert_allclose(mean, o @ loc, atol=1e-10 * np.abs(o @ loc).max())
        assert a.info["latent"] == latent and a.rank <= a.latent_points
    trace = boundary_dofs(space, "left")
    mix = np.random.default_rng(1).standard_normal((3, len(trace)))
    folded = mix @ d.dense(dofs=trace)
    assert folded.info.get("folded") and folded.latent_points == 3 and folded.rank == 3
    o = mix @ sp.eye(space.ndof, format="csr")[trace]
    k, _ = affine_law(folded)
    np.testing.assert_allclose(k, 9 * o @ cov @ o.T, atol=1e-10 * np.abs(9 * o @ cov @ o.T).max())
    k, mean = affine_law(gauss.dense(dofs=[0, 5, 9]))
    np.testing.assert_allclose(k, cov[np.ix_([0, 5, 9], [0, 5, 9])], atol=1e-12)
    np.testing.assert_allclose(mean, gauss.mean[[0, 5, 9]], atol=1e-14)


def test_range(space, rng):
    """Bernstein outputs observed at points stay in the support (float64 and float32), and the trace values from the
    coefficients too."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    e = ngsrng.observation_operator(space, points=LEFT)
    for dtype in ("float64", "float32"):
        y = d.dense(observe=e, dtype=dtype).sample(3000)
        assert y.dtype == np.dtype(dtype) and -1.0 <= y.min() and y.max() <= 20.0
    a = d.dense()
    full = np.zeros((500, space.ndof))
    full[:, a.dofs] = a.sample(500)
    vals = basis_at(space, np.random.default_rng(1).uniform(0, 1, (200, 2)) * [0, 1]) @ full.T  # on the trace
    assert -1.0 <= vals.min() and vals.max() <= 20.0


def test_a_float32_sampler_multiplies_in_float32_and_draws_the_float64_samplers_values(space, rng):
    """A Bernstein (log-normal) law: the float32 sampler's dense product and closed-form map are float32 (no float64
    factor on the device), its pull and conversion float64, so from the same (host) noise it draws the float64
    sampler's values up to float32 rounding, and stays in the support."""
    d = rng.distribution("lognormal", 0.0, 1.0, representation="bernstein")
    s64, s32 = d.dense(dtype="float64", seed=3), d.dense(dtype="float32", seed=3)
    y64, y32 = s64.sample(400), s32.sample(400)
    assert y32.dtype == np.float32
    np.testing.assert_allclose(y32, y64, rtol=1e-4, atol=1e-5 * np.abs(y64).max())
    full = np.zeros((400, space.ndof))
    full[:, s32.dofs] = y32
    assert (basis_at(space, np.random.default_rng(1).uniform(0, 1, (500, 2))) @ full.T).min() > 0  # the field's values
    linear = [v for k, v in s32._device.items() if k[0] == "linear"]
    assert [lin["factor"].dtype for lin in linear] == [torch.float32]


@pytest.mark.parametrize("lengths", [[3, 1, 3, 2], [1, 0, 7, 2, 2, 30, 1], [5] * 6])
def test_output_maps_by_gathers(lengths):
    """The device form of an output map is v A^T: dense, one padded bucket, buckets by powers of two, empty rows."""
    gen = np.random.default_rng(0)
    s = 40
    rows = [sp.csr_matrix((gen.standard_normal(k), (np.zeros(k, int), gen.choice(s, k, replace=False))), shape=(1, s))
            for k in lengths]
    a = sp.vstack(rows).tocsr()
    v = gen.standard_normal((9, s))
    for dense_out in (0, 10**6):
        old = dense_module.DENSE_OUT
        dense_module.DENSE_OUT = dense_out
        try:
            y = dense_module._OutputMap(a, torch.device("cpu"), torch.float64)(torch.from_numpy(v)).numpy()
        finally:
            dense_module.DENSE_OUT = old
        np.testing.assert_allclose(y, v @ a.toarray().T, rtol=1e-14, atol=1e-14)


def test_marginals_match_the_route(space, rng):
    """End to end: the marginal at a trace vertex is uniform(-1, 20), as the route's (the joint law is checked exactly
    above: latent covariance and the route's map)."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    a = d.dense()
    assert a.dofs[0] < space.mesh.nv  # a vertex DoF (H1 numbers them first): its value has the exact marginal
    y = a.sample(20000)
    assert_law(y[:, 0], st.uniform(-1, 21).cdf, what="dense")
    assert_normal(st.norm.ppf((y[:, 0] + 1) / 21), 0.0, 1.0, what="the latent, back from the values")
    assert_same_law(y[:, 0], d.sample(4000)[:, a.dofs[0]], what="dense against the route")


# ------------------------------------------------------------------ observations: E @ d and E @ dense


@pytest.mark.parametrize("kind", ["numpy", "scipy", "torch"])
def test_observation_syntaxes(space, rng, kind):
    """(E @ d).dense() == d.dense(observe=E); E @ d.dense() composes without a new set-up and gives the same values
    from the same z (here E reads every latent value); E @ d draws E @ d.sample(); E of NumPy, SciPy or torch type."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    e = ngsrng.observation_operator(space, points=LEFT)
    a = d.dense()
    ek = {"numpy": e.toarray(), "scipy": e, "torch": torch.from_numpy(e.toarray())}[kind]
    direct, observed = d.dense(observe=e), ek @ d
    assert isinstance(observed, ngsrng.ObservedField) and observed.dense() is direct
    composed = ek[:, a.dofs] @ a  # the sampler's outputs are its dofs
    assert isinstance(composed, ngsrng.DenseSampler) and composed.latent_points == a.latent_points
    z = np.random.default_rng(0).standard_normal((20, a.rank))
    full = np.zeros((20, space.ndof))
    full[:, a.dofs] = a.from_noise(z)
    np.testing.assert_allclose(composed.from_noise(z), full @ e.T, rtol=0, atol=1e-12 * 20)
    np.testing.assert_allclose(direct.from_noise(z), full @ e.T, rtol=0, atol=1e-12 * 20)
    y = observed.sample(5)
    assert y.shape == (5, len(LEFT)) and -1 <= y.min() and y.max() <= 20
    drawtime = ek @ ngsrng.default_rng(space, dtype="float64", correlation=CORR).distribution(
        "uniform", -1, 20, representation="bernstein", bnd="left")
    torch.manual_seed(0)
    ref = drawtime.sample(6)
    buf = np.empty((6, len(LEFT)))
    drawtime.work_values = 3 * (space.ndof + len(LEFT))  # chunks of 3 samples, one draw
    torch.manual_seed(0)
    assert drawtime.sample(out=buf) is buf
    np.testing.assert_allclose(buf, ref, rtol=1e-12, atol=1e-12)
    m = len(LEFT)
    sel = {"numpy": np.eye(m)[:3], "scipy": sp.eye(m, format="csr")[:3], "torch": torch.eye(m, dtype=torch.float64)[:3]}
    two = sel[kind] @ observed  # observations of observations: (E2 E) @ d
    assert isinstance(two, ngsrng.ObservedField) and two.m == 3 and two.dense().nout == 3


def test_rounding_entries_of_observations(space, rng):
    """A point evaluation at the vertices gives rounding-level weights (~1e-17) to basis functions that vanish there:
    with zero= (default 1e-15) they are dropped and each vertex value reads its vertex point alone; zero=None keeps
    them (a larger latent set, the same law to rounding)."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein")
    xyz = np.array([v.point for v in space.mesh.vertices])[:12]
    e = ngsrng.observation_operator(space, points=xyz)
    a = d.dense(observe=e)
    assert a.latent_points == len(xyz) and a.info["zero"] == 1e-15
    b = d.dense(observe=e, zero=None)
    assert b.latent_points >= a.latent_points and b.info["zero_drop"] == 0.0
    assert_law(a.sample(10000)[:, 3], st.uniform(-1, 21).cdf)


def test_composition_drops_unread_values(space, rng):
    """E @ dense reading a few outputs keeps only the latent values they read, recompresses the factor, and keeps
    the law (a Gaussian field: covariance E O Sigma O^T E^T; a vertex value: the exact marginal)."""
    gauss = rng.gaussian()
    cov = full_cov(gauss, space.ndof)
    trace = boundary_dofs(space, "left")
    a = rng.distribution("normal").dense(dofs=trace)
    e = sp.csr_matrix((np.ones(2), ([0, 1], [0, 1])), shape=(2, len(trace)))  # the first two outputs
    b = e @ a
    assert b.latent_points == 2 and b.rank == 2
    o = e @ sp.eye(space.ndof, format="csr")[trace]
    k, _ = affine_law(b)
    np.testing.assert_allclose(k, o @ cov @ o.T, atol=1e-12)
    u = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left").dense()
    v = sp.csr_matrix((np.ones(1), ([0], [0])), shape=(1, len(u.dofs))) @ u  # a vertex coefficient: one point
    assert v.latent_points == 1
    assert_law(v.sample(10000)[:, 0], st.uniform(-1, 21).cdf)


def test_observed_gaussians(space, rng):
    """E @ (a Gaussian field or a Gaussian) is an ObservedGaussian; its dense() reuses its factor; E2 @ it
    composes (noise included)."""
    gauss = rng.gaussian()
    cov = full_cov(gauss, space.ndof)
    e = ngsrng.observation_operator(space, points=LEFT[::5])
    d = rng.distribution("normal", 1.0, 2.0)
    obs = e @ d  # the law, no set-up: pathwise draws until dense()
    assert isinstance(obs, ngsrng.ObservedGaussian) and isinstance(e @ gauss, ngsrng.ObservedGaussian)
    assert obs.method == "pathwise" and obs.sample(3).shape == (len(LEFT[::5]), 3)
    a = obs.dense()
    assert a.law["factor"].shape[0] == len(LEFT[::5])  # fewer observations than the DoFs they read: at them
    own = ngsrng.ObservedGaussian(gauss, points=LEFT[::5], noise=0.01)  # a factor of its own: reused
    assert own.dense().info.get("reused") and own.dense().law["factor"].shape == tuple(own._F64.shape)
    grid = ngsrng.observation_operator(space, points=np.random.default_rng(3).uniform(0, 1, (400, 2)))
    many = (grid @ d).dense()  # more observations than DoFs read: the latent at the DoFs, the same law
    assert many.latent_points < 400 and many.info["latent"] == "coefficients"
    k3, _ = affine_law(many)
    want = 4 * grid @ cov @ grid.T
    np.testing.assert_allclose(k3, want, atol=1e-10 * np.abs(want).max())
    k, mean = affine_law(a)
    np.testing.assert_allclose(k, 4 * e @ cov @ e.T, atol=1e-10 * np.abs(4 * e @ cov @ e.T).max())
    np.testing.assert_allclose(mean, e @ d.field.loc, atol=1e-12)
    noisy = ngsrng.ObservedGaussian(gauss, points=LEFT[::5], noise=0.1)
    e2 = np.random.default_rng(0).standard_normal((2, noisy.m))
    both = e2 @ noisy
    np.testing.assert_allclose(both.covariance(), e2 @ noisy.covariance() @ e2.T, rtol=1e-9)
    k2, _ = affine_law(both.dense(rtol=None))
    np.testing.assert_allclose(k2, both.covariance(), atol=1e-10 * np.abs(k2).max())


def test_correlated(space, rng):
    """Correlated components: at a vertex the two latents (back from the values) are jointly N(0, [[1, R], [R, 1]]);
    every component keeps its own conditioning (posteriors with different data), and E @ fields observes each."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    pair = ngsrng.correlated([d, d], 0.8)
    c = pair.dense()
    y = c.sample(8000)
    assert y.shape == (8000, 2, len(c.dofs)) and c.components == 2
    z = st.norm.ppf((y[:, :, 0] + 1) / 21)  # a vertex: the value is the latent's map
    assert_gaussian(z, [0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]])
    e = ngsrng.observation_operator(space, points=LEFT)
    obs = e @ pair
    assert isinstance(obs, ngsrng.ObservedField) and obs.dense().sample(3).shape == (3, 2, len(LEFT))
    assert len(obs.sample(2)) == 2
    prior = rng.gaussian()
    vert = min((v for v in space.mesh.vertices if abs(v.point[0]) < 1e-12), key=lambda v: abs(v.point[1] - 0.5))
    posts = [prior.condition(points=np.array([vert.point]), noise=1e-24, data=[t]) for t in (-1.0, 1.5)]
    ds = [rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left", correlation=p) for p in posts]
    joint = ngsrng.correlated(ds, 0.9)
    jd = joint.dense(pin=1e-12)
    col = list(jd.dofs).index(vert.nr)
    y = jd.sample(10)
    ref = joint.sample(2)
    for k in range(2):
        assert np.ptp(y[:, k, col]) == 0.0
        np.testing.assert_allclose(y[0, k, col], ref[k][0, vert.nr], rtol=1e-10)


# ------------------------------------------------------------------ randomness, caching, storage


def test_randomness(space, rng):
    """Own streams continue across calls (chunk invariant), generator= is read once per draw, seed None follows
    torch's global generator, host noise on the CPU and keyed device noise both chunk invariant."""
    a = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left").dense()
    for noise in ("host", "device"):
        def fresh():
            s = a.to(seed=5, noise=noise)
            s.block = 3  # several keyed blocks per call
            return s

        one, two, three = fresh(), fresh(), fresh()
        ref = one.sample(10)
        # the noise is bitwise chunk invariant, the products to rounding (other block sizes in the GEMMs)
        np.testing.assert_allclose(ref, np.vstack([two.sample(4), two.sample(6)]), rtol=1e-13, atol=1e-13)
        three.work_values = 3 * (2 * len(a.dofs) + 2 * a.rank)  # chunks of a few samples
        buf = np.empty((10, len(a.dofs)))
        assert three.sample(out=buf) is buf
        np.testing.assert_allclose(buf, ref, rtol=1e-13, atol=1e-13)
    g1, g2 = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    np.testing.assert_array_equal(a.sample(5, generator=g1), a.sample(5, generator=g2))
    assert not np.array_equal(a.sample(5, generator=g1), a.sample(5, generator=g2.manual_seed(5)))
    free = a.to(seed=Source())  # draw-time: torch's global generator (to(seed=None) keeps the stream)
    torch.manual_seed(3)
    x = free.sample(4)
    torch.manual_seed(3)
    np.testing.assert_array_equal(x, free.sample(4))
    assert a.sample().shape == (len(a.dofs),)
    assert isinstance(a.sample(2, output="torch"), torch.Tensor)


def test_cache_truncation_and_limits(space, rng):
    """Set up once per arguments (the same sampler again); rank= / tol= truncate (and say so); a clear error
    beyond max_points; other device settings share the set-up."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    a = d.dense()
    assert d.dense() is a and rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left").dense() is a
    assert a.to(dtype="float32").law is a.law
    t = d.dense(rank=3)
    assert t.rank == 3 and not t.info["exact"] and t.info["captured_variance"] < 1.0
    tt = d.dense(tol=0.05)
    assert tt.info["captured_variance"] >= 0.95 and tt.rank < a.rank
    with pytest.raises(ValueError, match="max_points"):
        rng.distribution("uniform", representation="bernstein").dense(max_points=50)


def test_set_ups_survive_pickling_and_spawn(space, rng, monkeypatch):
    """The set-up kept on a distribution survives pickling (its owner keeps it) but not a shallow copy; spawned
    children share it (with their own randomness), also after pickling."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    a = d.dense()
    child = rng.spawn(1)[0]
    clone = pickle.loads(pickle.dumps(d))
    unpickled_child = pickle.loads(pickle.dumps(child))
    monkeypatch.setattr(dense_module, "_build_law", lambda *args: pytest.fail("set up again"))
    np.testing.assert_array_equal(clone.dense().law["factor"], a.law["factor"])
    cd = child.distribution("uniform", -1, 20, representation="bernstein", bnd="left").dense()
    assert cd.law is a.law and cd.source is not a.source
    unpickled_child.distribution("uniform", -1, 20, representation="bernstein", bnd="left").dense()
    holder = type("Holder", (), {})()
    owned_cache(holder, "_c")["x"] = 1
    assert owned_cache(copy.copy(holder), "_c") == {} and owned_cache(holder, "_c") == {"x": 1}


def test_save_load(space, rng, tmp_path):
    """save / load is bit for bit for closed forms, Gaussian fields, inversions and NumPy functions; a NumPy
    reference sampler (the module docstring's) gives from_noise's values; a user function is refused."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    e = ngsrng.observation_operator(space, points=LEFT)
    cases = [ngsrng.correlated([d, d], 0.8).dense(observe=e),
             rng.distribution("lognormal", 0.0, 0.5, representation="bernstein").dense(),
             rng.distribution("normal", 1.0, 2.0).dense(dofs=[3, 1, 4]),
             rng.distribution("gamma", 2.0, representation="bernstein", bnd="left").dense(),
             rng.distribution("transformed", np.tanh, representation="bernstein", bnd="left").dense()]
    for k, sampler in enumerate(cases):
        path = tmp_path / f"s{k}.npz"
        sampler.save(path)
        loaded = ngsrng.DenseSampler.load(path, seed=7)
        np.testing.assert_array_equal(loaded.sample(20), sampler.to(seed=7).sample(20))
        assert loaded.info["provenance"] == sampler.info["provenance"]
        if k < 4:
            z = np.random.default_rng(0).standard_normal((30, sampler.components, sampler.rank))
            out = sampler.from_noise(z if sampler.components > 1 else z[:, 0])
            out = out if sampler.components > 1 else out[:, None]
            np.testing.assert_allclose(np.stack(reference(path, z), 1), out, rtol=0, atol=1e-7 * np.abs(out).max())
    with pytest.raises(NotImplementedError, match="user function"):
        rng.distribution("transformed", _square, representation="bernstein", bnd="left").dense().save(tmp_path / "x")


def _square(x):
    return x * x


def reference(path, z):
    """The module docstring's NumPy reference sampler (NumPy and SciPy, no ngsrng)."""
    s = np.load(path)
    meta = json.loads(str(s["meta"]))
    z = np.einsum("ij,njr->nir", s["mix"], z)
    out = []
    for i, c in enumerate(meta["components"]):
        w = s[f"mean{i}"] + z[:, i] @ s["factor"].T
        p = {k: s[v] if isinstance(v, str) else v for k, v in c["params"].items()}
        if c["map"] == "inversion":
            q = c["stored"]
            v = q["loc"] + q["scale"] * getattr(st, q["scipy"])(*q["args"], **q["kwds"]).ppf(ndtr(w))
        else:
            v = {"uniform": lambda w, low, high: low + (high - low) * ndtr(w),
                 "lognormal": lambda w, mean, sigma: np.exp(mean + sigma * w),
                 "affine": lambda w, shift, scale: shift + scale * w}[c["map"]](w, **p)
        if c["out_shape"]:
            a = np.zeros(c["out_shape"])
            a[s[f"out{i}_rows"], s[f"out{i}_cols"]] = s[f"out{i}_vals"]
            v = v @ a.T
        out.append(v)
    return out


def test_errors(space, rng):
    gamma = rng.distribution("gamma", 2.0, representation="bernstein", bnd="left")
    g = gamma.dense()  # numerical inversion: the route's values for the same latent values
    w, route = route_draw(rng, gamma.field, 3)
    z, dist = noise_for(g, w)
    assert dist < 1e-11
    np.testing.assert_allclose(g.from_noise(z), route[:, g.dofs], atol=1e-9 * np.abs(route).max())
    with pytest.raises(ValueError, match="not both"):
        gamma.dense(dofs=[1], observe=np.ones((1, space.ndof)))
    with pytest.raises(NotImplementedError, match="bnd_values"):
        rng.distribution("normal", bnd="left", bnd_values=np.zeros(len(boundary_dofs(space, "left")))).dense()
    with pytest.raises(NotImplementedError, match="several latents"):
        rng.distribution("chisquare", 2).dense(dofs=[0])
    with pytest.raises(ValueError, match="columns"):
        np.ones((2, 3)) @ gamma.dense()


@pytest.mark.skipif(not cuda_usable(), reason="no usable CUDA device")
def test_gpu(space, rng):
    """On a GPU: host noise gives the CPU's samples (to rounding), device noise is chunk invariant there too, and a
    torch matrix on the GPU composes."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    a = d.dense()
    gpu = a.to(device="cuda", seed=2, noise="host")
    y = gpu.sample(100)
    assert isinstance(y, torch.Tensor) and y.device.type == "cuda"
    np.testing.assert_allclose(y.cpu().numpy(), a.to(seed=2).sample(100), rtol=0, atol=1e-11)
    one, two = a.to(device="cuda", seed=3), a.to(device="cuda", seed=3)
    one.block = two.block = 3
    torch.testing.assert_close(one.sample(10), torch.cat([two.sample(4), two.sample(6)]), rtol=1e-13, atol=1e-13)
    e = torch.from_numpy(ngsrng.observation_operator(space, points=LEFT).toarray()).cuda()
    assert (e[:, a.dofs] @ gpu).sample(5).shape == (5, len(LEFT))
