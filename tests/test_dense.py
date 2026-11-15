"""Dense samplers (ngsrng.dense): the law of d.sample() at chosen DoFs or under an observation, by dense products.

The checks follow the invariants: the latent covariance is the exact discrete one (conditioning and pinned points
included), a draw is the current route's draw for the same latent values (so the laws agree to rounding), bounded
Bernstein families stay in range, observations compose, correlated components keep R, the stream conventions hold,
and stored samplers round-trip bit for bit, also without NGSolve and against a NumPy reference sampler."""

import json
import math
import pickle
import subprocess
import sys

import ngsolve as ngs
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.stats as st
import torch
from conftest import cuda_usable
from scipy.special import ndtr

import ngsrng
from ngsrng.boundary import boundary_dofs
from ngsrng.seeding import Source
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
    and the distance (zero up to rounding, or up to the eigenvalue cutoff when K is singular: more latent points than
    the Gaussian has dimensions)."""
    law = sampler.law
    f, mu = law["factor"], law["components"][i]["mean"]
    ws = w[law["latent_index"]]
    z, *_ = np.linalg.lstsq(f, ws - mu[:, None], rcond=None)
    return z.T, float(np.abs(f @ z + mu[:, None] - ws).max())


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
    assert dist < (1e-11 if kw.get("representation") == "bernstein" else 1e-5)  # rank-deficient K: the cutoff
    np.testing.assert_allclose(a.from_noise(z), route[:, a.dofs], rtol=0, atol=max(1e-10, 10 * dist) *
                               np.abs(route).max())
    assert a.info["exact"] and a.info["latent"] == "points"


def test_latent_covariance_is_exact(space, rng):
    """F F^T is the standardised latent covariance D E Sigma E^T D at the latent points, here of a posterior
    (conditioning included); the points conditioned without noise are pinned: exactly their mean, also in the
    outputs (the corner values), and equal to the route's values there."""
    prior = rng.gaussian()
    corners = np.array([[0.0, 0.0], [0.0, 1.0]])
    post = prior.condition(points=corners, noise=1e-24, data=[0.3, -0.2])
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left", correlation=post)
    a = d.dense()
    s, f = a.law["latent_index"], a.law["factor"]
    pts = d.field.points
    e = pts.eval_full.tocsr()[s][:, post.space.free]
    dd = np.where(np.isfinite(1 / pts.std[s]), 1 / pts.std[s], 0.0)
    k = dd[:, None] * (e @ (e @ sigma(post)).T) * dd[None, :]  # E Sigma E^T, standardised
    np.testing.assert_allclose(f @ f.T, k, rtol=0, atol=1e-10 * np.abs(k).max())
    assert a.info["pinned"] == 2
    pinned = np.flatnonzero(~f.any(axis=1))
    np.testing.assert_allclose(a.law["latent_coordinates"][pinned], corners, atol=1e-12)
    verts = [v.nr for v in space.mesh.vertices if min(np.linalg.norm(np.subtract(v.point, c)) for c in corners) < 1e-12]
    cols = [list(a.dofs).index(v) for v in verts]
    y = a.sample(200)
    assert np.ptp(y[:, cols], axis=0).max() == 0.0
    np.testing.assert_allclose(y[0, cols], d.sample(2)[0, verts], rtol=1e-12)


def test_gaussian_fields(space, rng):
    """Gaussian FE fields: the covariance of the outputs is O Sigma O^T exactly (dofs, an observation with fewer
    rows than the DoFs it reads: the observations are then the latent), with loc and scale; Gaussians themselves have
    dense() too."""
    gauss = rng.gaussian()
    cov = np.zeros((space.ndof, space.ndof))
    free = gauss.space.free
    cov[np.ix_(free, free)] = sigma(gauss)
    e = ngsrng.observation_operator(space, points=[(0.3, 0.4), (0.7, 0.2), (0.5, 0.5)])
    d = rng.distribution("normal", 2.0, 3.0)
    loc = d.field.loc
    for kw, o in (({"dofs": [9, 0, 5]}, sp.eye(space.ndof, format="csr")[[9, 0, 5]]), ({"observe": e}, e)):
        a = d.dense(**kw)
        k, mean = affine_law(a)
        np.testing.assert_allclose(k, 9 * o @ cov @ o.T, rtol=0, atol=1e-10 * np.abs(9 * o @ cov @ o.T).max())
        np.testing.assert_allclose(mean, o @ loc, atol=1e-12)
    assert d.dense(observe=e).info["latent"] == "outputs" and d.dense(dofs=[9, 0, 5]).info["latent"] == "coefficients"
    k, mean = affine_law(gauss.dense(dofs=[0, 5, 9]))
    np.testing.assert_allclose(k, cov[np.ix_([0, 5, 9], [0, 5, 9])], atol=1e-12)
    np.testing.assert_allclose(mean, gauss.mean[[0, 5, 9]], atol=1e-14)


def affine_law(sampler):
    """Covariance and mean of a Gaussian dense sampler's outputs, from its affine map z -> y."""
    r = sampler.rank
    y = sampler.from_noise(np.vstack([np.zeros(r), np.eye(r)]))
    d = y[1:] - y[0]
    return d.T @ d, y[0]


def test_range_and_composition(space, rng):
    """Bernstein outputs observed at points stay in the support (float64 and float32), and the observation
    composes: B(z) = E A(z) for the same z."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    a = d.dense()
    e = ngsrng.observation_operator(space, points=LEFT)
    b = d.dense(observe=e)
    assert b.rank == a.rank and np.array_equal(b.law["latent_index"], a.law["latent_index"])
    z = np.random.default_rng(0).standard_normal((50, a.rank))
    full = np.zeros((50, space.ndof))
    full[:, a.dofs] = a.from_noise(z)
    np.testing.assert_allclose(b.from_noise(z), full @ e.T, rtol=0, atol=1e-12 * 20)
    for dtype in ("float64", "float32"):
        y = d.dense(observe=e, dtype=dtype).sample(3000)
        assert y.dtype == np.dtype(dtype) and -1.0 <= y.min() and y.max() <= 20.0
    full = np.zeros((500, space.ndof))
    full[:, a.dofs] = a.sample(500)
    vals = basis_at(space, np.random.default_rng(1).uniform(0, 1, (200, 2)) * [0, 1]) @ full.T  # on the trace
    assert -1.0 <= vals.min() and vals.max() <= 20.0


def test_marginals_match_the_route(space, rng):
    """Moments and marginals of the dense draws against the route's (Kolmogorov-Smirnov at a trace vertex)."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    a = d.dense()
    i = 0  # a vertex DoF (H1 numbers them first): its value has the exact marginal
    assert a.dofs[0] < space.mesh.nv
    y = a.sample(4000)
    assert st.kstest(y[:, i], st.uniform(-1, 21).cdf).pvalue > 1e-3
    ref = d.sample(4000)[:, a.dofs]
    assert st.ks_2samp(y[:, i], ref[:, i]).pvalue > 1e-3
    np.testing.assert_allclose(np.cov(y.T), np.cov(ref.T), atol=0.1 * np.cov(ref.T).max())


def test_correlated(space, rng):
    """Correlated components: the latent correlation is R, Spearman's (6 / pi) arcsin(R / 2) for uniform values,
    and every component keeps its own conditioning (posteriors with different data)."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    c = ngsrng.correlated([d, d], 0.8).dense()
    y = c.sample(4000)
    assert y.shape == (4000, 2, len(c.dofs)) and c.components == 2
    z = st.norm.ppf((y[:, :, 0] + 1) / 21)  # a vertex: the value is the latent's map
    assert np.corrcoef(z.T)[0, 1] == pytest.approx(0.8, abs=0.03)
    assert st.spearmanr(y[:, 0, 0], y[:, 1, 0]).statistic == pytest.approx(6 / math.pi * math.asin(0.4), abs=0.03)
    prior = rng.gaussian()
    verts = [v for v in space.mesh.vertices if abs(v.point[0]) < 1e-12]
    vert = min(verts, key=lambda v: abs(v.point[1] - 0.5))  # a vertex on the trace
    x = np.array([vert.point])
    posts = [prior.condition(points=x, noise=1e-24, data=[t]) for t in (-1.0, 1.5)]
    ds = [rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left", correlation=p) for p in posts]
    pair = ngsrng.correlated(ds, 0.9)
    joint = pair.dense()
    col = list(joint.dofs).index(vert.nr)
    y = joint.sample(10)
    ref = pair.sample(2)
    for k in range(2):
        assert np.ptp(y[:, k, col]) == 0.0
        np.testing.assert_allclose(y[0, k, col], ref[k][0, vert.nr], rtol=1e-12)


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
    assert pickle.loads(pickle.dumps(a)).sample(2).shape == (2, len(a.dofs))


def test_errors_and_object_maps(space, rng):
    gamma = rng.distribution("gamma", 2.0, representation="bernstein", bnd="left")
    g = gamma.dense()  # numerical inversion: samples in the process
    full = np.zeros((2000, space.ndof))
    full[:, g.dofs] = g.sample(2000)
    assert (basis_at(space, LEFT) @ full.T).min() > 0  # the values on the trace (coefficients may be negative)
    w, route = route_draw(rng, gamma.field, 3)
    z, dist = noise_for(g, w)
    assert dist < 1e-11
    np.testing.assert_allclose(g.from_noise(z), route[:, g.dofs], atol=1e-9 * np.abs(route).max())
    with pytest.raises(NotImplementedError, match="not stored"):
        g.save("never.npz")
    with pytest.raises(ValueError, match="not both"):
        gamma.dense(dofs=[1], observe=np.ones((1, space.ndof)))
    with pytest.raises(NotImplementedError, match="bnd_values"):
        rng.distribution("normal", bnd="left", bnd_values=np.zeros(len(boundary_dofs(space, "left")))).dense()
    with pytest.raises(NotImplementedError, match="several latents"):
        rng.distribution("chisquare", 2).dense(dofs=[0])


def test_save_load(space, rng, tmp_path):
    """save / load is bit for bit; a NumPy reference sampler (the module docstring's) gives from_noise's values;
    load needs no NGSolve."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    e = ngsrng.observation_operator(space, points=LEFT)
    for k, sampler in enumerate([ngsrng.correlated([d, d], 0.8).dense(observe=e),
                                 rng.distribution("lognormal", 0.0, 0.5, representation="bernstein").dense(),
                                 rng.distribution("normal", 1.0, 2.0).dense(dofs=[3, 1, 4])]):
        path = tmp_path / f"s{k}.npz"
        sampler.save(path)
        loaded = ngsrng.DenseSampler.load(path, seed=7)
        np.testing.assert_array_equal(loaded.sample(20), sampler.to(seed=7).sample(20))
        assert loaded.info["provenance"] == sampler.info["provenance"]
        z = np.random.default_rng(0).standard_normal((30, sampler.components, sampler.rank))
        ref = reference(path, z)
        out = sampler.from_noise(z if sampler.components > 1 else z[:, 0])
        out = out if sampler.components > 1 else out[:, None]
        np.testing.assert_allclose(np.stack(ref, 1), out, rtol=0, atol=1e-7 * np.abs(out).max())
    script = ("import sys; sys.modules['ngsolve'] = None\nimport ngsrng\n"
              f"s = ngsrng.DenseSampler.load({str(tmp_path / 's0.npz')!r}, seed=1)\n"
              "assert 'ngsolve' not in [m.split('.')[0] for m in sys.modules if sys.modules[m] is not None]\n"
              "print(s.sample(3).shape)")
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == f"(3, 2, {len(LEFT)})"


def reference(path, z):
    """The module docstring's NumPy reference sampler (NumPy and SciPy, no ngsrng)."""
    s = np.load(path)
    meta = json.loads(str(s["meta"]))
    z = np.einsum("ij,njr->nir", s["mix"], z)
    out = []
    for i, c in enumerate(meta["components"]):
        w = s[f"mean{i}"] + z[:, i] @ s["factor"].T
        p = {k: s[v] if isinstance(v, str) else v for k, v in c["params"].items()}
        v = {"uniform": lambda w, low, high: low + (high - low) * ndtr(w),
             "lognormal": lambda w, mean, sigma: np.exp(mean + sigma * w),
             "affine": lambda w, shift, scale: shift + scale * w}[c["map"]](w, **p)
        if c["out_shape"]:
            a = np.zeros(c["out_shape"])
            a[s[f"out{i}_rows"], s[f"out{i}_cols"]] = s[f"out{i}_vals"]
            v = v @ a.T
        out.append(v)
    return out


@pytest.mark.skipif(not cuda_usable(), reason="no usable CUDA device")
def test_gpu(space, rng):
    """On a GPU: host noise gives the CPU's samples (to rounding), device noise is chunk invariant there too."""
    d = rng.distribution("uniform", -1, 20, representation="bernstein", bnd="left")
    a = d.dense()
    gpu = a.to(device="cuda", seed=2, noise="host")
    y = gpu.sample(100)
    assert isinstance(y, torch.Tensor) and y.device.type == "cuda"
    np.testing.assert_allclose(y.cpu().numpy(), a.to(seed=2).sample(100), rtol=0, atol=1e-11)
    one, two = a.to(device="cuda", seed=3), a.to(device="cuda", seed=3)
    one.block = two.block = 3
    torch.testing.assert_close(one.sample(10), torch.cat([two.sample(4), two.sample(6)]), rtol=1e-13, atol=1e-13)
