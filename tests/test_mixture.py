"""Finite mixtures (ngsrng.mixture): labels with the weights' law, each draw its law's draw, dense samplers of several
laws (one product per law, per-law truncation), storage of several laws and the reference reader, observations.

Draws are checked by common random numbers (the same streams through two routes); the labels' law by an exact
distribution-free bound (stat_checks)."""

import ngsolve as ngs
import numpy as np
import pytest
import torch
from conftest import cuda_usable
from scipy.special import ndtr
from stat_checks import assert_law
from test_dense import reference

import ngsrng

LENGTHS = (0.2, 0.35, 0.6)
WEIGHTS = (0.2, 0.5, 0.3)


@pytest.fixture(scope="module")
def space():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2)


def laws(space, name="uniform", args=(-1.0, 2.0), seed=10, **kw):
    """One distribution per correlation length, from one generator (its own stream; the laws share its set-up)."""
    rng = ngsrng.default_rng(space, seed=seed, dtype="float64")
    return [rng.distribution(name, *args, correlation=ngsrng.matern_correlation(length, nu=2), **kw)
            for length in LENGTHS]


def categorical_cdf(w):
    c = np.concatenate([np.cumsum(w), [1.0]])
    return (lambda t: c[np.clip(np.floor(t).astype(int), 0, len(w))] * (t >= 0),
            lambda t: np.where(t > 0, c[np.clip(np.ceil(t).astype(int) - 1, 0, len(w))], 0.0))


def test_labels_follow_the_weights(space):
    """The labels are i.i.d. with probabilities the normalised weights, pathwise and dense."""
    mix = ngsrng.mixture(laws(space), [2, 5, 3], seed=1)
    np.testing.assert_allclose(mix.weights, WEIGHTS)
    cdf, left = categorical_cdf(np.asarray(WEIGHTS))
    _, lab = mix.sample(4000, labels=True)
    assert lab.dtype == np.int64 and lab.shape == (4000,)
    assert_law(lab, cdf, left_cdf=left, what="pathwise labels")
    _, lab = mix.dense().sample(20000, labels=True)
    assert_law(lab, cdf, left_cdf=left, what="dense labels")


def test_each_draw_is_its_laws_draw(space):
    """Common random numbers: the draws with label l are the next draws of law l, as law l alone gives them (its own
    stream); a batch split in two gives the same labels and draws; out= fills in place."""
    mix = ngsrng.mixture(laws(space, representation="bernstein"), WEIGHTS, seed=3)
    x, lab = mix.sample(60, labels=True)
    for l, fresh in enumerate(laws(space, representation="bernstein")):
        np.testing.assert_allclose(x[lab == l], fresh.sample(int((lab == l).sum())), rtol=1e-13, atol=1e-13)
    again = ngsrng.mixture(laws(space, representation="bernstein"), WEIGHTS, seed=3)
    a, la = again.sample(25, labels=True)
    b, lb = again.sample(35, labels=True)
    np.testing.assert_array_equal(np.concatenate([la, lb]), lab)
    np.testing.assert_allclose(np.vstack([a, b]), x, rtol=1e-13, atol=1e-13)
    third = ngsrng.mixture(laws(space, representation="bernstein"), WEIGHTS, seed=3)
    buf = np.empty_like(x)
    assert third.sample(out=buf) is buf
    np.testing.assert_allclose(buf, x, rtol=1e-13, atol=1e-13)
    one = mix.sample()
    assert isinstance(one, ngs.GridFunction)


def test_dense_mixture_is_the_laws_dense_samplers(space):
    """The dense sampler of a mixture holds every law's dense law (truncated relative to its own variance) and, from
    the same z and labels, gives each law's own dense sampler's values (a law with a smaller rank reads the leading
    columns of z)."""
    ds = laws(space, representation="bernstein")
    mix = ngsrng.mixture(ds, WEIGHTS, seed=4)
    a = mix.dense(tol=1e-4)
    own = [d.dense(tol=1e-4) for d in ds]
    assert [law["factor"].shape[1] for law in a.laws] == [s.rank for s in own] and len(set(s.rank for s in own)) > 1
    assert a.rank == max(s.rank for s in own) and mix.dense(tol=1e-4) is a
    z = np.random.default_rng(0).standard_normal((40, a.rank))
    lab = np.random.default_rng(1).integers(0, 3, 40)
    y = a.from_noise(z, lab)
    for l, s in enumerate(own):
        np.testing.assert_allclose(y[lab == l], s.from_noise(z[lab == l, :s.rank]), rtol=0, atol=1e-12)
    assert len(a._stages) == 1  # one family and representation: one conversion for every law
    with pytest.raises(ValueError, match="labels"):
        a.from_noise(z)
    other = laws(space, "lognormal", (0.0, 0.5), seed=20, representation="bernstein")
    mixed = [ds[0], other[1], ds[2]]
    b = ngsrng.mixture(mixed, WEIGHTS, seed=4).dense()
    assert sorted(map(sorted, b._stages)) == [[0, 2], [1]]
    y = b.from_noise(np.random.default_rng(0).standard_normal((40, b.rank)), lab)
    for l, d in enumerate(mixed):
        s = d.dense()
        np.testing.assert_allclose(y[lab == l], s.from_noise(
            np.random.default_rng(0).standard_normal((40, b.rank))[lab == l, :s.rank]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("noise", ["host", "device"])
def test_dense_mixture_streams(space, noise):
    """Labels and draws of a dense mixture do not depend on how a batch is split; out= and labels=True agree with the
    allocating call; the labels' draws are the laws' draws from the same z."""
    a = ngsrng.mixture(laws(space, representation="bernstein"), WEIGHTS, seed=5).dense()

    def fresh():
        s = a.to(seed=7, noise=noise)
        s.block = 4  # several keyed blocks per call
        return s

    one, two, three = fresh(), fresh(), fresh()
    y, lab = one.sample(30, labels=True)
    y1, l1 = two.sample(11, labels=True)
    y2, l2 = two.sample(19, labels=True)
    np.testing.assert_array_equal(np.concatenate([l1, l2]), lab)
    np.testing.assert_allclose(np.vstack([y1, y2]), y, rtol=1e-13, atol=1e-13)
    three.work_values = 3 * (6 * a.latent_points + 2 * a.rank)  # chunks of a few samples
    buf = np.empty_like(y)
    out, lb = three.sample(out=buf, labels=True)
    assert out is buf
    np.testing.assert_array_equal(lb, lab)
    np.testing.assert_allclose(buf, y, rtol=1e-13, atol=1e-13)


def test_one_law(space):
    """A mixture of one law draws as the law's dense sampler (no labels drawn), with labels zero."""
    d = laws(space)[0]
    a = ngsrng.mixture([d]).dense(seed=2)
    b = d.dense(seed=2)
    np.testing.assert_array_equal(a.sample(10), b.sample(10))
    assert not a.sample(3, labels=True)[1].any()


def test_save_load_and_reference_reader(space, tmp_path):
    """A dense mixture round-trips bit for bit, and the module docstring's NumPy reference reader gives from_noise's
    values from the same z and labels."""
    a = ngsrng.mixture(laws(space, representation="bernstein"), WEIGHTS, seed=6).dense(tol=1e-4)
    path = tmp_path / "mix.npz"
    a.save(path)
    b = ngsrng.DenseSampler.load(path, seed=8)
    np.testing.assert_array_equal(b.weights, a.weights)
    np.testing.assert_array_equal(b.sample(20, labels=True)[0], a.to(seed=8).sample(20, labels=True)[0])
    z = np.random.default_rng(0).standard_normal((30, 1, a.rank))
    lab = np.random.default_rng(1).integers(0, 3, 30)
    y = a.from_noise(z, lab)
    np.testing.assert_allclose(reference(path, z, lab)[0], y, rtol=0, atol=1e-7 * np.abs(y).max())


def test_observations(space):
    """E @ mixture is the mixture of the observed laws; E @ (the dense mixture) gives E applied to its draws from the
    same z and labels (E reads every latent value: no recompression); the observed mixture's dense sampler draws."""
    e = np.random.default_rng(2).standard_normal((5, space.ndof))
    mix = ngsrng.mixture(laws(space, "lognormal", (0.0, 0.5), representation="bernstein"), WEIGHTS, seed=9)
    obs = e @ mix
    assert isinstance(obs, ngsrng.Mixture) and obs.sample(4).shape == (4, 5)
    a = mix.dense()
    composed = e @ a
    z = np.random.default_rng(0).standard_normal((20, a.rank))
    lab = np.random.default_rng(1).integers(0, 3, 20)
    np.testing.assert_allclose(composed.from_noise(z, lab), a.from_noise(z, lab) @ e.T, rtol=1e-10, atol=1e-10)
    assert obs.dense().sample(4).shape == (4, 5)


def test_gaussian_laws(space):
    """A mixture of Gaussians draws (ndof, n) as they do; its dense sampler draws (n, ndof) rows, each draw an affine
    image of z with its law's mean and covariance."""
    gs = [ngsrng.Gaussian(space, ngsrng.matern(k, 2), seed=s) for s, k in ((1, 4.0), (2, 9.0))]
    mix = ngsrng.mixture(gs, seed=1)
    x = mix.sample(6)
    assert tuple(x.shape) == (space.ndof, 6)
    buf = torch.empty((space.ndof, 6), dtype=torch.float64)
    mix.sample(6, out=buf)
    a = mix.dense()
    lab = np.array([0, 1])
    z0 = a.from_noise(np.zeros((2, a.rank)), lab)
    z1 = a.from_noise(np.vstack([np.eye(a.rank)[:1]] * 2), lab)
    for l, g in enumerate(gs):
        np.testing.assert_allclose(z0[l], g.mean, atol=1e-12)
        np.testing.assert_allclose(z1[l] - z0[l], g.dense().from_noise(np.eye(g.dense().rank)[:1])[0] - g.mean,
                                   atol=1e-12)


def test_errors(space):
    d = laws(space)
    with pytest.raises(ValueError, match="weights"):
        ngsrng.mixture(d, [1, -1, 1])
    with pytest.raises(ValueError, match="weights"):
        ngsrng.mixture(d, [1, 1])
    with pytest.raises(ValueError, match="same kind"):
        ngsrng.mixture([d[0], ngsrng.Gaussian(space, ngsrng.matern(4.0, 2))])
    with pytest.raises(TypeError):
        ngsrng.mixture([d[0], 3])
    other = ngsrng.default_rng(space, seed=0, correlation=0.3).distribution("uniform", bnd="left")
    with pytest.raises(ValueError, match="output sizes"):
        ngsrng.mixture([d[0], other]).dense()


@pytest.mark.skipif(not cuda_usable(), reason="no usable CUDA device")
def test_gpu(space):
    """On a GPU: host noise gives the CPU's draws and labels (to rounding), device noise is chunk invariant there, and
    labels come back as a tensor on the device."""
    a = ngsrng.mixture(laws(space, representation="bernstein"), WEIGHTS, seed=5).dense()
    y, lab = a.to(device="cuda", seed=2, noise="host").sample(50, labels=True)
    yc, lc = a.to(seed=2).sample(50, labels=True)
    assert isinstance(lab, torch.Tensor) and lab.device.type == "cuda"
    np.testing.assert_array_equal(lab.cpu().numpy(), lc)
    np.testing.assert_allclose(y.cpu().numpy(), yc, rtol=0, atol=1e-11)
    one, two = a.to(device="cuda", seed=3), a.to(device="cuda", seed=3)
    one.block = two.block = 4
    torch.testing.assert_close(one.sample(10), torch.cat([two.sample(4), two.sample(6)]), rtol=1e-13, atol=1e-13)
    assert ndtr(0.0) == 0.5  # scipy present for the reference reader
