"""Seeding conventions (ngsrng.seeding): seed=None follows torch's global random state at every draw (torch.randn's
convention, so torch.manual_seed / Lightning's seed_everything reproduce ngsrng); generator objects are read at every
draw; ints and SeedSequences give own streams (numpy's convention); derived samplers inherit.

Separately built samplers, and a sampler's first draw, may differ from one another at rounding level (NGSolve's
sparse Cholesky rounds differently from one factorisation to the next, lazy set-up on the first draw), so those
comparisons are to rounding; repeated draws of one sampler are compared bit for bit."""

import warnings

import ngsolve as ngs
import numpy as np
import pytest
import torch

import ngsrng
from ngsrng.seeding import Source

FAMILIES = [("normal", (), {}), ("beta", (2.0, 3.0), {}), ("standard_t", (4,), {}),
            ("normal_inverse_gaussian", (0.5,), {}), ("karhunen_loeve", ("uniform",), {"rank": 8}),
            ("wishart", (3, np.eye(2)), {"construction": "bartlett", "method": "table"})]


@pytest.fixture(scope="module")
def fes():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.15)), order=2)


def close(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.allclose(a, b, rtol=0, atol=1e-12 * max(np.abs(a).max(), 1.0))


def draw(rng, name, args, kw, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rng.sample(name, *args, **kw, **extra)


@pytest.mark.parametrize("name, args, kw", FAMILIES)
def test_default_follows_torch_manual_seed(fes, name, args, kw):
    rng = ngsrng.default_rng(fes, correlation=0.3, dtype="float64", work_values=200)  # small chunks for out=
    draw(rng, name, args, kw, size=2)  # first draw: lazy set-up
    torch.manual_seed(0)
    a = draw(rng, name, args, kw, size=6)
    c = draw(rng, name, args, kw, size=6)
    torch.manual_seed(0)
    b = draw(rng, name, args, kw, size=6)
    np.testing.assert_array_equal(a, b)  # reseeding torch reproduces the draw, after construction too
    assert not close(a, c)  # without reseeding: the next draw
    torch.manual_seed(0)
    out = np.empty_like(a)
    draw(rng, name, args, kw, out=out)
    assert close(out, a)  # one draw: chunked in-place fills give the allocating call's values (float64: to rounding,
    # where products act on whole chunks; the in-place contract, ngsrng.inplace)


def test_a_draw_takes_one_key_from_torch(fes):
    """Every draw advances torch's global generator by the same amount, whatever its size; reseeding before
    construction reproduces a freshly built sampler too."""
    rng = ngsrng.default_rng(fes, correlation=0.3, dtype="float64")
    after = []
    for size in (1, 7):
        torch.manual_seed(0)
        rng.normal(size=size)
        after.append(torch.rand(1).item())
    assert after[0] == after[1]
    xs = []
    for _ in range(2):
        torch.manual_seed(3)
        xs.append(ngsrng.default_rng(fes, correlation=0.3, dtype="float64").beta(2.0, 3.0, size=3))
    assert close(*xs)


def test_explicit_seed_is_an_own_stream(fes):
    """An int seed neither depends on nor advances torch's global state; equal seeds give equal streams."""
    a = ngsrng.default_rng(fes, seed=5, correlation=0.3, dtype="float64")
    b = ngsrng.default_rng(fes, seed=5, correlation=0.3, dtype="float64")
    torch.manual_seed(1)
    x = a.standard_t(4, size=3)
    torch.manual_seed(2)
    y = b.standard_t(4, size=3)
    assert close(x, y)
    torch.manual_seed(0)
    a.normal(size=3)
    t = torch.rand(1).item()
    torch.manual_seed(0)
    assert torch.rand(1).item() == t
    assert not close(a.normal(size=3), a.normal(size=3))  # the stream continues
    s1, s2 = a.spawn(2)
    assert not close(s1.normal(size=2), s2.normal(size=2))


@pytest.mark.parametrize("kind", ["torch", "numpy"])
def test_generator_objects_are_read_at_draw_time(fes, kind):
    gen = torch.Generator().manual_seed(3) if kind == "torch" else np.random.default_rng(3)
    rng = ngsrng.default_rng(fes, seed=gen, correlation=0.3, dtype="float64")
    rng.normal(size=1)
    if kind == "torch":
        gen.manual_seed(3)
    else:
        gen.bit_generator.state = np.random.default_rng(3).bit_generator.state
    a = rng.normal(size=3)
    if kind == "torch":
        gen.manual_seed(3)
    else:
        gen.bit_generator.state = np.random.default_rng(3).bit_generator.state
    np.testing.assert_array_equal(rng.normal(size=3), a)


@pytest.mark.parametrize("seed", [None, 7])
def test_derived_samplers_inherit(fes, seed):
    """Posteriors (FE-space and noise-space updates, Matheron), KL expansions, sample covariances and low-rank
    posteriors follow their prior's randomness: identical scripts give identical samples. (Drawing the ensemble of a
    sample covariance is a draw, hence the torch seed at the start.) KL samples agree to rounding divided by the
    eigenvalue gaps: rounding decides the basis within (near-)degenerate eigenspaces, here a pair 1e-8 apart."""

    def script():
        torch.manual_seed(1)
        prior = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", seed=seed)
        obs = ngsrng.LinearObservations(prior, points=[(0.3, 0.4), (0.6, 0.5)], noise=0.1)
        kl = ngsrng.karhunen_loeve(prior, 6)
        ens = ngsrng.sample_covariance(prior, 5)
        samplers = [prior.condition(points=[(0.3, 0.4)], noise=0.1, data=[1.0]),
                    obs.posterior(data=[1.0, 0.0], form="noise"), ngsrng.MatheronPosterior(obs, data=[1.0, 0.0]),
                    kl, ens, kl.condition(points=[(0.5, 0.5)], noise=0.1, data=[0.3])]
        torch.manual_seed(0)
        return [s.sample(3).numpy() for s in samplers]

    for k, (x, y) in enumerate(zip(script(), script())):
        if k in (3, 5):  # KL samplers
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-8 * np.abs(x).max())
        else:
            assert close(x, y)


def test_kl_modes_are_deterministic(fes):
    """The range finder has a fixed seed of its own: the modes do not depend on the sampler's randomness (nor
    consume it), whatever the sampler's seed."""
    prior = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64")
    k1, k2 = ngsrng.karhunen_loeve(prior, 6), ngsrng.karhunen_loeve(prior, 6, seed=3)
    np.testing.assert_array_equal(k1.modes, k2.modes)
    assert k1.source.draw_time and not k2.source.draw_time


def test_the_plan_names_the_randomness(fes):
    assert "torch's global generator" in ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2)).plan["randomness"]
    assert "own stream" in ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), seed=1).plan["randomness"]


@pytest.mark.parametrize("bad", ["1", 1.5, True, object()])
def test_bad_seeds_raise(bad):
    with pytest.raises(TypeError):
        Source(bad)


def _child_draw(gauss, torch_seed, queue):
    torch.manual_seed(torch_seed)  # what a DataLoader worker does: torch reseeded per process
    queue.put(gauss.sample(2).numpy())


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork")
def test_forked_processes_sample(fes):
    """A child forked after the parent sampled with threads draws threaded batches (the parent's pool threads do not
    exist in the child: ngsrng starts new pools there), and follows the child's torch seed."""
    import multiprocessing as mp

    g = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", method={"threads": 4})
    g.sample(4)  # the parent's pool now has 4 idle threads; a child batch of 2 used to wait for them forever
    ctx = mp.get_context("fork")
    out = []
    for torch_seed in (1, 2, 1):
        queue = ctx.SimpleQueue()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # "multi-threaded process, fork() may deadlock"
            p = ctx.Process(target=_child_draw, args=(g, torch_seed, queue))
            p.start()
        p.join(60)
        if p.is_alive():
            p.kill()
            pytest.fail("the forked child hung")
        assert p.exitcode == 0
        out.append(queue.get())
    np.testing.assert_array_equal(out[0], out[2])
    assert not close(out[0], out[1])


def test_single_draws_repeat_bit_for_bit(fes):
    """A single draw is computed like a batch column (single-threaded, on a pool thread): it repeats exactly, equals
    the first column of a batch, and is not affected by a caller's TaskManager block (NGSolve's task parallelism adds
    in a varying order)."""
    g = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64")
    g.method["threads"] = 4  # the module's small mesh would otherwise use one thread for batches

    def single():
        torch.manual_seed(0)
        return g.sample().vec.FV().NumPy().copy()

    first = single()
    np.testing.assert_array_equal(single(), first)
    torch.manual_seed(0)
    np.testing.assert_array_equal(g.sample(3).numpy()[:, 0], first)
    with ngs.TaskManager():
        np.testing.assert_array_equal(single(), first)
    rng = ngsrng.default_rng(fes, correlation=0.3, dtype="float64")
    for name in ("normal", "lognormal"):
        draws = []
        for _ in range(2):
            torch.manual_seed(1)
            draws.append(rng.sample(name).vec.FV().NumPy().copy())
        if name == "normal":
            np.testing.assert_array_equal(*draws)
        else:  # a pointwise transform after the Gaussian: once seen to differ by 1 ulp in 3 of 2450 values (three
            # runs on 2026-11-03, not reproduced in ~45 since; cause not found), hence to rounding
            np.testing.assert_allclose(*draws, rtol=1e-15, atol=0)
