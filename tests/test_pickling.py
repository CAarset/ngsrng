"""Every sampler pickles (spawn/forkserver worker processes, checkpoints), and an unpickled copy draws what the original
draws: the random state travels with it, and the NGSolve objects are rebuilt (refactorised: equal to rounding).

No lambda or nested function may be stored on a sampler (ngsrng.backend.Rebuildable); these tests catch one."""

import multiprocessing as mp
import pickle
import warnings

import ngsolve as ngs
import numpy as np
import pytest
import torch
from test_inplace import ARGS, KWARGS, VARIANTS
from torch.utils.data import DataLoader, Dataset

import ngsrng
from ngsrng.space import mass_form


@pytest.fixture(scope="module")
def fes():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2)


def twin(obj):
    return pickle.loads(pickle.dumps(obj))


def close(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10 * max(np.abs(a).max(), 1.0))


def _samplers(fes, seed):
    l2 = ngs.L2(fes.mesh, order=1, dgjumps=True)
    prior = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", seed=seed)
    obs = ngsrng.LinearObservations(prior, points=[(0.3, 0.4), (0.6, 0.5)], noise=0.1)
    kl = ngsrng.karhunen_loeve(prior, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "power 2": prior,
            "power 1.6 (rational)": ngsrng.Gaussian(fes, ngsrng.matern(5.0, 1.6), dtype="float64", seed=seed),
            "power 3": ngsrng.Gaussian(fes, ngsrng.matern(5.0, 3), dtype="float64", seed=seed),
            "cg": ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", seed=seed, method={"solver": "cg"}),
            "torch engine": ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), dtype="float64", seed=seed,
                                            method={"engine": "torch"}),
            "DG": ngsrng.Gaussian(l2, ngsrng.matern_dg(l2, 5.0, 2), dtype="float64", seed=seed),
            "assembled": ngsrng.Gaussian(fes, ngsrng.AssembledPrecision(mass_form), dtype="float64", seed=seed),
            "posterior (FE update)": prior.condition(points=[(0.3, 0.4)], noise=0.1, data=[1.0]),
            "posterior (noise update)": obs.posterior(data=[1.0, 0.0], form="noise"),
            "Matheron": ngsrng.MatheronPosterior(obs, data=[1.0, 0.0]),
            "KL": kl,
            "sample covariance": ngsrng.sample_covariance(prior, 4),
            "low-rank posterior": kl.condition(points=[(0.5, 0.5)], noise=0.1, data=[0.3]),
        }


@pytest.mark.parametrize("seed", [3, None])
def test_samplers_pickle(fes, seed):
    for name, g in _samplers(fes, seed).items():
        c = twin(g)
        torch.manual_seed(0)  # draw-time randomness (seed None); own streams ignore it
        a = g.sample(3).numpy()
        torch.manual_seed(0)
        b = c.sample(3).numpy()
        try:
            close(a, b)
        except AssertionError as err:
            raise AssertionError(f"{name}: the unpickled copy draws differently") from err


def _families():
    out = [(name, args, KWARGS.get(name, {})) for name, args in ARGS.items()]
    return out + list(VARIANTS)


@pytest.mark.parametrize("name, args, kw", _families(), ids=lambda v: v if isinstance(v, str) else "")
def test_generator_families_pickle(fes, name, args, kw):
    rng = ngsrng.default_rng(fes, seed=4, correlation=0.4, dtype="float64")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rng.sample(name, *args, size=1, **kw)  # build the field
        c = twin(rng)
        built = len(c._fields)
        close(rng.sample(name, *args, size=2, **kw), c.sample(name, *args, size=2, **kw))
    if name != "marginal":  # a scipy distribution is cached by identity: the copy holds its own copy of it
        assert len(c._fields) == built  # the unpickled field was found again (identity-keyed caches re-keyed)


def _spawned_draw(rng, queue):
    queue.put(rng.normal(size=2))


def test_a_spawned_process_continues_the_stream(fes):
    """A fresh interpreter (spawn) unpickles the generator, rebuilds the NGSolve objects and continues its stream."""
    rng = ngsrng.default_rng(fes, seed=4, correlation=0.4, dtype="float64")
    rng.normal()
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    p = ctx.Process(target=_spawned_draw, args=(rng, queue))
    p.start()
    p.join(120)
    assert p.exitcode == 0
    close(queue.get(), rng.normal(size=2))


class _Fields(Dataset):
    def __init__(self, rng):
        self.rng = rng

    def __len__(self):
        return 6

    def __getitem__(self, i):
        return self.rng.normal(size=1)[0]  # NumPy through the worker's pipe


def _first(batch):
    return batch[0]


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_dataloader_workers_draw_their_own_samples(fes, context):
    """An explicitly seeded generator copied (fork) or pickled (spawn) into DataLoader workers is re-keyed with each
    worker's seed (base seed + worker id): no two workers draw alike, and torch.manual_seed reproduces the epoch."""
    rng = ngsrng.default_rng(fes, seed=5, correlation=0.4, dtype="float64")
    rng.normal(size=2)  # built (and its thread pool used) in the main process

    def epoch(torch_seed):
        torch.manual_seed(torch_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # fork() of a multi-threaded process
            loader = DataLoader(_Fields(rng), batch_size=1, num_workers=3, multiprocessing_context=context,
                                collate_fn=_first, timeout=180)
            return np.stack(list(loader))

    a, b, c = epoch(0), epoch(0), epoch(1)
    rows = a.reshape(len(a), -1)
    assert len({r.tobytes() for r in rows}) == len(rows)  # every item differs: no worker repeats another's stream
    close(a, b)
    assert not np.allclose(a, c)
