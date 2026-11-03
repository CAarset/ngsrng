"""The in-place contract (ngsrng.inplace): every sampler, current and future, fills preallocated buffers with the
values the allocating call returns, in chunks whose memory does not grow with the batch."""

import importlib
import inspect
import pkgutil
import warnings

import ngsolve as ngs
import numpy as np
import pytest
import scipy.stats as st
import torch

import ngsrng
from ngsrng.distributions.transforms import CLOSED, SCIPY
from ngsrng.generator import SPECIAL
from ngsrng.inplace import InPlace

# Arguments for every registered family. A new family must be added here (test_every_family_is_covered).
ARGS = {
    "normal": (), "uniform": (), "lognormal": (0.0, 0.5), "exponential": (), "weibull": (1.5,), "laplace": (),
    "gumbel": (), "logistic": (), "rayleigh": (), "pareto": (3.0,), "power": (2.0,), "standard_cauchy": (),
    "triangular": (0.0, 0.3, 1.0), "halfnormal": (),
    "gamma": (2.0,), "beta": (2.0, 5.0), "chisquare": (3,), "noncentral_chisquare": (3.0, 1.0), "standard_t": (4.0,),
    "f": (5.0, 7.0), "noncentral_f": (5.0, 7.0, 1.0), "wald": (1.0, 2.0), "vonmises": (0.0, 2.0), "poisson": (3.0,),
    "binomial": (10, 0.3), "bernoulli": (0.3,), "negative_binomial": (5, 0.5), "geometric": (0.3,),
    "logseries": (0.6,), "zipf": (2.5,), "hypergeometric": (10, 8, 5),
    "standard_normal": (), "white_noise": (), "dirichlet": ([1.0, 2.0, 3.0],), "wishart": (3, np.eye(2)),
    "inverse_wishart": (4, np.eye(2)), "matrix_lognormal": (2,), "normal_inverse_gaussian": (), "variance_gamma": (),
    "indicator": (0.5,), "transformed": (np.tanh,), "marginal": (st.genextreme(0.2),), "standard_exponential": (),
    "standard_gamma": (2.0,), "karhunen_loeve": ("uniform",),
}
KWARGS = {"karhunen_loeve": {"rank": 12}}
# other constructions and representations of the same families
VARIANTS = [
    ("normal", (1.0, 0.5), {}), ("normal", (1.0, 0.5), {"points": [[0.2, 0.3], [0.7, 0.6]]}),
    ("normal", (), {"representation": "quadrature"}), ("chisquare", (2.5,), {}),
    ("standard_t", (4.0,), {"construction": "copula"}), ("wishart", (2.5, np.eye(2)), {"construction": "bartlett"}),
    ("uniform", (), {"representation": "projection"}), ("uniform", (), {"representation": "bernstein"}),
    ("gamma", (2.0,), {"points": [[0.2, 0.3], [0.7, 0.6], [0.5, 0.5]]}),
    ("gamma", (2.0,), {"method": "table"}), ("poisson", (3.0,), {"method": "discrete-table"}),
]
SMALL_WORK = 200  # values per chunk: forces fills of a few samples per chunk


@pytest.fixture(scope="module")
def fes():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.35)), order=1)


@pytest.fixture(autouse=True)
def one_sample_per_chunk_minimum(monkeypatch):
    monkeypatch.setattr("ngsrng.inplace.available_threads", lambda: 1)  # small chunks also on the CPU engine


def _check_out_equals_returned(fes, name, args, kw, size=(3, 2)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = ngsrng.default_rng(fes, seed=4, correlation=0.4).sample(name, *args, size=size, **kw)
        ref = np.asarray(ref)
        buf = np.empty_like(ref)
        rng = ngsrng.default_rng(fes, seed=4, correlation=0.4, work_values=SMALL_WORK)
        got = rng.sample(name, *args, size=size, out=buf, **kw)
    assert got is buf
    if ref.dtype == np.float64:  # FE coefficients (bernstein): batched solves and products round differently
        np.testing.assert_allclose(buf, ref, rtol=1e-12, atol=1e-14 * np.abs(ref).max())
    else:  # float32 and integer outputs: bit for bit
        np.testing.assert_array_equal(buf, ref)
    field = next(iter(rng._fields.values()))[1]
    assert isinstance(field, InPlace)


def test_every_family_is_covered():
    assert set(CLOSED) | set(SCIPY) | set(SPECIAL) <= set(ARGS), "add arguments for the new family to ARGS"


@pytest.mark.parametrize("name", sorted(ARGS))
def test_every_family_fills_in_place(fes, name):
    """Chunked in-place fills (a few samples per chunk) equal the allocating call (CPU engine)."""
    _check_out_equals_returned(fes, name, ARGS[name], KWARGS.get(name, {}))


@pytest.mark.parametrize("name,args,kw", VARIANTS, ids=[f"{v[0]}-{'-'.join(map(str, v[2]))}" for v in VARIANTS])
def test_every_variant_fills_in_place(fes, name, args, kw):
    _check_out_equals_returned(fes, name, args, kw)


def test_every_sampler_class_has_an_in_place_form():
    """Any class in the package with a ``sample`` method implements the contract (FieldDistribution and Generator
    forward ``out=`` to it)."""
    missing = []
    for mod in pkgutil.walk_packages(ngsrng.__path__, "ngsrng."):
        module = importlib.import_module(mod.name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or name in ("Generator", "FieldDistribution"):
                continue
            if callable(getattr(cls, "sample", None)) and not issubclass(cls, InPlace):
                missing.append(f"{cls.__module__}.{name}")
    assert not missing, f"samplers without an in-place form: {missing}"


@pytest.mark.parametrize("engine", ["ngsolve", "torch"])
def test_gaussian_sample_out(fes, engine):
    def prior():
        return ngsrng.Gaussian(fes, ngsrng.matern(5.0), method={"engine": engine}, seed=1)

    post_of = lambda g: g.condition(points=[[0.5, 0.5]], noise=1e-3, data=[1.0], seed=2)  # noqa: E731
    for make in (prior, lambda: post_of(prior())):
        ref = make().sample(9).numpy()
        for buf in (np.empty((fes.ndof, 9), np.float32), torch.empty(fes.ndof, 9, dtype=torch.float64)):
            got = make().sample(9, out=buf)
            assert got is buf
            np.testing.assert_allclose(np.asarray(buf, dtype=np.float32), ref, rtol=0, atol=0)


def test_low_rank_and_matheron_sample_out(fes):
    prior = ngsrng.Gaussian(fes, ngsrng.matern(5.0), dtype="float64", seed=1)
    kl = lambda: ngsrng.karhunen_loeve(prior, 8, seed=3)  # noqa: E731
    obs = ngsrng.LinearObservations(prior, points=[[0.5, 0.5]], noise=1e-3, factorize=False)
    matheron = lambda: ngsrng.MatheronPosterior(obs, data=[1.0], seed=5)  # noqa: E731
    for make in (kl, matheron):
        ref = make().sample(7).numpy()
        sampler = make()
        sampler.work_values = 3 * fes.ndof  # several chunks
        buf = np.empty((fes.ndof, 7))
        assert sampler.sample(7, out=buf) is buf
        np.testing.assert_allclose(buf, ref, rtol=1e-12, atol=1e-14 * np.abs(ref).max())  # BLAS batch widths


def _peak_rss_increase(fill):
    """Peak resident memory added by ``fill()`` (Linux: VmHWM after resetting it)."""
    def hwm():
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024

    def rss():
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024

    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pytest.skip("needs Linux /proc/self/clear_refs")
    before = rss()
    fill()
    return hwm() - before


@pytest.mark.parametrize("name,args,kw", [("normal", (), {}), ("normal", (1.0, 0.5), {}), ("uniform", (), {}),
                                          ("gamma", (2.0,), {}), ("chisquare", (3,), {}),
                                          ("standard_t", (4.0,), {})])
def test_in_place_fills_do_not_allocate_the_batch(name, args, kw):
    """Refilling a large buffer adds far less memory than the buffer itself (old fills went through batches of up to
    2e7 values and two extra copies)."""
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.05)), order=1)
    rng = ngsrng.default_rng(fes, seed=0, correlation=0.3, work_values=2**17)
    d = rng.distribution(name, *args, **kw)
    per = d.sample(1).shape[-1]
    buf = np.ones((max(1, int(6e7 // (4 * per))), per), np.float32)  # ~60 MB, every page touched
    d.sample(out=buf[:2])  # set-up (std, tables) outside the measurement
    extra = _peak_rss_increase(lambda: d.sample(out=buf))
    assert extra < 0.25 * buf.nbytes, (extra / 2**20, buf.nbytes / 2**20)


def test_exact_posterior_memory_check(fes, monkeypatch):
    prior = ngsrng.Gaussian(fes, ngsrng.matern(5.0), dtype="float64", seed=1)
    pts = np.random.default_rng(0).uniform(0.1, 0.9, (20, 2))
    monkeypatch.setattr("ngsrng.posterior.available_memory", lambda: 1e4)
    with pytest.raises(MemoryError, match="exact posterior needs .* Use rank=k"):
        ngsrng.LinearObservations(prior, points=pts, noise=1e-2)
    obs = ngsrng.LinearObservations(prior, points=pts, noise=1e-2, factorize=False)
    with pytest.raises(MemoryError, match="Matheron conditioning needs"):
        ngsrng.MatheronPosterior(obs)
    monkeypatch.setattr("ngsrng.posterior.available_memory", lambda: 1e12)
    ngsrng.LinearObservations(prior, points=pts, noise=1e-2)


@pytest.mark.parametrize("name,args,kw", [
    ("normal", (), {"bnd": "left"}),
    ("uniform", (), {"bnd": "left"}),
    ("gamma", (2.0,), {"bnd": "bottom", "representation": "bernstein"}),
    ("normal", (), {"bnd": "left", "bnd_values": 0.25}),
], ids=["restrict-normal", "restrict-uniform", "restrict-bernstein", "condition"])
def test_boundary_draws_fill_in_place(fes, name, args, kw):
    """bnd= draws (ngsrng/boundary.py) through out=: the boundary step (restriction or conditioning) runs on the
    filled buffer chunk by chunk, so the result equals the allocating call."""
    kw = dict(kw)
    if "bnd_values" in kw:  # prescribed data on the boundary DoFs
        n_bnd = ngsrng.boundary_dofs(fes, kw["bnd"]).size
        kw["bnd_values"] = np.full(n_bnd, kw["bnd_values"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.asarray(ngsrng.default_rng(fes, seed=4, correlation=0.4).sample(name, *args, size=6, **kw))
        buf = np.empty_like(ref)
        rng = ngsrng.default_rng(fes, seed=4, correlation=0.4, work_values=SMALL_WORK)
        assert rng.sample(name, *args, size=6, out=buf, **kw) is buf
    np.testing.assert_allclose(buf, ref, rtol=1e-6, atol=1e-6 * np.abs(ref).max())  # float32 / float64 rounding
    off = np.setdiff1d(np.arange(fes.ndof), ngsrng.boundary_dofs(fes, kw["bnd"], free_only=False))
    if "bnd_values" not in kw:
        assert np.all(buf[:, off] == 0)  # restrict: zero off the boundary
