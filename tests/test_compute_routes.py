"""Device and host routes (``compute=``) compute the same thing: the torch code path on a Gaussian's device agrees
with the NumPy/NGSolve path to rounding, given the same random numbers. On the CPU the torch path runs on the torch
engine's CPU device (compute="device"), so every device route is checked without a GPU; with a usable CUDA device the
same checks run there too."""

import copy
import pickle
import warnings

import ngsolve as ngs
import numpy as np
import pytest
import scipy.stats as st
import torch
from conftest import cuda_usable

import ngsrng
from ngsrng.backend import DeviceCache, compute_device, device_copy, resolve_device, same_device
from ngsrng.quadrature import QuadratureRule

DEVICES = [("cpu", {"engine": "torch"})] + ([("cuda", None)] if cuda_usable() else [])
cuda = pytest.mark.skipif(not cuda_usable(), reason="needs a usable CUDA device")


@pytest.fixture(scope="module")
def fes():
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.15))
    return ngs.H1(mesh, order=2, dirichlet="left")


@pytest.fixture(scope="module", params=DEVICES, ids=[d for d, _ in DEVICES])
def gauss(request, fes):
    device, method = request.param
    # tight tolerance: the CPU torch engine solves by Jacobi PCG, so routes differ by the solver tolerance otherwise
    return ngsrng.Gaussian(fes, ngsrng.matern_correlation(0.3, nu=1).bind(fes), device=device, method=method,
                           dtype="float64", tol=1e-13)


def rel(a, b):
    a = a.detach().double().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, dtype=float)
    b = b.detach().double().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b, dtype=float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ------------------------------------------------------------------ devices and caches


def test_device_helpers():
    assert same_device("cpu", torch.device("cpu"))
    owner = type("Owner", (), {})()
    a = np.arange(5.0)
    t = device_copy(owner, a, "cpu", torch.float32)
    assert device_copy(owner, a, "cpu", torch.float32) is t  # copied once
    assert device_copy(owner, a.copy(), "cpu", torch.float32) is not t  # a new array, a new copy
    twin = copy.copy(owner)  # a shallow copy starts its own cache
    assert device_copy(twin, a, "cpu", torch.float32) is not t
    assert pickle.loads(pickle.dumps(DeviceCache(x=torch.ones(3)))) == {}
    with pytest.raises(ValueError, match="compute must be"):
        compute_device("gpu", owner)


@cuda
def test_cuda_device_has_an_index():
    dev = resolve_device("cuda")
    assert dev.index is not None and same_device("cuda", dev) and same_device(dev, torch.empty(0, device="cuda").device)


@cuda
def test_inplace_fill_takes_the_direct_path(fes, monkeypatch):
    """A buffer on "cuda" is on the Gaussian's device (cuda:0): the in-place path fills it without an intermediate."""
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), device="cuda", dtype="float64", seed=None)
    calls = []
    orig = type(g)._torch_rows
    monkeypatch.setattr(type(g), "_torch_rows", lambda self, *a, **k: calls.append(1) or orig(self, *a, **k))
    buf = torch.empty((6, fes.ndof), device="cuda", dtype=torch.float64)
    torch.manual_seed(1)
    g.sample_into(buf)
    torch.manual_seed(1)
    ref = g.sample(6).T
    assert calls and rel(buf, ref) < 1e-12


# ------------------------------------------------------------------ pointwise families


def _draw(gauss, fes, name, args, **opts):
    rng = ngsrng.default_rng(fes, seed=7, correlation=gauss, device=gauss.device, dtype="float64")
    d = rng.distribution(name, *args, **opts)
    torch.manual_seed(0)  # draw-time randomness of the Gaussian: both routes draw the same Gaussian samples
    return d, d.sample(5)


@pytest.mark.parametrize("representation", ["quadrature", "projection", "bernstein"])
@pytest.mark.parametrize("name, args", [("uniform", ()), ("gamma", (2.0,)), ("lognormal", (0.0, 0.5)),
                                        ("dirichlet", ((1.0, 2.0, 3.0),)), ("chisquare", (3,)),
                                        ("wishart", (4, np.eye(2)))])
def test_pointwise_device_equals_host(gauss, fes, name, args, representation):
    dh, xh = _draw(gauss, fes, name, args, representation=representation, compute="host")
    dd, xd = _draw(gauss, fes, name, args, representation=representation, compute="device")
    assert dh.info["compute"] == "host" and dd.field.on_device and not dh.field.on_device
    # "auto" inversions may tabulate on the device (relative error < 1e-10) and call scipy's ppf for a small host
    # draw; the Bernstein margin adapts to each solver's rounding (about 1e-10 relative)
    assert rel(xd, xh) < 1e-9


def test_bernstein_range_on_device(gauss, fes):
    """The convex-hull guarantee holds for FE functions converted on the device: their values (at the points of a
    fine rule, not their hierarchical coefficients) stay inside the range."""
    _, x = _draw(gauss, fes, "uniform", (-1.0, 2.0), representation="bernstein", compute="device")
    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    values = QuadratureRule(gauss.space, 8).evaluation_full @ x.T
    assert values.min() > -1.0 and values.max() < 2.0


def test_bnd_on_device(gauss, fes):
    rng_h = ngsrng.default_rng(fes, seed=7, correlation=gauss, device=gauss.device, dtype="float64")
    rng_d = ngsrng.default_rng(fes, seed=7, correlation=gauss, device=gauss.device, dtype="float64")
    torch.manual_seed(0)
    xd = rng_d.uniform(size=3, bnd="bottom", compute="device")
    torch.manual_seed(0)
    xh = rng_h.uniform(size=3, bnd="bottom", compute="host")
    assert rel(xd, xh) < 1e-9


def test_compute_option_errors(fes):
    cpu = ngsrng.default_rng(fes, seed=1, correlation=0.3)  # the NGSolve engine: no torch device
    with pytest.raises(ValueError, match="torch device"):
        cpu.uniform(size=2, compute="device")
    rng = ngsrng.default_rng(fes, seed=1, correlation=0.3, method={"engine": "torch"})
    with pytest.raises(ValueError, match="no torch path"):  # a user's NumPy function (np.tanh itself is recognised)
        rng.transformed(lambda x: 1 / (1 + np.exp(-x)), size=2, compute="device")
    with pytest.raises(ValueError, match="applies to pointwise families"):
        rng.standard_t(4, size=2, compute="host")
    with pytest.raises(ValueError, match="compute must be"):
        rng.uniform(size=2, compute="gpu")


def test_untabulable_inversion_is_honest(fes):
    """An inversion whose table cannot be built says so: no torch path, a warning, the field draws on the host."""
    from ngsrng.distributions.inversion import Inversion

    inv = Inversion(st.poisson(1e12), method="auto")
    assert inv.torch_capable
    with pytest.warns(RuntimeWarning, match="too wide"):
        assert inv._resolve(10**6, True) == "ppf"
    assert not inv.torch_capable and inv.info["inversion"] == "ppf"


# ------------------------------------------------------------------ low rank, KL, posteriors, covariances


def test_karhunen_loeve_device_equals_host(gauss):
    kh = ngsrng.karhunen_loeve(gauss, 12, compute="host")
    kd = ngsrng.karhunen_loeve(gauss, 12, compute="device")
    assert rel(kd.eigenvalues, kh.eigenvalues) < 1e-9
    assert rel((kd.modes * kd.eigenvalues) @ kd.modes.T, (kh.modes * kh.eigenvalues) @ kh.modes.T) < 1e-8
    assert kh.device.type == "cpu" and kd.device.type == gauss.device.type
    for method in ("covariance",):
        a = ngsrng.karhunen_loeve(gauss, 12, method=method, compute="host")
        b = ngsrng.karhunen_loeve(gauss, 12, method=method, compute="device")
        assert rel(b.eigenvalues, a.eigenvalues) < 1e-9


@cuda
def test_low_rank_gaussian_on_cuda(fes):
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), dtype="float64", seed=0)
    kl = ngsrng.karhunen_loeve(g, 10)
    on = ngsrng.LowRankGaussian(fes, kl.b, kl.k, mean=kl.mean, seed=4, device="cuda")
    off = ngsrng.LowRankGaussian(fes, kl.b, kl.k, mean=kl.mean, seed=4)
    x = on.sample(5)
    assert x.device.type == "cuda" and rel(x, off.sample(5)) < 1e-12  # the same NumPy coefficients
    eta = torch.randn(kl.nrand, 3, dtype=torch.float64)
    assert rel(on.apply(eta), off.apply(eta)) < 1e-12
    y = torch.randn(on.space.nfree, 3, dtype=torch.float64)
    assert rel(on.apply_t(y), off.apply_t(y)) < 1e-12
    post = on.condition(points=[(0.5, 0.5)], noise=0.1, data=[1.0])
    assert post.device == on.device
    twin = pickle.loads(pickle.dumps(on))  # the stream state travels; the device factor is made again
    assert rel(twin.sample(2), on.sample(2)) < 1e-12


@pytest.mark.parametrize("rank", [None, 6])
def test_linear_observations_device_equals_host(gauss, rank):
    pts = np.random.default_rng(0).uniform(0.2, 0.8, (12, 2))
    oh = ngsrng.LinearObservations(gauss, points=pts, noise=1e-2, rank=rank, compute="host")
    od = ngsrng.LinearObservations(gauss, points=pts, noise=1e-2, rank=rank, compute="device")
    assert rel(od.singular_values, oh.singular_values) < 1e-10
    assert rel(od.lq() @ od.R, oh.lq() @ oh.R) < 1e-10  # L Q R = L F^T: invariant under the sign of Q's columns
    assert rel(od.q @ od.R, oh.q @ oh.R) < 1e-10
    eta = torch.randn(gauss.nrand, 3, dtype=torch.float64)
    for form in ("fe", "noise"):
        ph = oh.posterior(data=np.ones(12), form=form)
        pd = od.posterior(data=np.ones(12), form=form)
        assert rel(pd.apply(eta), ph.apply(eta)) < 1e-10 and rel(pd.mean, ph.mean) < 1e-10
        twin = pickle.loads(pickle.dumps(pd))  # the device factors pickle as host arrays
        assert rel(twin.apply(eta), pd.apply(eta)) < 1e-10


def test_matheron_device_equals_host(gauss):
    pts = np.random.default_rng(0).uniform(0.2, 0.8, (12, 2))
    obs = ngsrng.LinearObservations(gauss, points=pts, noise=1e-2, factorize=False)
    mh = ngsrng.MatheronPosterior(obs, data=np.ones(12), compute="host")
    md = ngsrng.MatheronPosterior(obs, data=np.ones(12), compute="device")
    eta = torch.randn(mh.nrand, 3, dtype=torch.float64)
    y = torch.randn(gauss.space.nfree, 3, dtype=torch.float64)
    assert rel(md.apply(eta), mh.apply(eta)) < 1e-10 and rel(md.apply_t(y), mh.apply_t(y)) < 1e-10
    assert rel(md.mean, mh.mean) < 1e-12 and rel(md.bmat, mh.bmat) < 1e-10
    x = md.sample(4)
    assert x.device.type == gauss.device.type and x.shape == (gauss.space.ndof, 4)
    twin = pickle.loads(pickle.dumps(md))
    assert rel(twin.apply(eta), md.apply(eta)) < 1e-12


@pytest.mark.parametrize("method", ["exact", "lowrank", "mc"])
def test_local_covariance_device_equals_host(gauss, method):
    e = QuadratureRule(gauss.space).evaluation
    lh = ngsrng.local_covariance(gauss, method, compute="host", samples=30, rank=20)
    ld = ngsrng.local_covariance(gauss, method, compute="device", samples=30, rank=20)
    assert rel(ld.values(e), lh.values(e)) < 1e-9


def test_posterior_local_covariance_device_equals_host(gauss):
    post = gauss.condition(points=[(0.4, 0.5), (0.6, 0.3)], noise=1e-2)
    e = QuadratureRule(gauss.space).evaluation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ngsrng.local_covariance(post, compute="device").values(e)
        b = ngsrng.local_covariance(post, compute="host").values(e)
    assert rel(a, b) < 1e-9


def test_correlated_gaussian_fields_stay_on_the_device(gauss, fes):
    rng = ngsrng.default_rng(fes, seed=2, correlation=gauss, device=gauss.device, dtype="float64")
    d = rng.distribution("normal")
    a, b = ngsrng.correlated([d, d], 0.6).sample(4)
    if gauss.device.type == "cuda":
        assert isinstance(a, torch.Tensor) and a.device.type == "cuda"
    assert a.shape == b.shape == (4, fes.ndof)


# ------------------------------------------------------------------ precision, user transforms, stationary fields


@pytest.mark.parametrize("power", [1, 2, 3])
def test_apply_precision_on_the_input_device(fes, power):
    """A block on a GPU is processed there, with the host's result; CPU blocks as before. The Gaussian pickles
    afterwards (its host operators used to hold lambdas)."""
    g = ngsrng.Gaussian(fes, ngsrng.matern(5.0, power), dtype="float64", seed=1)
    c = torch.randn(g.space.nfree, 3, dtype=torch.float64)
    host = g.apply_precision(c)
    assert host.device.type == "cpu"
    pickle.loads(pickle.dumps(g))
    if cuda_usable():
        on = g.apply_precision(c.cuda())
        assert on.device.type == "cuda" and rel(on, host) < 1e-12
        assert rel(g.apply_precision(c[:, 0].cuda()), host[:, 0]) < 1e-12


def _own_torch(x):  # a user's function written for torch tensors
    return torch.special.expit(2 * x)


def _own_numpy(x):
    return 1 / (1 + np.exp(-2 * x))


@pytest.mark.parametrize("func, kw, capable", [(np.tanh, {}, True), (torch.tanh, {}, True),
                                               (_own_torch, {"torch": True}, True), (_own_numpy, {}, False)],
                         ids=["np.tanh", "torch.tanh", "own-torch", "own-numpy"])
def test_transformed_dispatches_by_function(gauss, fes, func, kw, capable):
    """NumPy ufuncs and torch functions are recognised; torch=True declares a torch function of one's own. Each runs
    on whatever device the field computes on and gives the host route's values; a NumPy function stays on the host."""
    dh, xh = _draw(gauss, fes, "transformed", (func,), compute="host", **kw)
    if not capable:
        assert not dh.field.torch_capable
        with pytest.raises(ValueError, match="no torch path"):
            _draw(gauss, fes, "transformed", (func,), compute="device", **kw)
        return
    dd, xd = _draw(gauss, fes, "transformed", (func,), compute="device", **kw)
    assert dd.field.on_device and rel(xd, xh) < 1e-12
    ref = _draw(gauss, fes, "transformed", (_own_numpy if func is _own_torch else np.tanh,), compute="host")[1]
    assert rel(xh, ref) < 1e-12  # the same map, whichever array type it is written for


def test_transformed_torch_function_on_the_cpu_engine(fes):
    rng = ngsrng.default_rng(fes, seed=1, correlation=0.3)  # NGSolve engine: NumPy values, wrapped as CPU tensors
    torch.manual_seed(0)
    a = rng.transformed(torch.tanh, size=3)
    rng = ngsrng.default_rng(fes, seed=1, correlation=0.3)
    torch.manual_seed(0)
    assert isinstance(a, np.ndarray) and rel(a, rng.transformed(np.tanh, size=3)) < 1e-14


def test_transformed_torch_promise_is_checked():
    from ngsrng.distributions.pointwise import XP
    from ngsrng.distributions.transforms import PointMap

    with pytest.raises(TypeError, match="torch=True promises"):
        PointMap(lambda x: np.asarray(x), torch=True)(torch.zeros(3), XP(True))


STATIONARY = [("restrict", "dense"), ("restrict", "fft"), ("torus", "spde")]


@pytest.mark.parametrize("construction, method", STATIONARY)
@pytest.mark.parametrize("dirichlet", ["", "left"])
def test_stationary_matern_torch_path_equals_numpy(construction, method, dirichlet):
    """The torch code path (the torch engine on the CPU; and on CUDA) of every construction, with and without the
    Dirichlet boundary update, against the NumPy path on the same random input."""
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25)), order=1, dirichlet=dirichlet)
    kw = dict(construction=construction, method=method, seed=1, resolution=0.15)
    host = ngsrng.StationaryMatern(fes, 0.4, 1.0, **kw)
    devices = [("cpu", "torch")] + ([("cuda", None)] if cuda_usable() else [])
    eta = torch.randn(host.nrand, 3, dtype=torch.float64)
    y = torch.randn(host.space.nfree, 3, dtype=torch.float64)
    for device, engine in devices:
        f = ngsrng.StationaryMatern(fes, 0.4, 1.0, device=device, engine=engine, **kw)
        # the torus Gaussian solves by Jacobi PCG on the CPU torch engine (tol 1e-8), by cuDSS on CUDA
        tol = 1e-6 if (method == "spde" and device == "cpu") else 1e-9
        assert rel(f.apply(eta), host.apply(eta)) < tol and rel(f.apply_t(y), host.apply_t(y)) < tol
        x = f.sample(3)
        assert x.device.type == device and x.shape == (fes.ndof, 3)
        assert rel(pickle.loads(pickle.dumps(f)).apply(eta), f.apply(eta)) < 1e-12


@cuda
def test_stationary_matern_follows_the_generator_device(fes):
    rng = ngsrng.default_rng(fes, seed=0, device="cuda",
                             correlation=ngsrng.matern_correlation(0.3, nu=1, boundary="restrict"))
    u = rng.uniform(size=4)  # unchanged syntax: the field and the transform on the GPU
    assert rng.gaussian().device.type == "cuda" and u.device.type == "cuda" and rng.info("uniform")["device"] == "gpu"
    assert float(u.min()) > 0 and float(u.max()) < 1
