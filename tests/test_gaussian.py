import warnings

import ngsolve as ngs
import numpy as np
import pytest
import torch
from conftest import cuda_usable
from ngsolve.meshes import Make1DMesh

import ngsrng
from ngsrng.reference import covariance_errors, exact_covariance, sampler_covariance


def _square(maxh=0.25):
    return ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=maxh))


def _check(g, tol):
    err = covariance_errors(sampler_covariance(g), exact_covariance(g))
    assert err["rel_fro"] < tol, err
    assert err["max_rel_var"] < tol, err


@pytest.mark.parametrize("power", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "engine,solver,tol",
    [("ngsolve", "direct", 1e-11), ("ngsolve", "cg", 1e-9), ("torch", "direct", 1e-11), ("torch", "torch-pcg", 1e-9)],
)
def test_h1_matern_exact(power, engine, solver, tol):
    fes = ngs.H1(_square(), order=2, dirichlet="left")
    g = ngsrng.Gaussian(
        fes, ngsrng.matern(3.0, power=power), method={"solver": solver, "engine": engine}, tol=1e-12, dtype="float64"
    )
    _check(g, tol)


@pytest.mark.parametrize(
    "fes",
    [
        lambda: ngs.H1(Make1DMesh(20), order=4),
        lambda: ngs.H1(ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5)), order=2, dirichlet="left"),
    ],
    ids=["1D-p4", "3D-p2"],
)
def test_h1_matern_other_dims(fes):
    g = ngsrng.Gaussian(fes(), ngsrng.matern(2.0, power=2), dtype="float64")
    _check(g, 1e-11)


@pytest.mark.parametrize("engine", ["ngsolve", "torch"])
@pytest.mark.parametrize("noise", ["local-gram", "quadrature", "dense-cholesky", "chebyshev"])
def test_noise_override(noise, engine):
    method = {"noise": noise, "engine": engine, "noise_tol": 1e-12, "solver": "direct"}
    g = ngsrng.Gaussian(ngs.H1(_square(), order=3), ngsrng.matern(2.0), method=method, dtype="float64")
    assert g.plan["noise"] == noise
    _check(g, 1e-11)


@pytest.mark.parametrize("power", [2, 4])
def test_l2_dg_matern_exact(power):
    fes = ngs.L2(_square(), order=2, dgjumps=True)
    g = ngsrng.Gaussian(fes, ngsrng.matern_dg(fes, 3.0, power=power), dtype="float64")
    assert g.plan["noise"] == "diagonal"
    _check(g, 1e-11)


@pytest.mark.parametrize("power", [1, 3])
def test_l2_dg_odd_power_takes_the_rational_route(power):
    """DG forms couple elements, so there is no local precision-shaped noise: white noise, sparse solves and the
    rational square root instead of a dense factorisation."""
    fes = ngs.L2(_square(), order=1, dgjumps=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # power 1 is not trace class in 2D
        g = ngsrng.Gaussian(fes, ngsrng.matern_dg(fes, 3.0, power=power), dtype="float64")
    assert g.plan["noise"] == "diagonal" and "rational (M^-1 K)^-0.5" in g.plan["structure"]
    assert g.plan["form_noise_choice"].startswith("rational")
    _check(g, 1e-6)
    exact = ngsrng.Gaussian(fes, ngsrng.matern_dg(fes, 3.0, power=3), dtype="float64",
                            method={"form_noise": "dense-cholesky"})  # still available on request
    assert exact.plan["noise"] == "dense-cholesky"


def test_assembled_precision_with_facet_terms():
    fes = ngs.L2(_square(), order=1, dgjumps=True)
    form = ngsrng.matern_dg(fes, 3.0, power=1).form
    g = ngsrng.Gaussian(fes, ngsrng.AssembledPrecision(form, scale=2.0), dtype="float64")
    assert g.plan["structure"].startswith("Sigma^-1 = Q assembled: white noise + rational")
    q = g.space.assemble_free(form).toarray()
    err = covariance_errors(sampler_covariance(g), 4.0 * np.linalg.inv(q))
    assert err["rel_fro"] < 1e-6 and err["max_rel_var"] < 1e-6, err


def test_rational_route_on_request():
    fes = ngs.H1(_square(), order=2, dirichlet="left")
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0, power=3), dtype="float64", method={"form_noise": "rational"})
    assert "rational" in g.plan["structure"]
    _check(g, 1e-6)


def test_assembled_fourth_order_precision():
    fes = ngs.H1(_square(0.3), order=3, dirichlet="left")

    def form(u, v):
        hess = ngs.InnerProduct(u.Operator("hesse"), v.Operator("hesse"))
        return (hess + ngs.grad(u) * ngs.grad(v) + u * v) * ngs.dx

    g = ngsrng.Gaussian(fes, ngsrng.AssembledPrecision(form), dtype="float64")
    assert g.plan["noise"] == "local-gram"
    _check(g, 1e-11)


@pytest.mark.parametrize("engine", ["ngsolve", "torch"])
def test_sample_shapes_mean_and_dirichlet(engine):
    fes = ngs.H1(_square(), order=2, dirichlet="left")
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), mean=ngs.x + 2, seed=0, method={"engine": engine})
    gf = g.sample()
    assert isinstance(gf, ngs.GridFunction)
    xs = g.sample(7)
    assert xs.shape == (fes.ndof, 7) and xs.dtype == torch.float32
    dirichlet = ~np.array(fes.FreeDofs(), dtype=bool)
    mean = ngs.GridFunction(fes)
    mean.Set(ngs.x + 2)
    np.testing.assert_allclose(
        xs[dirichlet].numpy(), np.repeat(mean.vec.FV().NumPy()[dirichlet, None], 7, 1), rtol=1e-6
    )
    mv = g.sample(3, out="multivector")
    assert len(mv) == 3


@pytest.mark.parametrize("engine", ["ngsolve", "torch"])
def test_seed_reproducibility(engine):
    fes = ngs.H1(_square(), order=2)
    a = ngsrng.Gaussian(fes, ngsrng.matern(3.0), seed=42, method={"engine": engine}).sample(4)
    b = ngsrng.Gaussian(fes, ngsrng.matern(3.0), seed=42, method={"engine": engine}).sample(4)
    c = ngsrng.Gaussian(fes, ngsrng.matern(3.0), seed=43, method={"engine": engine}).sample(4)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_trace_class_warning():
    with pytest.warns(UserWarning, match="not trace class"):
        ngsrng.Gaussian(ngs.H1(_square(), order=1), ngsrng.matern(1.0, power=1))


@pytest.mark.parametrize("engine", ["ngsolve", "torch"])
def test_monte_carlo_functional_variance(engine):
    """Var <f_h, psi> from actual samples matches psi^T Sigma psi (statistical, 5 sigma)."""
    fes = ngs.H1(_square(0.2), order=2)
    g = ngsrng.Gaussian(fes, ngsrng.matern(2.0), seed=3, dtype="float64", method={"engine": engine})
    psi = ngs.LinearForm(ngs.exp(-10 * ((ngs.x - 0.3) ** 2 + (ngs.y - 0.6) ** 2)) * fes.TestFunction() * ngs.dx)
    psi.Assemble()
    m = psi.vec.FV().NumPy()[g.space.free]
    exact = m @ exact_covariance(g) @ m
    n = 4000
    vals = m @ g.sample(n).numpy()[g.space.free]
    est = vals.var()
    assert abs(est - exact) < 5 * exact * np.sqrt(2 / (n - 1))


cuda = pytest.mark.skipif(not cuda_usable(), reason="needs a usable CUDA device")


@cuda
@pytest.mark.parametrize("solver", ["cudss", "torch-pcg"])
@pytest.mark.parametrize("dtype,tol", [("float64", 1e-9), ("float32", 1e-4)])
@pytest.mark.parametrize("power", [2, 3])
def test_gpu_engine_exact(solver, dtype, tol, power):
    if solver == "cudss":
        pytest.importorskip("nvmath")
    fes = ngs.H1(_square(), order=2, dirichlet="left")
    g = ngsrng.Gaussian(
        fes, ngsrng.matern(3.0, power=power), method={"solver": solver}, device="cuda", dtype=dtype, tol=1e-12
    )
    assert g.plan["engine"] == "torch" and g.plan["solver"].startswith(solver.split("-")[0])
    _check(g, tol)
    x = g.sample(40)
    assert x.shape == (fes.ndof, 40) and x.device.type == "cuda"


@cuda
def test_cudss_column_slices_and_odd_batches():
    """Regression: single columns sliced from a batch have strides cuDSS rejects unless copied."""
    pytest.importorskip("nvmath")
    fes = ngs.H1(_square(), order=2)
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"solver": "cudss"}, device="cuda", dtype="float64")
    eta = torch.randn(g.nrand, 37, device="cuda", dtype=torch.float64)
    full = g.apply(eta)
    for j in (0, 33, 36):
        torch.testing.assert_close(g.apply(eta[:, j]), full[:, j])
    torch.testing.assert_close(g.apply(eta[:, 3:4]), full[:, 3:4])


# ------------------------------------------------------------------ silent-failure guards (review 2026-10-31)


def test_method_keys_and_names_are_validated():
    """A misspelt key must not be ignored (it would e.g. run a GPU study on the CPU), nor a bad name be a KeyError."""
    fes = ngs.H1(_square(0.4), order=1)
    with pytest.raises(ValueError, match="did you mean 'engine'"):
        ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"engien": "torch"})
    with pytest.raises(ValueError, match="unknown noise 'lokal-gram'.*local-gram"):
        ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"noise": "lokal-gram"})
    with pytest.raises(ValueError, match="unknown solver"):
        ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"solver": "cholesky"})
    with pytest.raises(ValueError, match="unknown method key"):  # at construction, before any draw
        ngsrng.default_rng(fes, correlation=0.3, method={"solvr": "cg"})
    with pytest.raises(TypeError, match="per-call option"):
        ngsrng.default_rng(fes, correlation=0.3, method="table")
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"engine": "auto", "solver": "auto"})
    assert g.engine == "ngsolve"
    with pytest.raises(ValueError, match="unknown work_dtype"):
        ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"work_dtype": "float16"})
    # multigrid needs a mesh hierarchy (it crashed the process on a generated mesh); pardiso returned garbage
    with pytest.raises(ValueError, match="unknown precond 'multigrid'"):
        ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"solver": "cg", "precond": "multigrid"})
    with pytest.raises(ValueError, match="unknown inverse 'pardiso'"):
        ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"solver": "direct", "inverse": "pardiso"})


@pytest.mark.parametrize("precond", ["bddc", "h1amg", "local"])
def test_iterative_batches_match_direct(precond):
    """A threaded batch through CG equals the direct solve: concurrent applications of one BDDC preconditioner
    corrupted the samples (relative error 0.1), so pipelines with thread-unsafe solvers run samples one at a time."""
    fes = ngs.H1(_square(0.1), order=2)
    ref = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), method={"solver": "direct"}, dtype="float64")
    eta = torch.randn(ref.nrand, 8, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    c0 = ref.apply(eta).numpy()
    g = ngsrng.Gaussian(fes, ngsrng.matern(5.0, 2), method={"solver": "cg", "precond": precond}, dtype="float64",
                        tol=1e-12)
    assert g._threadsafe == (precond == "h1amg")
    assert np.abs(g.apply(eta).numpy() - c0).max() < 1e-8 * np.abs(c0).max()


def test_float64_arithmetic_by_default():
    """Every engine computes in float64 and casts the output: float32 solves lost the samples on fine meshes (GPU
    benchmark). float32 arithmetic is an opt-in of the GPU engine; on the CPU it has no effect and no warning."""
    fes = ngs.H1(_square(0.4), order=1)
    g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"engine": "torch"})
    assert g.dtype == torch.float32 and g._work_dtype == torch.float64
    assert g._platform == "cpu-torch"
    assert ngsrng.tuning.platform_of("torch", "cuda", torch.float64) == "cuda-f64"
    assert ngsrng.tuning.platform_of("torch", "cuda", "float32") == "cuda-f32"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"work_dtype": "float32"})
    assert g._work_dtype == torch.float64
    if cuda_usable():
        with pytest.warns(UserWarning, match="single precision"):
            g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), device="cuda", method={"work_dtype": "float32"})
        # GPU tables are per device (tuning format 2): the platform names the card
        assert g._work_dtype == torch.float32 and g._platform.startswith("cuda-f32@")
        assert ngsrng.Gaussian(fes, ngsrng.matern(3.0), device="cuda")._platform.startswith("cuda-f64@")


@pytest.mark.parametrize("noise,expected", [("lumped", "relative covariance error"), ("chebyshev", "1e-08")])
def test_approximate_noise_states_its_accuracy(noise, expected):
    fes = ngs.H1(_square(), order=1)  # row-sum lumping is invalid (non-positive) for P2
    with pytest.warns(UserWarning, match=f"approximate white noise \\({noise}\\).*{expected}"):
        g = ngsrng.Gaussian(fes, ngsrng.matern(3.0), method={"noise": noise, "noise_tol": 5e-9}, dtype="float64")
    assert expected in g.plan["noise_accuracy"]
    if noise == "lumped":  # the stated error is the actual one
        m = g.space.mass_free.toarray()
        assert f"{np.abs(np.diag(m.sum(axis=1)) - m).max() / np.abs(m).max():.1e}" in g.plan["noise_accuracy"]


def test_dense_cholesky_checks_memory_first(monkeypatch):
    fes = ngs.L2(_square(), order=1, dgjumps=True)
    spec = ngsrng.matern_dg(fes, 3.0, power=3)
    monkeypatch.setattr("ngsrng.noise.available_memory", lambda: 1e3)
    with pytest.raises(MemoryError, match="needs .* GiB"):
        ngsrng.Gaussian(fes, spec, dtype="float64", method={"form_noise": "dense-cholesky"})


def test_derived_gaussians_leave_the_prior_untouched():
    """apply_t of a posterior (CPU engine) must not re-set-up the noise object it shares with its prior."""
    fes = ngs.H1(_square(), order=2, dirichlet="left")
    prior = ngsrng.Gaussian(fes, ngsrng.matern(3.0), dtype="float64", seed=0)
    noise_state = dict(vars(prior.noise))
    post = prior.condition(points=[[0.5, 0.5], [0.3, 0.7]], noise=1e-3)
    y = torch.randn(prior.space.nfree, 3, dtype=torch.float64)
    lt = post.apply_t(y)
    assert vars(prior.noise).keys() == noise_state.keys()  # no torch set-up on the shared sampler
    # adjoint of apply: <L eta, y> == <eta, L^T y>
    eta = torch.randn(post.nrand, 3, dtype=torch.float64)
    torch.testing.assert_close((post.apply(eta) * y).sum(), (eta * lt).sum(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_posterior_out_buffer_uses_the_posterior_mean(device):
    """Regression: posteriors are copies of their prior, and inherited the prior's cached device mean, so batches
    filled in place (``out=``, torch engine) were centred on the prior mean."""
    fes = ngs.H1(_square(0.3), order=2)
    prior = ngsrng.Gaussian(fes, ngsrng.matern(5.0), method={"engine": "torch"}, device=device, seed=1)
    rng = ngsrng.default_rng(fes, seed=0, correlation=prior, device=device)
    buf = torch.empty(2000, fes.ndof, dtype=torch.float32, device=device)
    rng.normal(out=buf)  # caches the prior mean on the device
    post = prior.condition(points=[[0.5, 0.5]], noise=1e-6, data=[5.0])
    g = ngsrng.observation_operator(fes, points=[[0.5, 0.5]])
    for gauss, target in ((post, 5.0), (prior, 0.0)):
        rng.normal(out=buf, correlation=gauss)
        assert abs((g @ buf.double().cpu().numpy().T).mean() - target) < 0.05
    shifted = ngsrng.LinearObservations(prior, points=[[0.5, 0.5]], noise=1e-6).posterior(mean=ngs.CF(2.0))
    rng.normal(out=buf, correlation=shifted)
    assert abs((g @ buf.double().cpu().numpy().T).mean() - 2.0) < 0.05


def test_apply_precision_by_specification():
    fes = ngs.H1(_square(), order=2, dirichlet="left")
    s = ngsrng.space.SpaceInfo(fes)
    k = s.assemble_free(lambda u, v: (ngs.grad(u) * ngs.grad(v) + 9 * u * v) * ngs.dx).toarray()
    m = s.mass_free.toarray()
    q = torch.from_numpy(k @ np.linalg.solve(m, k))
    c = torch.randn(s.nfree, 2, dtype=torch.float64)
    expected = (q @ c) / 4.0
    pde = ngsrng.Gaussian(fes, ngsrng.matern(3.0, power=2, scale=2.0), dtype="float64")
    torch.testing.assert_close(pde.apply_precision(c), expected)
    op = ngsrng.Gaussian(fes, ngsrng.OperatorPrecision(lambda x: q @ x, scale=2.0), dtype="float64")
    torch.testing.assert_close(op.apply_precision(c), expected)
    frac = ngsrng.Gaussian(fes, ngsrng.matern(3.0, power=2.5), dtype="float64")
    with pytest.raises(NotImplementedError, match="integer power"):
        frac.apply_precision(c)


def test_torch_pcg_zero_columns_and_non_convergence():
    from ngsrng.solvers import TorchPCG

    s = ngsrng.space.SpaceInfo(ngs.H1(_square(), order=2, dirichlet="left"))
    a = s.assemble_free(lambda u, v: (ngs.grad(u) * ngs.grad(v) + u * v) * ngs.dx)
    b = torch.randn(s.nfree, 3, dtype=torch.float64)
    b[:, 1] = 0.0
    x = TorchPCG(a, "cpu", torch.float64, tol=1e-12).apply(b)
    assert torch.isfinite(x).all() and (x[:, 1] == 0).all()
    np.testing.assert_allclose(a @ x.numpy(), b.numpy(), atol=1e-9)
    with pytest.warns(UserWarning, match="did not converge in 3 iterations"):
        TorchPCG(a, "cpu", torch.float64, tol=1e-12, maxiter=3).apply(b)
