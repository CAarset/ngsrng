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


def test_l2_dg_odd_power_falls_back_to_dense():
    fes = ngs.L2(_square(), order=1, dgjumps=True)
    with pytest.warns(UserWarning, match="dense Cholesky"):
        g = ngsrng.Gaussian(fes, ngsrng.matern_dg(fes, 3.0, power=3), dtype="float64")
    _check(g, 1e-11)


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
