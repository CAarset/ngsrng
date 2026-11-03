"""The fast paths are exact: threaded batches, in-place ``out=`` filling, torch transforms on the CPU, and the
threaded factorisations reproduce the reference results."""

import warnings

import ngsolve as ngs
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.special as sps
import torch

import ngsrng
from ngsrng.backend import NormalStream
from ngsrng.distributions.bernstein import SymmetricFactor
from ngsrng.distributions.pointwise import XP
from ngsrng.distributions.transforms import CLOSED
from ngsrng.solvers import NGSolveDirect
from ngsrng.space import SpaceInfo, mass_form


@pytest.fixture(scope="module")
def fes():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.05)), order=2, dirichlet="left")


@pytest.mark.parametrize("nu", [1.0, 2.0, 1.6])  # power 2 (white noise), 3 (stiffness-shaped noise), 2.6 (rational)
def test_batches_do_not_depend_on_the_thread_count(fes, nu):
    g = ngsrng.default_rng(fes, seed=3, correlation=ngsrng.matern_correlation(0.3, nu=nu), dtype="float64").gaussian()
    post = g.condition(points=[(0.3, 0.4), (0.7, 0.6)], noise=0.01, data=[1.0, -1.0])
    for gauss in (g, post):
        out = {}
        for threads in (1, 3, 8):
            gauss.method["threads"] = threads
            gauss.stream = NormalStream(11)
            out[threads] = gauss.sample(12).numpy()
        assert np.array_equal(out[1], out[3]) and np.array_equal(out[1], out[8])
        # and a batch equals sequential single samples, bit for bit (single samples are computed like batch columns)
        gauss.stream = NormalStream(11)
        singles = np.column_stack([gauss.sample().vec.FV().NumPy() for _ in range(12)])
        np.testing.assert_array_equal(singles, out[1])


def test_apply_and_solver_columns_are_exact_in_threads(fes):
    g = ngsrng.default_rng(fes, seed=1, correlation=0.3, dtype="float64").gaussian()
    eta = torch.from_numpy(np.random.default_rng(0).standard_normal((g.nrand, 9)))
    cols = torch.cat([g.apply(eta[:, [k]]) for k in range(9)], dim=1)  # single columns: NGSolve task parallelism
    # its summation order varies from run to run: entries near zero differ at rounding level (seen: 1.9e-15)
    np.testing.assert_allclose(g.apply(eta).numpy(), cols.numpy(), rtol=1e-11, atol=1e-13 * cols.abs().max().item())
    solver = NGSolveDirect(SpaceInfo(fes), mass_form)
    x = torch.from_numpy(np.random.default_rng(1).standard_normal((len(SpaceInfo(fes).free), 7)))
    one_by_one = torch.cat([solver.apply(x[:, k]).reshape(-1, 1) for k in range(7)], dim=1)
    np.testing.assert_allclose(solver.apply(x).numpy(), one_by_one.numpy(), rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize("name,args", [("normal", ()), ("normal", (1.0, 0.5)), ("lognormal", (0.0, 0.5)),
                                       ("uniform", ()), ("gamma", (2.0,)), ("poisson", (3.0,))])
def test_out_fills_in_place(fes, name, args):
    ref = getattr(ngsrng.default_rng(fes, seed=4, correlation=0.3), name)(*args, size=6)
    assert ref.flags.c_contiguous  # results are C-ordered: samples are contiguous rows
    for out in (np.empty(ref.shape, dtype=ref.dtype), torch.empty(ref.shape, dtype=torch.from_numpy(ref).dtype)):
        got = getattr(ngsrng.default_rng(fes, seed=4, correlation=0.3), name)(*args, out=out)
        assert got is out
        np.testing.assert_array_equal(np.asarray(out), ref)
    # float64 buffers receive the same values (cast from the same computation)
    out64 = np.empty(ref.shape)
    getattr(ngsrng.default_rng(fes, seed=4, correlation=0.3, dtype="float64"), name)(*args, out=out64)
    np.testing.assert_allclose(out64, ref, rtol=1e-6, atol=1e-6)


def test_out_repeated_in_place_continues_the_stream(fes):
    rng = ngsrng.default_rng(fes, seed=5, correlation=0.3)
    buf = np.empty((4, fes.ndof), dtype=np.float32)
    first = rng.normal(out=buf).copy()
    second = rng.normal(out=buf).copy()
    ref = ngsrng.default_rng(fes, seed=5, correlation=0.3).normal(size=8)
    np.testing.assert_array_equal(np.vstack([first, second]), ref)
    big = np.empty((3, 2, fes.ndof), dtype=np.float32)  # size (3, 2), given or inferred from out
    rng.normal(size=(3, 2), out=big)
    with pytest.raises(ValueError, match="size"):
        rng.normal(size=5, out=buf)
    with pytest.raises(ValueError, match="contiguous"):
        rng.normal(out=np.empty((fes.ndof, 4), dtype=np.float32).T)
    with pytest.raises(ValueError, match="float64"):
        rng.uniform(representation="bernstein", out=np.empty((2, fes.ndof), dtype=np.float32))
    with pytest.raises(ValueError, match="values per sample"):
        rng.normal(out=np.empty((2, 7), dtype=np.float32))


def test_torch_normal_cdf_keeps_both_tails():
    z = np.linspace(-37.0, 8.0, 100001)
    ref = sps.ndtr(z)
    val = XP(True).ndtr(torch.from_numpy(z)).numpy()
    assert np.max(np.abs(val - ref) / ref) < 1e-11  # torch.special.ndtr: 100% error below z = -10


def test_closed_forms_in_torch_match_numpy():
    z = np.random.default_rng(0).standard_normal((500, 20)) * 2.5
    for name, (func, defaults, _) in CLOSED.items():
        params = {k: (2.5 if v is None else v) for k, v in defaults.items()}
        if name == "triangular":
            params = {"left": 0.0, "mode": 0.3, "right": 1.0}
        a = func(z, XP(False), **params)
        b = func(torch.from_numpy(z), XP(True), **params).numpy()
        nz = np.abs(a) > 0
        assert np.max(np.abs(a[nz] - b[nz]) / np.abs(a[nz])) < 1e-11, name


def test_symmetric_factor_matches_superlu():
    n = 300
    a = sp.random(n, n, density=0.02, random_state=0)
    a = (a @ a.T + sp.identity(n) * 2.0).tocsr()
    b = np.random.default_rng(0).standard_normal((n, 11))
    np.testing.assert_allclose(SymmetricFactor(a).solve(b), spla.splu(a.tocsc()).solve(b), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(SymmetricFactor(a).solve(b[:, 0]), spla.splu(a.tocsc()).solve(b[:, 0]), rtol=1e-10)


def test_pointwise_values_come_from_float64_coefficients(fes):
    """The CPU engine evaluates float64 coefficients at the points (no float32 round trip before the transform)."""
    rng = ngsrng.default_rng(fes, seed=6, correlation=0.3)  # float32 output
    pts = rng._gaussian_at_points(rng.gaussian(), std="exact")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals = pts.draw(3)
    assert vals.dtype == np.float64
    g = rng.gaussian()
    g.stream = NormalStream(1)
    x = pts.draw(2)
    g.stream = NormalStream(1)
    full = g._ngs_rows(2, np.empty((2, fes.ndof)))
    np.testing.assert_allclose(x, (pts.eval_full @ full.T), rtol=1e-12, atol=1e-12)
