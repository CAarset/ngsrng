"""Per-domain tuning of the Robin factor (ngsrng.tune_robin, matern_correlation(..., robin="tune"))."""

import pickle
import warnings

import ngsolve as ngs
import numpy as np
import pytest
from ngsolve.meshes import Make1DMesh

import ngsrng
from ngsrng import robin


@pytest.fixture(scope="module")
def square():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.08)), order=2)


def test_exponential_covariance_needs_beta_equal_kappa():
    """nu = 1/2 in 1D is the Ornstein-Uhlenbeck process, stationary on an interval exactly for beta = kappa: the tuner
    finds c = 1 and a flat variance."""
    r = ngsrng.tune_robin(ngs.H1(Make1DMesh(200), order=2), 0.2, 0.5)
    assert abs(r.factor - 1) < 0.03 and r.rms < 0.01
    assert r.default_rms > 5 * r.rms and abs(r.beta - r.factor * r.kappa) < 1e-12


def test_smooth_fields_get_a_smaller_factor(square):
    """nu = 5 in 2D: the classical 1/1.42 pulls the boundary variance far below one; the tuned factor (about 1/4)
    keeps it within about 30%."""
    r = ngsrng.tune_robin(square, 0.5, 5.0)
    assert 0.15 < r.factor < 0.45
    assert r.rms < 0.5 * r.default_rms and r.default_range[0] < 0.4 < r.variance_range[0]
    assert "Robin factor" in str(r)


def test_tune_option_of_matern_correlation(square):
    """robin="tune" binds with the tuned factor, tunes once per space (the result travels with pickled copies, e.g.
    to worker processes), and works through a Generator."""
    corr = ngsrng.matern_correlation(0.5, 1.0, boundary="robin", robin="tune")
    spec = corr.bind(square)
    r = corr.robin_tuning(square)
    assert spec.form.beta == pytest.approx(r.factor * corr.kappa())
    assert 0.5 < r.factor < 0.9  # near 1/1.42 for nu = 1 in 2D
    assert corr.robin_tuning(square) is r and corr.robin_factor(square) == r.factor
    copy = pickle.loads(pickle.dumps(corr))
    robin._CACHE.clear()
    assert copy.robin_tuning(square) == r  # no new tuning: robin_tuning would otherwise rebuild Gaussians
    assert copy == corr and "_tuned" not in repr(corr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = ngsrng.default_rng(square, seed=1, correlation=corr, dtype="float64").normal(size=2)
    assert np.asarray(y).shape == (2, square.ndof)
    assert ngsrng.matern_correlation(0.5, 1.0, boundary="robin", robin=0.3).robin_factor(square) == 0.3


def test_tuning_ignores_dirichlet_boundaries():
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.1))
    points, _ = robin._layer_probes(ngs.H1(mesh, order=1, dirichlet="left"), 0.2, 10_000)
    assert points[:, 0].min() >= 0.2 - 1e-12  # nothing within one length of the Dirichlet side
    with pytest.raises(ValueError, match="Dirichlet"):
        ngsrng.tune_robin(ngs.H1(mesh, order=1, dirichlet=".*"), 0.2)


def test_probe_subsample_keeps_boundary_points():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.05)), order=1)
    points, weights = robin._layer_probes(fes, 0.3, 100)
    on = np.isclose(points, 0).any(1) | np.isclose(points, 1).any(1)
    assert len(points) == 100 and on.sum() == 50
    assert weights.sum() == pytest.approx(1 - 0.4**2, rel=0.02)  # area of the layer: estimated by the subsample


def test_invalid_options():
    with pytest.raises(ValueError, match="boundary='robin'"):
        ngsrng.matern_correlation(0.3, robin="tune")
    with pytest.raises(ValueError, match="'tune'"):
        ngsrng.matern_correlation(0.3, boundary="robin", robin="auto")
    with pytest.raises(ValueError, match="nonnegative"):
        ngsrng.matern_correlation(0.3, boundary="robin", robin=-1.0)
