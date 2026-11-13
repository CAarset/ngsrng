"""Estimating the Robin coefficient per space (ngsrng.tune_robin, matern_correlation(..., beta=None))."""

import pickle
import warnings

import ngsolve as ngs
import numpy as np
import pytest
from netgen.occ import Circle, OCCGeometry
from ngsolve.meshes import Make1DMesh

import ngsrng
from ngsrng import robin


@pytest.fixture(scope="module")
def square():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.08)), order=2)


def test_exponential_covariance_needs_beta_equal_kappa():
    """nu = 1/2 in 1D is the Ornstein-Uhlenbeck process, stationary on an interval exactly for beta = kappa: the
    estimate is beta = kappa with a flat variance."""
    r = ngsrng.tune_robin(ngs.H1(Make1DMesh(200), order=2), 0.2, 0.5, compare=True)
    assert abs(r.factor - 1) < 0.02 and r.rms < 0.01
    assert r.classical_rms > 5 * r.rms and abs(r.beta - r.factor * r.kappa) < 1e-12


def test_interior_level_keeps_discretisation_offsets_out_of_beta():
    """A coarse P1 Ornstein-Uhlenbeck field has a variance below one between the nodes, everywhere: matching the
    interior level (not one) keeps beta near kappa, where the variance is constant (aiming at one gave 0.71 kappa)."""
    r = ngsrng.tune_robin(ngs.H1(Make1DMesh(50), order=1), 0.2, 0.5)
    assert r.reference == "interior" and r.level < 0.97 and r.factor > 0.9


def test_smooth_fields_get_a_smaller_coefficient(square):
    """nu = 5 in 2D: the classical kappa / 1.42 pulls the boundary variance far below one; the estimate (about
    0.30 kappa) keeps it within about 40%."""
    r = ngsrng.tune_robin(square, 0.5, 5.0, compare=True)
    assert 0.25 < r.factor < 0.35
    assert r.rms < 0.5 * r.classical_rms and r.classical_range[0] < 0.4 < r.variance_range[0]
    assert "Robin beta" in str(r) and "kappa / 1.42" in str(r)


def test_estimate_does_not_depend_on_the_vertices():
    """Where the variance is resolved, coarse and fine meshes give the same estimate (vertex-based quadrature made
    it 13% apart for nu = 5 between h = 0.1 and 0.05: the boundary dip, ~1/(2 kappa) wide, was counted over h/2)."""
    c = [ngsrng.tune_robin(ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=h)), order=2), 0.3, 5.0).factor
         for h in (0.1, 0.05)]
    assert abs(c[0] / c[1] - 1) < 0.02 and 0.29 < c[1] < 0.305


def test_layer_quadrature_integrates_the_layer():
    """The weights sum to the layer's area: unit square with a Dirichlet side (left; the layer is what lies within
    length of the other sides and not within length of the left one), a curved disk (an annulus) and a cube."""
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.1))
    pts, w, edge, inner, inner_w = robin._layer_quadrature(ngs.H1(mesh, order=2, dirichlet="left"), 0.3, 10.0, 1.0,
                                                           4000)
    assert w.sum() == pytest.approx(0.7 - 0.4**2, rel=3e-3) and (w > 0).all()  # the background resolves the cut
    assert inner_w.sum() == pytest.approx(0.4**2, rel=3e-3) and np.abs(inner - 0.5).max() < 0.2  # the interior
    assert pts[:, 0].min() >= 0.3 and np.minimum(np.minimum(pts[:, 1], 1 - pts[:, 1]), 1 - pts[:, 0]).max() < 0.3
    assert {(1.0, 0.0), (1.0, 1.0)} <= {tuple(p) for p in np.round(edge[:2], 9)}  # the free corners come first
    disk = ngs.Mesh(OCCGeometry(Circle((0, 0), 1).Face(), dim=2).GenerateMesh(maxh=0.2)).Curve(3)
    _, w, _, _, inner_w = robin._layer_quadrature(ngs.H1(disk, order=2), 0.3, 10.0, 1.0, 4000)
    assert w.sum() == pytest.approx(np.pi * (1 - 0.7**2), rel=3e-3)
    assert inner_w.sum() == pytest.approx(np.pi * 0.7**2, rel=3e-3)
    cube = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.35))  # 3D: depths to the samples' tangent planes
    _, w, edge, _, inner_w = robin._layer_quadrature(ngs.H1(cube, order=1), 0.3, 4.0, 1.0, 4000)
    assert w.sum() == pytest.approx(1 - 0.4**3, rel=1e-3) and inner_w.sum() == pytest.approx(0.4**3, rel=1e-2)
    assert (np.abs(edge[:8] - 0.5) == 0.5).all()  # the corners first
    *_, inner, _ = robin._layer_quadrature(ngs.H1(mesh, order=1), 0.5, 10.0, 1.0, 4000)
    assert len(inner) == 0  # no point is farther than 0.5 from the boundary: the level is the nominal one
    with pytest.raises(ValueError, match="Dirichlet"):
        ngsrng.tune_robin(ngs.H1(mesh, order=1, dirichlet=".*"), 0.2)


def test_device_and_engine_do_not_change_the_estimate(square):
    """The torch engine (the GPU's code path, here on the CPU) gives the CPU engine's estimate."""
    a = ngsrng.tune_robin(square, 0.5, 1.0, device="cpu")  # "auto" would take a GPU where one is usable
    b = ngsrng.tune_robin(square, 0.5, 1.0, device="cpu", method={"engine": "torch"})
    assert b.beta == pytest.approx(a.beta, rel=2e-3) and a.device == "cpu"
    assert ngsrng.tune_robin(square, 0.5, 1.0, device="cpu") is a  # cached per device


def test_estimated_beta_of_matern_correlation(square):
    """beta=None binds with the estimated coefficient, estimates once per space (the result travels with pickled
    copies, e.g. to worker processes), and works through a Generator."""
    corr = ngsrng.matern_correlation(0.5, 1.0, boundary="robin", beta=None)
    spec = corr.bind(square)
    r = corr.robin_tuning(square)
    assert spec.form.beta == pytest.approx(r.beta)
    assert 0.6 < r.factor < 0.8  # near 1/1.42 for nu = 1 in 2D
    assert corr.robin_tuning(square) is r and corr.robin_beta(square) == r.beta
    copy = pickle.loads(pickle.dumps(corr))
    robin._CACHE.clear()
    assert copy.robin_tuning(square) == r  # no new estimate: robin_tuning would otherwise rebuild Gaussians
    assert copy == corr and "_tuned" not in repr(corr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = ngsrng.default_rng(square, seed=1, correlation=corr, dtype="float64").normal(size=2)
    assert np.asarray(y).shape == (2, square.ndof)
    given = ngsrng.matern_correlation(0.5, 1.0, boundary="robin", beta=3.0)
    assert given.robin_beta(square) == 3.0 and given.bind(square).form.beta == 3.0


def test_robin_coefficient_has_no_default():
    """boundary='robin' without beta is an error that explains beta=None; beta without boundary='robin' as well."""
    for make in (lambda: ngsrng.matern_correlation(0.3, boundary="robin"),
                 lambda: ngsrng.MaternCorrelation(0.3, boundary="robin"),
                 lambda: ngsrng.default_rng(ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.5))),
                                            correlation={"length": 0.3, "boundary": "robin"}).gaussian()):
        with pytest.raises(ValueError, match="beta=None to estimate"):
            make()
    with pytest.raises(ValueError, match="boundary='robin'"):
        ngsrng.matern_correlation(0.3, beta=None)
    with pytest.raises(ValueError, match="nonnegative"):
        ngsrng.matern_correlation(0.3, boundary="robin", beta=-1.0)
    with pytest.raises(TypeError, match="replaced by beta="):
        ngsrng.matern_correlation(0.3, boundary="robin", robin=0.7)
    with pytest.raises(ValueError, match="was removed"):
        ngsrng.matern(5.0, robin="auto")
    neumann = ngsrng.matern_correlation(0.3)
    assert pickle.loads(pickle.dumps(neumann)) == neumann  # the "required" marker survives pickling
