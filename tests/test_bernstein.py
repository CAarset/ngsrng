"""The Bernstein-Bezier representation: exact algebra, and range guarantees at every point of the FE function."""

import netgen.geom2d as g2
import ngsolve as ngs
import numpy as np
import pytest
import scipy.stats as st
from ngsolve.meshes import Make1DMesh, MakeStructured2DMesh, MakeStructured3DMesh

import ngsrng
from ngsrng.distributions.bernstein import BernsteinForm
from ngsrng.space import SpaceInfo


def _disc():
    geo = g2.SplineGeometry()
    geo.AddCircle((0, 0), 1, bc="c")
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.4))
    mesh.Curve(4)
    return mesh


SPACES = {
    "1D-H1-p4": lambda: ngs.H1(Make1DMesh(12), order=4),
    "2D-H1-p1": lambda: ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=1),
    "2D-H1-p3-dirichlet": lambda: ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=3, dirichlet="left"),
    "2D-H1-p5-curved": lambda: ngs.H1(_disc(), order=5),
    "2D-L2-p0": lambda: ngs.L2(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=0, dgjumps=True),
    "2D-L2-p2": lambda: ngs.L2(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=2, dgjumps=True),
    "2D-H1-quads-p3": lambda: ngs.H1(
        MakeStructured2DMesh(quads=True, nx=5, ny=5, mapping=lambda x, y: (x + 0.2 * y * y, y)), order=3
    ),
    "3D-H1-tets-p2": lambda: ngs.H1(ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5)), order=2),
    "3D-H1-hexes-p2": lambda: ngs.H1(MakeStructured3DMesh(hexes=True, nx=3, ny=3, nz=3), order=2),
}


def dense_values(fes, coeffs, nref=150, seed=0):
    """Values of FE functions (rows of ``coeffs``) at ~nref random reference points in every element."""
    mesh, d = fes.mesh, fes.mesh.dim
    r = np.random.default_rng(seed).uniform(0, 1, (nref, d))
    if next(iter(mesh.Elements(ngs.VOL))).type in (ngs.ET.TRIG, ngs.ET.TET):
        r = r[r.sum(1) <= 1]
    mp = mesh.MapToAllElements(ngs.IntegrationRule([tuple(x) + (0.0,) * (3 - d) for x in r], [1.0] * len(r)), ngs.VOL)
    gf = ngs.GridFunction(fes)
    out = []
    for c in np.atleast_2d(coeffs):
        gf.vec.FV().NumPy()[:] = c
        out.append(np.asarray(gf(mp)).ravel())
    return np.array(out)


@pytest.mark.parametrize("name", SPACES)
def test_bernstein_form_algebra(name):
    fes = SPACES[name]()
    form = BernsteinForm(SpaceInfo(fes))
    assert form.npoints == fes.ndof
    # partition of unity: all coefficients one -> the constant one
    ones, resid = form.to_coefficients(np.ones((form.npoints, 1)))
    np.testing.assert_allclose(dense_values(fes, ones[:, 0]), 1.0, atol=1e-10)
    assert resid.max() < 1e-10
    # the evaluation matrix gives the values at the domain points
    gf = ngs.GridFunction(fes)
    gf.Set(ngs.sin(3 * ngs.x) + ngs.cos(2 * ngs.y if fes.mesh.dim > 1 else ngs.x))
    np.testing.assert_allclose(form.evaluation @ gf.vec.FV().NumPy(), form.cf_values(gf), atol=1e-10)


def test_bernstein_reproduces_affine_functions():
    """Q_h f = f for affine f on straight elements (coefficients = values at the domain points)."""
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=3)
    form = BernsteinForm(SpaceInfo(fes))
    f = 1 + ngs.x - 2 * ngs.y
    c, _ = form.to_coefficients(form.cf_values(f)[:, None])
    exact = ngs.GridFunction(fes)
    exact.Set(f)
    np.testing.assert_allclose(c[:, 0], exact.vec.FV().NumPy(), atol=1e-11)


@pytest.mark.parametrize("name", SPACES)
def test_uniform_stays_inside_everywhere(name):
    fes = SPACES[name]()
    rng = ngsrng.default_rng(fes, seed=1, correlation=0.4, dtype="float32")
    u = rng.uniform(size=20, representation="bernstein")
    assert u.dtype == np.float64  # the guarantee is about these exact coefficients
    v = dense_values(fes, u)
    assert v.min() > 0 and v.max() < 1
    assert isinstance(rng.uniform(representation="bernstein"), ngs.GridFunction)
    g = dense_values(fes, rng.gamma(0.3, size=5, representation="bernstein"))
    assert g.min() > 0


def test_bernstein_marginals_and_other_ranges():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.15)), order=2, dirichlet="left")
    rng = ngsrng.default_rng(fes, seed=2, correlation=0.4, dtype="float64")
    u = rng.uniform(size=2000, representation="bernstein")
    # NGSolve's first nv dofs are the vertex values, and Q_h f equals f at the vertices: exact uniform marginals
    free_vertices = [v for v in range(fes.mesh.nv) if fes.FreeDofs()[v]]
    for i in free_vertices[:: max(1, len(free_vertices) // 4)][:4]:
        assert st.kstest(u[:, i], st.uniform().cdf).pvalue > 1e-4
    left = [v.nr for v in fes.mesh.vertices if abs(v.point[0]) < 1e-12]
    np.testing.assert_allclose(u[:, left], 0.5)  # a deterministic (Dirichlet) point maps to the median
    dirichlet = rng.dirichlet([0.5, 1.0, 2.0], size=10, representation="bernstein")
    comps = np.stack([dense_values(fes, dirichlet[:, :, k]) for k in range(3)])
    assert comps.min() > 0 and np.abs(comps.sum(0) - 1).max() < 1e-12
    w = rng.wishart(3, np.eye(2), size=5, representation="bernstein")
    e = {(a, b): dense_values(fes, w[:, :, a, b]) for a in range(2) for b in range(2)}
    assert min(e[0, 0].min(), e[1, 1].min()) > 0 and (e[0, 0] * e[1, 1] - e[0, 1] * e[1, 0]).min() > 0
    for name, args, hi in (("poisson", (3.0,), np.inf), ("bernoulli", (0.3,), 1.0), ("indicator", (0.5,), 1.0)):
        v = dense_values(fes, rng.sample(name, *args, size=10, representation="bernstein"))
        assert v.min() >= 0 and v.max() <= hi


def test_spatially_varying_parameters_in_bernstein_form():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2)
    rng = ngsrng.default_rng(fes, seed=3, correlation=0.3, dtype="float64")
    for name, args in (("bernoulli", (ngs.x,)), ("poisson", (1.0 + 5.0 * ngs.y,)), ("uniform", (0.0, 1.0 + ngs.x))):
        v = dense_values(fes, rng.sample(name, *args, size=5, representation="bernstein"))
        assert v.min() >= 0 and np.isfinite(v).all()
