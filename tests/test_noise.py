import ngsolve as ngs
import numpy as np
import pytest
import torch
from ngsolve.meshes import Make1DMesh, MakeStructured2DMesh

from ngsrng import noise as nz
from ngsrng.space import SpaceInfo

CPU64 = (torch.device("cpu"), torch.float64)


def _mapped_quads():
    return MakeStructured2DMesh(quads=True, nx=3, ny=3, mapping=lambda x, y: (x + 0.3 * x * y, y))


SPACES = {
    "1D-H1-p3-dir": lambda: ngs.H1(Make1DMesh(6), order=3, dirichlet="left"),
    "2D-H1-p1-dir": lambda: ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=1, dirichlet="left|bottom"),
    "2D-H1-p4-dir": lambda: ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=4, dirichlet="left|bottom"),
    "3D-H1-p3-dir": lambda: ngs.H1(ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5)), order=3, dirichlet="left"),
    "2D-H1-p3-mapped": lambda: ngs.H1(_mapped_quads(), order=3),
    "2D-L2-p3-mapped": lambda: ngs.L2(_mapped_quads(), order=3),
    "3D-L2-p2": lambda: ngs.L2(ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5)), order=2),
    "2D-L2-p2-dgjumps": lambda: ngs.L2(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=2, dgjumps=True),
}


def _cov_error(sampler, space):
    sampler.setup(*CPU64)
    h = sampler.factor().toarray()
    m = space.mass_free.toarray()
    return np.abs(h @ h.T - m).max() / np.abs(m).max()


@pytest.fixture(params=list(SPACES), scope="module")
def space(request):
    return SpaceInfo(SPACES[request.param]())


@pytest.mark.parametrize("kind", ["local-gram", "local-gram-explicit", "quadrature", "dense-cholesky"])
def test_exact_white_noise(space, kind):
    sampler = {
        "local-gram": lambda: nz.LocalGramNoise(space),
        "local-gram-explicit": lambda: nz.LocalGramNoise(space).explicit(),
        "quadrature": lambda: nz.quadrature_noise(space),
        "dense-cholesky": lambda: nz.dense_cholesky_noise(space.mass_free),
    }[kind]()
    assert _cov_error(sampler, space) < 1e-12


def test_chebyshev_noise_meets_tolerance(space):
    assert _cov_error(nz.ChebyshevNoise(space.mass_free, tol=1e-8), space) < 1e-7


def test_diagonal_noise_when_mass_is_diagonal(space):
    if not space.mass_is_diagonal:
        pytest.skip("mass matrix not diagonal")
    assert _cov_error(nz.DiagonalNoise(space.mass_free), space) < 1e-12


def test_reference_mode_on_affine_mesh():
    mesh = ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.4))
    noise = nz.LocalGramNoise(SpaceInfo(ngs.H1(mesh, order=4)))
    assert noise.n_batched_elements == 0
    assert noise.n_ref_elements == mesh.ne


def test_dirichlet_dofs_draw_no_random_numbers():
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2))
    full = nz.LocalGramNoise(SpaceInfo(ngs.H1(mesh, order=3)))
    dirichlet = nz.LocalGramNoise(SpaceInfo(ngs.H1(mesh, order=3, dirichlet=".*")))
    assert dirichlet.nrand < full.nrand


def test_stiffness_noise_is_exact():
    """Local-Gram noise for a locally PSD fourth-order-type form reproduces the assembled precision."""
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
    space = SpaceInfo(ngs.H1(mesh, order=3, dirichlet="left"))

    def form(u, v):
        hess = ngs.InnerProduct(u.Operator("hesse"), v.Operator("hesse"))
        return (hess + 0.1 * ngs.grad(u) * ngs.grad(v) + 0.01 * u * v) * ngs.dx

    q = space.assemble_free(form).toarray()
    sampler = nz.LocalGramNoise(space, form=form).setup(*CPU64)
    h = sampler.factor().toarray()
    assert np.abs(h @ h.T - q).max() / np.abs(q).max() < 1e-12


def test_dg_form_is_rejected_by_local_gram():
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3))
    fes = ngs.L2(mesh, order=1, dgjumps=True)

    def form(u, v):
        return u * v * ngs.dx + (u - u.Other()) * (v - v.Other()) * ngs.dx(skeleton=True)

    with pytest.raises(RuntimeError):
        nz.LocalGramNoise(SpaceInfo(fes), form=form)


def test_quadrature_noise_rejects_inexact_rules():
    """Regression: the self-check used to be diagonal-only and skipped below order 2p, i.e. exactly where it fails
    (P2 at order 1: ~1e-2 relative covariance error)."""
    space = SpaceInfo(ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=2))
    with pytest.raises(RuntimeError, match="not exact"):
        nz.quadrature_noise(space, order=1)
    # the rule of order o has degree 2 o: order p is already exact
    assert _cov_error(nz.quadrature_noise(space, order=2), space) < 1e-12


def test_factor_error_sees_off_diagonal_errors():
    space = SpaceInfo(ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=1))
    m = space.mass_free
    lumped = nz.LumpedNoise(m)
    assert nz.factor_error(nz.dense_cholesky_noise(m).h, m) < 1e-12
    assert nz.factor_error(lumped.h, m) > 1e-2
    dense = m.toarray()
    assert lumped.rel_error == pytest.approx(np.abs(np.diag(dense.sum(axis=1)) - dense).max() / np.abs(dense).max())


def _mixed_mesh(nx=4, ny=3):
    """Unit square: quadrilaterals on the left half, triangles on the right; boundaries "left" and "rest"."""
    import netgen.meshing as nm

    mesh = nm.Mesh(dim=2)
    p = {(i, j): mesh.Add(nm.MeshPoint(nm.Pnt(i / nx, j / ny, 0))) for j in range(ny + 1) for i in range(nx + 1)}
    dom = mesh.AddRegion("dom", dim=2)
    left, rest = mesh.AddRegion("left", dim=1), mesh.AddRegion("rest", dim=1)
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = p[i, j], p[i + 1, j], p[i + 1, j + 1], p[i, j + 1]
            if i < nx // 2:
                mesh.Add(nm.Element2D(dom, [a, b, c, d]))
            else:
                mesh.Add(nm.Element2D(dom, [a, b, c]))
                mesh.Add(nm.Element2D(dom, [a, c, d]))
    for j in range(ny):
        mesh.Add(nm.Element1D([p[0, j + 1], p[0, j]], index=left))
        mesh.Add(nm.Element1D([p[nx, j], p[nx, j + 1]], index=rest))
    for i in range(nx):
        mesh.Add(nm.Element1D([p[i, 0], p[i + 1, 0]], index=rest))
        mesh.Add(nm.Element1D([p[i + 1, ny], p[i, ny]], index=rest))
    return ngs.Mesh(mesh)


@pytest.mark.parametrize(
    "make",
    [lambda: SPACES["2D-H1-p4-dir"](), lambda: SPACES["3D-H1-p3-dir"](), lambda: SPACES["2D-L2-p2-dgjumps"](),
     lambda: SPACES["2D-H1-p3-mapped"](), lambda: ngs.H1(_mixed_mesh(), order=2, dirichlet="left"),
     lambda: ngs.L2(_mixed_mesh(), order=1)],
    ids=["2D-H1-dir", "3D-H1-dir", "2D-L2-dgjumps", "mapped-quads", "mixed-H1-dir", "mixed-L2"],
)
@pytest.mark.parametrize("kind", ["local-gram", "local-gram-explicit", "quadrature"])
def test_element_of_column(make, kind):
    """Per-element operations (type-G mixing) scale noise column c by a variable of element element_of_column[c]
    (an NGSolve element number): every column must live on that element's dofs, and every element own columns."""
    space = SpaceInfo(make())
    if kind == "quadrature" and not space.single_element_type:
        pytest.skip("quadrature representations need one element type")
    sampler = {"local-gram": lambda: nz.LocalGramNoise(space).setup(*CPU64),
               "local-gram-explicit": lambda: nz.LocalGramNoise(space).explicit(),
               "quadrature": lambda: nz.quadrature_noise(space)}[kind]()
    h = sampler.factor().tocsc()
    emap = sampler.element_of_column
    assert emap.shape == (sampler.nrand,)
    assert set(emap) == set(range(space.mesh.ne))
    freepos = np.full(space.ndof, -1)
    freepos[space.free] = np.arange(space.nfree)
    owned = [set(freepos[list(el.dofs)]) - {-1} for el in space.fes.Elements(ngs.VOL)]
    for c in range(h.shape[1]):
        rows = h.indices[h.indptr[c]:h.indptr[c + 1]][np.abs(h.data[h.indptr[c]:h.indptr[c + 1]]) > 0]
        assert set(rows) <= owned[emap[c]], (c, emap[c])


def test_quadrature_noise_default_order_is_exact_on_curved_meshes():
    """The default rule (order p) is the one NGSolve assembles M with, also on curved elements; order 2p is not."""
    from netgen.occ import Circle, OCCGeometry

    mesh = ngs.Mesh(OCCGeometry(Circle((0, 0), 1).Face(), dim=2).GenerateMesh(maxh=0.5))
    mesh.Curve(3)
    space = SpaceInfo(ngs.H1(mesh, order=3))
    noise = nz.quadrature_noise(space)
    assert noise.quad_order == 3 and _cov_error(noise, space) < 1e-12
    with pytest.raises(RuntimeError, match="not exact"):
        nz.quadrature_noise(space, order=6)
