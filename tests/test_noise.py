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
