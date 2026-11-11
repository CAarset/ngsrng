"""The bundled test geometries (ngsrng.domains) build curved NGSolve meshes."""

import math

import ngsolve as ngs
import pytest

from ngsrng import domains

SHAPES = [name for name in domains.__all__ if name != "NGSGeo"]


@pytest.mark.parametrize("name", SHAPES)
def test_domain_builds(name):
    shape = getattr(domains, name)(maxh=0.15)
    assert shape.mesh.ne > 0 and shape.mesh.dim == 2
    # outer_bnd is usable as dirichlet=: every name it lists is a boundary of the mesh
    assert set(shape.outer_bnd.split("|")) <= set(shape.mesh.GetBoundaries())


def test_gappy_outer_boundary_is_the_square():
    mesh = domains.NGSGappy(maxh=0.1).mesh
    length = {b: ngs.Integrate(1, mesh, ngs.BND, definedon=mesh.Boundaries(b)) for b in set(mesh.GetBoundaries())}
    assert length["outer"] == pytest.approx(4.0) and length["interface"] == pytest.approx(0.4)


@pytest.mark.parametrize("size", [0.1, 0.5])
def test_circle_size_is_the_inner_radius(size):
    mesh = domains.NGSCircle(size=size, maxh=0.1).mesh
    assert ngs.Integrate(1, mesh.Materials("inner")) == pytest.approx(math.pi * size**2, rel=1e-3)
