"""The bundled test geometries (ngsrng.domains) build curved NGSolve meshes."""

import pytest

from ngsrng import domains

SHAPES = [name for name in domains.__all__ if name != "NGSGeo"]


@pytest.mark.parametrize("name", SHAPES)
def test_domain_builds(name):
    mesh = getattr(domains, name)(maxh=0.15).mesh
    assert mesh.ne > 0 and mesh.dim == 2
