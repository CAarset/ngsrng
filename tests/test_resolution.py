"""The check that a mesh resolves a Matérn field's correlation length (ngsrng.resolution): the measured variance
errors (structured periodic meshes: the same on every machine), the periodised continuous variance, and the check
where fields are made (correlations bound to a space, StationaryMatern, random lengths at their smallest value)."""

import ngsolve as ngs
import pytest
import scipy.stats as st

import ngsrng
import ngsrng.resolution as res


@pytest.fixture(autouse=True)
def checking(monkeypatch):
    monkeypatch.setattr(res, "RESOLUTION_TOL", 0.01)  # conftest turns the check off for the other tests


@pytest.fixture(scope="module")
def coarse():
    return ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2)), order=2)


def test_periodised_variance():
    """A box much wider than the correlation length has variance 1; a narrow one adds its images' correlation."""
    assert res.periodised_variance(20.0, 1.0, 1.0, 2) == pytest.approx(1.0, abs=1e-12)
    assert res.periodised_variance(1.0, 1.0, 1.0, 1) > 1.2


@pytest.mark.parametrize("order, nu, per_length, low, high", [
    (2, 1.0, 4, 0.025, 0.04), (2, 1.0, 8, 0.006, 0.012), (2, 5.0, 3, 0.008, 0.015), (1, 0.5, 16, 0.06, 0.11)])
def test_variance_errors(order, nu, per_length, low, high):
    """The variance error decreases with the elements per correlation length, faster for smoother fields and higher
    orders (rough fields converge slowly)."""
    err = res.resolution_error(1.0 / per_length, order, 2, 1.0, nu, 10**5)
    assert low < err < high


def test_check_where_fields_are_made(coarse):
    """An under-resolved length raises (with the error and an element size that would pass) for correlations bound
    to a space and for StationaryMatern; resolution_tol=None turns the check off; a resolved length passes."""
    with pytest.raises(ValueError, match=r"does not resolve.*would pass"):
        ngsrng.default_rng(coarse, correlation=ngsrng.matern_correlation(0.2, nu=1)).gaussian()
    with pytest.raises(ValueError, match="does not resolve"):
        ngsrng.StationaryMatern(coarse, 0.2, 1.0, construction="torus")
    ngsrng.default_rng(coarse, correlation=ngsrng.matern_correlation(0.2, nu=1, resolution_tol=None)).gaussian()
    ngsrng.default_rng(coarse, correlation=ngsrng.matern_correlation(2.0, nu=2)).gaussian()


def test_random_lengths_are_checked_at_their_smallest(coarse):
    rng = ngsrng.default_rng(coarse, seed=0)
    with pytest.raises(ValueError, match="does not resolve"):
        rng.distribution("uniform", correlation=ngsrng.matern_correlation(length=st.loguniform(0.2, 3.0), nu=2,
                                                                           boundary="torus"))
    rng.distribution("uniform", correlation=ngsrng.matern_correlation(length=st.loguniform(1.5, 3.0), nu=2,
                                                                       boundary="torus"))
