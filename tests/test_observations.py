import ngsolve as ngs
import numpy as np

from ngsrng import observation_operator


def test_point_observations_are_exact():
    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=3)
    gf = ngs.GridFunction(fes)
    gf.Set(ngs.x * ngs.y + ngs.x**2)
    pts = np.array([[0.3, 0.4], [0.71, 0.2]])
    g = observation_operator(fes, points=pts)
    np.testing.assert_allclose(g @ gf.vec.FV().NumPy(), pts[:, 0] * pts[:, 1] + pts[:, 0] ** 2, rtol=1e-12)
    h = observation_operator(fes, functionals=[ngs.CoefficientFunction(1.0)])
    assert abs((h @ gf.vec.FV().NumPy())[0] - (1 / 4 + 1 / 3)) < 1e-12


def test_general_observation_maps():
    """Any finite set of bounded linear functionals: averages, LinearOperator, adjoint callable, boundary forms."""
    import scipy.sparse.linalg as sla

    fes = ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.1)), order=2)
    one = ngs.GridFunction(fes)
    one.Set(1.0)
    for kernel in ("ball", "gaussian"):
        g = observation_operator(fes, averages=([[0.5, 0.5], [0.2, 0.7]], 0.15, kernel))
        np.testing.assert_allclose(g @ one.vec.FV().NumPy(), 1.0, rtol=1e-12)
    dense = observation_operator(fes, points=[[0.3, 0.3], [0.6, 0.1]]).toarray()
    op = sla.aslinearoperator(dense)
    np.testing.assert_allclose(observation_operator(fes, matrix=op).toarray(), dense, atol=1e-14)
    np.testing.assert_allclose(
        observation_operator(fes, adjoint=lambda y: dense.T @ y, m=2).toarray(), dense, atol=1e-14
    )
    lf = ngs.LinearForm(fes.TestFunction() * ngs.ds("right"))
    assert abs((observation_operator(fes, functionals=[lf]) @ one.vec.FV().NumPy())[0] - 1.0) < 1e-12
