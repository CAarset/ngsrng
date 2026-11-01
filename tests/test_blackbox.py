import ngsolve as ngs
import numpy as np
import pytest
import scipy.linalg as sla
import torch

import ngsrng
from ngsrng.blackbox import MultiShiftPCG, inverse_iteration_min, lanczos_interval
from ngsrng.rational import fractional_power
from ngsrng.reference import covariance_errors, exact_covariance, sampler_covariance
from ngsrng.space import SpaceInfo


@pytest.fixture(scope="module")
def bilaplacian():
    """Space and dense K, M of kappa^2 - Laplace (H1, Dirichlet on one side); Q = K M^-1 K is the precision."""
    s = SpaceInfo(ngs.H1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.3)), order=2, dirichlet="left"))
    k = s.assemble_free(lambda u, v: (ngs.grad(u) * ngs.grad(v) + 9 * u * v) * ngs.dx).toarray()
    m = s.mass_free.toarray()
    return s, k, m, k @ np.linalg.solve(m, k)


def test_multishift_pcg_all_shifts(bilaplacian):
    _, _, m, q = bilaplacian
    qt, mi = torch.from_numpy(q), torch.from_numpy(np.linalg.inv(m))
    r = fractional_power(0.5, 10.0, 1e8, rtol=1e-7)  # shifts spanning many decades (underflow regression)
    b = torch.randn(q.shape[0], 3, dtype=torch.float64)
    cg = MultiShiftPCG(lambda x: qt @ x, lambda x: mi @ x, r.shifts, r.coeffs, tol=1e-11)
    ref = sum(c * np.linalg.solve(q + t * m, b.numpy()) for c, t in zip(r.coeffs, r.shifts))
    out = cg.solve(b).numpy()
    assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 1e-9


def test_spectral_bounds(bilaplacian):
    _, _, m, q = bilaplacian
    qt, mt, mi = torch.from_numpy(q), torch.from_numpy(m), torch.from_numpy(np.linalg.inv(m))
    lam = sla.eigh(q, m, eigvals_only=True)
    lmin = inverse_iteration_min(lambda x: qt @ x, lambda x: mt @ x, lambda x: mi @ x, q.shape[0])
    assert lam[0] <= lmin <= 1.01 * lam[0]
    _, lmax, res = lanczos_interval(lambda x: qt @ x, lambda x: mi @ x, q.shape[0], iters=60)
    assert lmax <= lam[-1] * (1 + 1e-12) and lmax + res >= lam[-1] * (1 - 1e-6)


@pytest.mark.parametrize("convention", ["primal->dual", "dual->dual"])
def test_operator_precision_is_exact(bilaplacian, convention):
    s, _, m, q = bilaplacian
    op = q if convention == "primal->dual" else q @ np.linalg.inv(m)
    opt = torch.from_numpy(op)
    spec = ngsrng.OperatorPrecision(lambda c: opt @ c, convention=convention)
    g = ngsrng.Gaussian(s.fes, spec, dtype="float64", method={"engine": "torch"})  # batched check
    assert "apply-only" in g.plan["structure"]
    err = covariance_errors(sampler_covariance(g), np.linalg.inv(q))
    assert err["rel_fro"] < 1e-6 and err["max_rel_var"] < 1e-6, err
    np.testing.assert_allclose(exact_covariance(g), np.linalg.inv(q), rtol=1e-8, atol=1e-14)


def test_multishift_pcg_zero_column(bilaplacian):
    """A zero right-hand side column used to divide 0 / 0 and turn into NaN."""
    _, _, m, q = bilaplacian
    qt, mi = torch.from_numpy(q), torch.from_numpy(np.linalg.inv(m))
    r = fractional_power(0.5, 10.0, 1e6, rtol=1e-7)
    b = torch.randn(q.shape[0], 3, dtype=torch.float64)
    b[:, 0] = 0.0
    out = MultiShiftPCG(lambda x: qt @ x, lambda x: mi @ x, r.shifts, r.coeffs, tol=1e-11).solve(b)
    assert torch.isfinite(out).all() and (out[:, 0] == 0).all()
    ref = sum(c * np.linalg.solve(q + t * m, b.numpy()[:, 1:]) for c, t in zip(r.coeffs, r.shifts))
    assert np.linalg.norm(out[:, 1:].numpy() - ref) / np.linalg.norm(ref) < 1e-9
