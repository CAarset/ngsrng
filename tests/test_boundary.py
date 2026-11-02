"""Tests for boundary (trace) sampling: ``bnd="name"`` on any ngsrng random function.

Mathematical content of each check:

* the trace of an FE function is a linear map on coefficients (row selection), so restricting a sample to the
  DoFs supported on the boundary is the exact marginal law of those DoFs -- test_restrict_is_marginal;
* for a precision C = A^-1 (power = 1) the boundary block of Sigma_h = K^-1 is the inverse of the assembled
  Dirichlet-to-Neumann (Schur complement) matrix, and the regression coefficient Sigma_IB Sigma_BB^-1 is the
  K-harmonic lifting -- test_dtn_theorem, test_lifting_is_harmonic;
* the boundary marginal of the sampler has the covariance Sigma_BB (Monte Carlo) -- test_boundary_moments;
* mode "field" rewrites the trace by u* = u - Sigma_B Sigma_BB^-1 (R u - g) with g ~ N(0, Sigma_BB) independent,
  which leaves the joint law of (u*, R u*) = (u, g) at the prior, and imposes g exactly -- test_field_mode_*;
* non-Gaussian (copula) families act pointwise, so the same restriction is exact for them -- test_copula_marginal.
"""

import ngsolve as ngs
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import ngsrng
from ngsrng.boundary import BoundarySampler, boundary_covariance, boundary_dofs
from ngsrng.space import SpaceInfo, assemble

MC = 4000


@pytest.fixture(scope="module")
def mesh():
    return ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.25))


@pytest.fixture(scope="module")
def mesh3():
    return ngs.Mesh(ngs.unit_cube.GenerateMesh(maxh=0.5))


def h1(mesh, order=1, dirichlet="right"):
    return ngs.H1(mesh, order=order, dirichlet=dirichlet)


def rng_for(fes, seed=7, **corr):
    return ngsrng.default_rng(fes, seed=seed, correlation=corr.pop("correlation", 0.25), **corr)


# ------------------------------------------------------------------ DoF discovery


def test_boundary_dofs_counts(mesh):
    """Lagrange p1: the 5 nodes on x=0; p2 adds the 4 edge midpoints; hierarchical interior DoFs never appear."""
    d1 = boundary_dofs(h1(mesh, 1), "left")
    d2 = boundary_dofs(h1(mesh, 2), "left")
    assert d1.size == 5
    assert set(d1.tolist()) < set(d2.tolist())
    assert d2.size == 9
    # every reported DoF has a non-zero trace, and every DoF with a non-zero trace is reported
    for fes, d in ((h1(mesh, 1), d1), (h1(mesh, 2), d2)):
        u, v = fes.TnT()
        a = ngs.BilinearForm(fes)
        a += ngs.InnerProduct(u, v) * ngs.ds("left")
        a.Assemble()
        vals, cols, ptr = a.mat.CSR()
        diag = sp.csr_matrix((np.array(vals), np.array(cols), np.array(ptr)),
                            shape=(fes.ndof, fes.ndof)).diagonal()
        assert np.all(diag[d] > 0)
        assert np.count_nonzero(diag) == d.size


def test_boundary_dofs_is_a_principal_block(mesh):
    """The reported DoFs are free DoFs, so Sigma_BB is a principal block of the coefficient covariance."""
    fes = h1(mesh, 2)
    d = boundary_dofs(fes, "left")
    free = SpaceInfo(fes).free
    assert np.all(np.isin(d, free))


def test_dirichlet_constrained_boundary_raises(mesh):
    """A space built with dirichlet="left" has no free trace DoFs: error, not a silent empty sample."""
    fes = ngs.H1(mesh, order=1, dirichlet="left|right")
    rng = rng_for(fes)
    with pytest.raises(ValueError, match="dirichlet"):
        rng.normal(size=3, bnd="left")


def test_l2_space_raises(mesh):
    fes = ngs.L2(mesh, order=1, dgjumps=True)
    rng = rng_for(fes)
    with pytest.raises(NotImplementedError, match="trace"):
        rng.normal(size=3, bnd="left")


def test_quadrature_representation_raises(mesh):
    """A point-values representation is not an FE function, so it has no DoF trace: say so, per call."""
    rng = ngsrng.default_rng(h1(mesh, 1), seed=3, correlation=0.25)
    with pytest.raises(ValueError, match="representation"):
        rng.uniform(size=3, bnd="left", representation="quadrature")
    # with the generator-level default (never asked for), bnd= switches to the coefficient representation and says so
    info = rng.info("uniform", bnd="left")
    assert info["bnd"]["mode"] == "restrict"
    assert info["representation"] == "projection"


# ------------------------------------------------------------------ exact marginal: restrict


def test_restrict_is_marginal(mesh):
    """Restricting is exactly taking the marginal: same seeds give the same numbers on the boundary DoFs."""
    fes = h1(mesh, 2)
    d = boundary_dofs(fes, "left")
    a = rng_for(fes).normal(size=50)
    b = rng_for(fes).normal(size=50, bnd="left")
    assert np.array_equal(b[:, d], a[:, d])
    other = np.setdiff1d(np.arange(fes.ndof), d)
    assert np.all(b[:, other] == 0.0)


def test_restrict_single_sample_is_gridfunction(mesh):
    fes = h1(mesh, 1)
    d = boundary_dofs(fes, "left")
    plain = rng_for(fes).normal()
    tr = rng_for(fes).normal(bnd="left")
    assert isinstance(tr, ngs.GridFunction)
    assert np.array_equal(tr.vec.FV().NumPy()[d], plain.vec.FV().NumPy()[d])
    other = np.setdiff1d(np.arange(fes.ndof), d)
    assert np.all(np.abs(tr.vec.FV().NumPy()[other]) == 0.0)


def test_restrict_off_diagonal_zero_and_covariance_block(mesh):
    """Empirical second moments on the boundary DoFs equal the boundary block of Sigma_h."""
    fes = h1(mesh, 1)
    rng = rng_for(fes)
    d = boundary_dofs(fes, "left")
    Sbb = boundary_covariance(rng.gaussian(), "left", fes=fes)
    vals = rng.normal(size=MC, bnd="left")[:, d]
    emp = np.cov(vals.T, bias=False)
    tol = 8 * np.abs(Sbb).max() / np.sqrt(MC)
    assert np.abs(emp - Sbb).max() < tol
    assert np.abs(emp - emp.T).max() < 1e-12


# ------------------------------------------------------------------ the DtN / Schur theorem


def test_dtn_theorem(mesh):
    """power = 1: Sigma_BB = (K_BB - K_BI K_II^-1 K_IB)^-1, the discrete Steklov-Poincare operator."""
    fes = h1(mesh, 2)
    form = lambda u, v: ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) * ngs.dx
    rng = ngsrng.default_rng(fes, seed=1, precision=ngsrng.PDEPrecision(form, power=1))
    d = boundary_dofs(fes, "left")
    Sbb = boundary_covariance(rng.gaussian(), "left", fes=fes)
    assert np.linalg.eigvalsh(Sbb).min() > 0

    s = SpaceInfo(fes)
    form = lambda u, v: ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) * ngs.dx
    K = sp.csc_matrix(assemble(fes, form)[s.free][:, s.free])
    kb = np.isin(s.free, d)
    Kii = sla.splu(K[~kb][:, ~kb].tocsc())
    # SuperLU takes a dense right-hand side; the solve is K_II^-1 K_IB, an n_int x n_bnd block with n_bnd small
    X = Kii.solve(np.asarray(K[~kb][:, kb].todense()))
    DtN = sp.csr_matrix(np.asarray(K[kb][:, kb].todense()) - np.asarray(sp.csr_matrix(K[kb][:, ~kb]) @ X))
    Schur_inv = sla.splu(DtN.tocsc()).solve(np.eye(d.size))
    assert np.abs(Sbb - 0.5 * (Schur_inv + Schur_inv.T)).max() < 1e-9 * np.abs(Sbb).max()


def test_lifting_is_harmonic(mesh):
    """power = 1: the regression coefficient Sigma_IB Sigma_BB^-1 is the K-harmonic lifting -K_II^-1 K_IB."""
    fes = h1(mesh, 2)
    form = lambda u, v: ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) * ngs.dx
    rng = ngsrng.default_rng(fes, seed=1, precision=ngsrng.PDEPrecision(form, power=1))
    gauss = rng.gaussian()
    d = boundary_dofs(fes, "left")
    sam = BoundarySampler(gauss, "left")
    s = SpaceInfo(fes)
    K = assemble(fes, lambda u, v: ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) * ngs.dx)[s.free][:, s.free]
    kb = np.isin(s.free, d)
    Kii = sla.splu(sp.csc_matrix(K[~kb][:, ~kb]))
    H_exact = -Kii.solve(K[~kb][:, kb].toarray())
    assert np.abs(sam.lifting[~kb] - H_exact).max() < 1e-8 * np.abs(H_exact).max()
    assert np.abs(sam.lifting[kb] - np.eye(d.size)).max() < 1e-12  # R P = I


def test_conditional_covariance_is_homogeneous_dirichlet(mesh):
    """power = 1: Cov(u_I | u_B) = K_II^-1, the covariance of the space with homogeneous data on Gamma."""
    fes = h1(mesh, 1)
    form = lambda u, v: ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) * ngs.dx
    rng = ngsrng.default_rng(fes, seed=1, precision=ngsrng.PDEPrecision(form, power=1))
    sam = BoundarySampler(rng.gaussian(), "left")
    s = SpaceInfo(fes)
    K = assemble(fes, lambda u, v: ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) * ngs.dx)[s.free][:, s.free]
    kb = np.isin(s.free, sam.dofs)
    Kii_inv = sla.splu(sp.csc_matrix(K[~kb][:, ~kb])).solve(np.eye((~kb).sum()))
    assert np.abs(sam.residual_covariance - Kii_inv).max() < 1e-8 * np.abs(Kii_inv).max()


# ------------------------------------------------------------------ field mode


def test_field_mode_is_prior_consistent(mesh):
    """Field mode keeps the joint law: full covariance equals the plain sampler's, trace equals the drawn g."""
    fes = h1(mesh, 1)
    d = boundary_dofs(fes, "left")
    other = np.setdiff1d(np.arange(fes.ndof), d)
    rng = rng_for(fes)
    plain = rng_for(fes).normal(size=MC)
    field = rng.normal(size=MC, bnd="left", bnd_mode="field")
    tol = 10 * np.abs(np.cov(plain.T, bias=False)).max() / np.sqrt(MC)
    assert np.abs(np.cov(field.T, bias=False) - np.cov(plain.T, bias=False)).max() < tol
    # the trace is a fresh independent draw with the right marginal, not a copy of the original trace
    assert np.abs(field[:, d] - plain[:, d]).max() > 1e-6
    assert np.abs(np.cov(field[:, d].T, bias=False) - boundary_covariance(rng.gaussian(), "left", fes=fes)).max() \
        < 8 * np.abs(np.cov(plain[:, d].T, bias=False)).max() / np.sqrt(MC)
    # and the field is not supported on the boundary (interior still random)
    assert np.abs(field[:, other]).max() > 0


def test_field_mode_prescribes_data(mesh):
    """Given boundary data, field mode is exact hard Dirichlet conditioning."""
    fes = h1(mesh, 2)
    # float64: the prescribed data must survive the delivered array's dtype untouched (the default float32 would
    # round it at 1e-7, which is not a failure of the conditioning). power = 1 makes the conditional mean exactly
    # the harmonic lifting of the data, so the mean check below has a closed form to compare against.
    import torch

    form = lambda u, v: ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) * ngs.dx
    rng = ngsrng.default_rng(fes, seed=7, precision=ngsrng.PDEPrecision(form, power=1), dtype=torch.float64)
    d = boundary_dofs(fes, "left")
    g = np.sin(3 * np.pi * np.linspace(0, 1, d.size))
    vals = rng.normal(size=20, bnd="left", bnd_values=g)
    assert np.abs(vals[:, d] - g).max() < 1e-12
    # interior mean over draws is the lifting of the data (mean of the conditional law)
    sam = BoundarySampler(rng.gaussian(), "left")
    emp_mean = vals.mean(axis=0)
    assert np.abs(emp_mean - sam.lift(np.zeros(fes.ndof), g)).max() \
        < 8 * np.abs(sam.lifting).max() * np.sqrt(np.abs(sam.residual_covariance).max()) / np.sqrt(20)


def test_field_mode_mean_shift(mesh):
    """The prior mean on the boundary is respected when no data is prescribed."""
    fes = h1(mesh, 1)
    d = boundary_dofs(fes, "left")
    rng = ngsrng.default_rng(fes, seed=11, correlation=0.25)
    vals = rng.normal(loc=2.0, size=600, bnd="left", bnd_mode="field")
    assert np.abs(vals[:, d].mean(axis=0) - 2.0).max() < 0.25


# ------------------------------------------------------------------ other families


def test_copula_marginal(mesh):
    """Pointwise maps commute with the trace, so copula families get exact boundary control too.

    bernstein is the representation whose DoF values carry the marginal law: a bernstein sample is a convex
    combination of quantiles of the target marginal, so the support constraint survives at the DoFs, hence at the
    trace DoFs. The projection representation delivers L2 projections, whose DoF values are smoothed averages and
    need neither stay in [0, 1] nor have the marginal moments (measured: DoFs range over [-0.44, 1.60] for
    uniform(0, 1)). What bnd= promises in either representation, and what the last block checks, is that
    restriction reproduces the marginal of the trace DoFs of that same representation.
    """
    fes = h1(mesh, 1)
    d = boundary_dofs(fes, "left")
    other = np.setdiff1d(np.arange(fes.ndof), d)
    vals = rng_for(fes).uniform(size=800, representation="bernstein", bnd="left")
    assert np.all(vals[:, d] >= 0) and np.all(vals[:, d] <= 1)
    assert np.all(vals[:, other] == 0.0)
    assert np.abs(vals[:, d].mean(axis=0) - 0.5).max() < 0.1
    assert np.abs(vals[:, d].std(axis=0) - 1 / np.sqrt(12)).max() < 0.05
    plain = rng_for(fes).uniform(size=800, representation="bernstein")
    tol = 8 * np.abs(np.cov(plain[:, d].T, bias=False)).max() / np.sqrt(800)
    assert np.abs(np.cov(vals[:, d].T, bias=False) - np.cov(plain[:, d].T, bias=False)).max() < tol


def test_projection_dofs_are_not_the_copula_marginal(mesh):
    """Documented asymmetry: the L2 projection of a copula field leaves the support at its DoFs."""
    fes = h1(mesh, 1)
    d = boundary_dofs(fes, "left")
    vals = rng_for(fes).uniform(size=400, representation="projection", bnd="left")
    assert vals[:, d].min() < 0.0 or vals[:, d].max() > 1.0
    # restriction itself is still exact: zero off Gamma, and the trace DoFs keep the law of the same representation
    assert np.all(vals[:, np.setdiff1d(np.arange(fes.ndof), d)] == 0.0)
    plain = rng_for(fes).uniform(size=400, representation="projection")
    assert np.array_equal(vals[:, d], plain[:, d])


def test_lognormal_and_student_on_boundary(mesh):
    fes = h1(mesh, 1)
    d = boundary_dofs(fes, "left")
    ln = rng_for(fes).lognormal(size=200, representation="bernstein", bnd="left")
    assert np.all(ln[:, d] > 0)  # the convex hull stays in the marginal support; a projection would not
    t = rng_for(fes).standard_t(df=4, size=200, bnd="left")
    plain = rng_for(fes).standard_t(df=4, size=200)
    assert np.array_equal(t[:, d], plain[:, d])  # restrict touches nothing on Gamma, whatever the representation


def test_posterior_boundary_sampling(mesh):
    """A posterior (Matheron) is a Gaussian: its boundary marginal is its own boundary block."""
    fes = h1(mesh, 1)
    d = boundary_dofs(fes, "left")
    rng = rng_for(fes)
    obs = np.array([[0.1, 0.5], [0.9, 0.5]])
    post = rng.gaussian().condition(points=obs, data=np.zeros(len(obs)), noise=1e-6)
    vals = ngsrng.default_rng(fes, seed=5, precision=post).normal(size=200, bnd="left")
    Sbb = boundary_covariance(post, "left", fes=fes)
    emp = np.cov(vals[:, d].T, bias=False)
    assert np.abs(emp - Sbb).max() < 10 * np.abs(Sbb).max() / np.sqrt(200)


def test_boundary_eigenvalue_decay(mesh):
    """gamma C gamma* is a pseudo-differential operator of order -(2 power - 1) on the (d-1)-dim boundary:
    its eigenvalues decay like j^-(2 power - 1) in 2D, i.e. trace class exactly when 2 power > d."""
    powers = (2, 3)
    slopes = {}
    for power in powers:
        fes = h1(mesh, 2)
        form = lambda uu, vv: ngs.InnerProduct(ngs.grad(uu), ngs.grad(vv)) * ngs.dx
        rng = ngsrng.default_rng(fes, seed=1, precision=ngsrng.PDEPrecision(form, power=power))
        w = BoundarySampler(rng.gaussian(), "left").trace_eigenvalues()
        assert w.min() > 0, w  # gamma C gamma* is positive definite on the trace space
        keep = w > w.max() * 1e-8
        j = np.arange(1, w.size + 1)[keep]
        slopes[power] = np.polyfit(np.log(j), np.log(w[keep]), 1)[0]
        assert slopes[power] < -1.0, (power, slopes)
    # a smoother field (larger power) has a more compact trace operator: order -(2 power - 1) on a (d-1)-face
    assert slopes[3] < slopes[2], slopes
    # The asymptotic exponent -(2 power - 1) needs a boundary mesh fine enough to resolve the bulk of the
    # spectrum; with a handful of trace DoFs only the qualitative behaviour is measurable. The exponent is
    # measured on a refined mesh.


def test_3d_face(mesh3):
    fes = ngs.H1(mesh3, order=1, dirichlet="right")
    d = boundary_dofs(fes, "left")
    rng = rng_for(fes)
    # independent cross-check of the trace-DoF detection: the surface load int_Gamma phi_i dS is positive exactly
    # on the DoFs whose basis function does not vanish on Gamma (phi_i >= 0 for Lagrange), and by partition of
    # unity its total is the (d-1)-measure of the named patch - here one face of the unit cube.
    lform = ngs.LinearForm(fes.TestFunction() * ngs.ds("left"))
    with ngs.TaskManager():
        lform.Assemble()
    load = lform.vec.FV().NumPy()
    assert np.abs(load.sum() - 1.0) < 1e-12, load.sum()
    assert np.all(load[d] > 0)
    assert np.all(load[np.setdiff1d(np.arange(fes.ndof), d)] == 0.0)
    Sbb = boundary_covariance(rng.gaussian(), "left", fes=fes)
    assert np.linalg.eigvalsh(Sbb).min() > 0
    vals = rng.normal(size=400, bnd="left")
    assert np.all(vals[:, np.setdiff1d(np.arange(fes.ndof), d)] == 0)
    emp = np.cov(vals[:, d].T, bias=False)
    assert np.abs(emp - Sbb).max() < 8 * np.abs(Sbb).max() / np.sqrt(400)


def test_info_reports_boundary(mesh):
    fes = h1(mesh, 2)
    rng = rng_for(fes)
    info = rng.info("normal", bnd="left")
    assert info["bnd"]["mode"] == "restrict"
    assert info["bnd"]["n_bnd"] == 9
    assert list(info["bnd"]["dofs"]) == list(boundary_dofs(fes, "left"))


def test_torch_engine_matches(mesh):
    """The engine must not change the law: the boundary block is the same object in both engines."""
    import torch

    fes = h1(mesh, 1)

    cpu = ngsrng.default_rng(fes, seed=7, correlation=0.25, dtype=torch.float64)
    gpu = ngsrng.default_rng(fes, seed=7, correlation=0.25, dtype=torch.float64, method={"engine": "torch"})
    if gpu.gaussian().engine != "torch":
        pytest.skip("torch engine unavailable")
    S1 = boundary_covariance(cpu.gaussian(), "left", fes=fes)
    S2 = boundary_covariance(gpu.gaussian(), "left", fes=fes)
    # The two engines factor C differently (a sparse Cholesky path against the torch one), so this compares the
    # law to solver accuracy, not bit for bit. Measured here: 2e-9 relative.
    assert np.abs(S1 - S2).max() < 1e-6 * np.abs(S1).max()
