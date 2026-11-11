"""Tests for boundary (trace) sampling: ``bnd="name"`` on any ngsrng random function.

Mathematical content of each check:

* the trace of an FE function is a linear map on coefficients (row selection), so restricting a sample to the
  DoFs supported on the boundary is the exact marginal law of those DoFs -- test_restrict_is_marginal;
* for a precision C = A^-1 (power = 1) the boundary block of Sigma_h = K^-1 is the inverse of the assembled
  Dirichlet-to-Neumann (Schur complement) matrix, and the regression coefficient Sigma_IB Sigma_BB^-1 is the
  K-harmonic lifting -- test_dtn_theorem, test_lifting_is_harmonic;
* the boundary marginal of the sampler has the covariance Sigma_BB (Monte Carlo) -- test_boundary_moments;
* bnd_values=g conditions the field on its trace, u* = u - Sigma_B Sigma_BB^-1 (R u - g): g is imposed exactly and
  the interior has the conditional law -- test_conditioning_*;
* non-Gaussian (copula) families act pointwise, so the same restriction is exact for them -- test_copula_marginal.
"""

import ngsolve as ngs
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from conftest import cuda_usable

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
    # with the generator-level default (never asked for), bnd= switches to the Bernstein representation and says so
    info = rng.info("uniform", bnd="left")
    assert info["bnd"]["mode"] == "restrict"
    assert info["representation"] == "bernstein"


@pytest.mark.parametrize("order", [1, 2])
def test_default_boundary_draws_respect_the_range(mesh, order):
    """Without representation=, bnd= draws of non-Gaussian families are Bernstein fields: the boundary data stay in
    the family's range at every point of Gamma (checked at the trace's domain points), for any order."""
    fes = h1(mesh, order)
    rng = rng_for(fes)
    vals = np.asarray(rng.uniform(size=200, bnd="left"), dtype=float)
    # values on Gamma = {x = 0} only: off Gamma the restricted function has zeroed coefficients and no guarantee
    G = ngsrng.observation_operator(fes, points=np.column_stack([np.zeros(41), np.linspace(0, 1, 41)]))
    on_gamma = G @ vals.T
    assert on_gamma.min() >= -1e-6 and on_gamma.max() <= 1 + 1e-6, (on_gamma.min(), on_gamma.max())
    assert rng.info("uniform", representation="projection", bnd="left")["representation"] == "projection"


def test_default_boundary_representation_falls_back_on_variable_order(mesh):
    """Bernstein needs a uniform order; on a variable-order space the bnd= default stays the L2 projection."""
    fes = h1(mesh, 2)
    fes.SetOrder(ngs.NodeId(ngs.EDGE, 0), 3)  # one cubic edge: 86 dofs instead of 85
    fes.Update()
    assert ngsrng.default_rng(fes, seed=1, correlation=0.25).info("uniform", bnd="left")["representation"] \
        == "projection"


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


def test_restrict_keeps_dirichlet_trace_dofs(mesh):
    """Restriction zeroes the DoFs off the boundary only: trace DoFs fixed by dirichlet= keep their values (zeroing
    them would change the boundary values)."""
    fes = ngs.H1(mesh, order=2, dirichlet="bottom")
    make = lambda: ngsrng.default_rng(fes, seed=4, correlation=ngsrng.matern_correlation(0.3, nu=1))  # noqa: E731
    full = np.asarray(make().uniform(size=3, representation="bernstein"))
    restricted = np.asarray(make().uniform(size=3, representation="bernstein", bnd="left"))
    trace = boundary_dofs(fes, "left", free_only=False)
    assert np.setdiff1d(trace, boundary_dofs(fes, "left")).size  # the corner shared with the Dirichlet boundary
    np.testing.assert_allclose(restricted[:, trace], full[:, trace], rtol=0, atol=1e-12)  # two set-ups: rounding
    assert not restricted[:, np.setdiff1d(np.arange(fes.ndof), trace)].any()


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


# ------------------------------------------------------------------ conditioning on the trace


def test_conditioning_prescribes_data(mesh):
    """Given boundary data, bnd_values= is exact hard Dirichlet conditioning."""
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


def test_bnd_mode_is_gone_and_values_need_a_gaussian(mesh):
    """bnd= restricts and bnd_values= conditions; there is no third mode, and a stale bnd_mode= is an error rather
    than a silently ignored keyword. Conditioning is linear, so it needs an affine (Gaussian) field."""
    fes = h1(mesh, 1)
    rng = rng_for(fes)
    with pytest.raises(TypeError, match="bnd_mode"):
        rng.normal(size=2, bnd="left", bnd_mode="field")
    with pytest.raises(NotImplementedError, match="nonlinear"):
        rng.uniform(size=2, bnd="left", bnd_values=np.full(boundary_dofs(fes, "left").size, 0.5))
    n_bnd = boundary_dofs(fes, "left").size
    assert rng.info("normal", bnd="left", bnd_values=np.zeros(n_bnd))["bnd"]["mode"] == "condition"


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
    # compared with the continuum spectrum on refined meshes outside the test suite.


def test_trace_spectrum_is_that_of_the_trace_operator():
    """The eigenvalues are those of Sigma_BB M_Gamma: they sum to E||gamma u||^2 = tr(M_Gamma Sigma_BB), and at
    power = 1 they approximate the Neumann-to-Dirichlet map of the square, mu_0 = 1, mu_k = tanh(k pi) / (k pi)."""
    fes = h1(ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.125)), 2)
    form = lambda uu, vv: ngs.InnerProduct(ngs.grad(uu), ngs.grad(vv)) * ngs.dx
    rng = ngsrng.default_rng(fes, seed=1, precision=ngsrng.PDEPrecision(form, power=1), dtype="float64")
    sam = BoundarySampler(rng.gaussian(), "left")
    w = sam.trace_eigenvalues()
    Mg = ngsrng.boundary.boundary_mass(fes, "left").tocsc()[sam.dofs][:, sam.dofs]
    assert abs(w.sum() - np.trace(Mg @ sam.sigma_bb)) < 1e-12 * w.sum()
    k = np.arange(1, 4)
    mu = np.concatenate([[1.0], np.tanh(k * np.pi) / (k * np.pi)])
    assert np.abs(w[:4] / mu - 1).max() < 1e-2, w[:4] / mu
    top = sam.trace_eigenvalues(n=3)
    assert np.allclose(top, w[:3], rtol=1e-8), (top, w[:3])


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


@pytest.mark.parametrize("power", [1, 2])
def test_correction_paths_agree(mesh, monkeypatch, power):
    """The three ways of applying the lifting P without forming it -- sparse harmonic lifting (power 1), small dense
    lifting, matrix-free Sigma application -- give the same conditioned samples, also when the samples are conditioned
    in blocks of two and when they arrive as torch rows."""
    import torch

    import ngsrng.boundary as bmod

    fes = h1(mesh, 2)
    form = lambda u, v: (ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) + 4.0 * u * v) * ngs.dx
    g = ngsrng.Gaussian(fes, ngsrng.PDEPrecision(form, power=power), dtype="float64", seed=0)
    rows = g.sample(5).T.numpy().copy()
    data = np.linspace(-1.0, 1.0, boundary_dofs(fes, "left").size)
    results = []
    for dense_limit, sparse in ((2**24, True), (2**24, False), (0, False)):
        monkeypatch.setattr(bmod, "DENSE_LIFTING_VALUES", dense_limit)
        sam = BoundarySampler(g, "left")
        if not sparse:
            sam._sparse_form = None
        elif power != 1:
            continue
        results.append(sam.condition(rows.copy(), data))
        monkeypatch.setattr(bmod, "BLOCK_VALUES", 2 * fes.ndof)
        results.append(sam.condition(rows.copy(), data))
        results.append(sam.condition(torch.from_numpy(rows.copy()), data).numpy())
        monkeypatch.setattr(bmod, "BLOCK_VALUES", 2**24)
    assert len(results) >= 6
    for r in results[1:]:
        np.testing.assert_allclose(r, results[0], rtol=1e-8, atol=1e-10 * np.abs(results[0]).max())
    assert np.all(results[0][:, sam.dofs] == data)


@pytest.mark.skipif(not cuda_usable(), reason="needs a CUDA device")
@pytest.mark.parametrize("power", [1, 2])
def test_conditioning_on_the_device(mesh, monkeypatch, power):
    """Torch rows on a GPU are conditioned there -- cuDSS with the interior block for a sparse precision, the source's
    own pipeline otherwise -- and agree with the same rows conditioned on the host."""
    import torch

    import ngsrng.boundary as bmod

    monkeypatch.setattr(bmod, "DENSE_LIFTING_VALUES", 0)  # power 2: the matrix-free path, not the small dense lifting
    fes = h1(mesh, 2)
    form = lambda u, v: (ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) + 4.0 * u * v) * ngs.dx
    g = ngsrng.Gaussian(fes, ngsrng.PDEPrecision(form, power=power), device="cuda", dtype="float64", seed=0)
    rows = g.sample(5).T.contiguous()
    sam = BoundarySampler(g, "left")
    data = np.linspace(-1.0, 1.0, sam.n_bnd)
    on_device = sam.condition(rows.clone(), data)
    assert on_device.device.type == "cuda"
    assert sam._device(rows.device)[0] == ("sparse" if power == 1 else "sigma")
    on_host = sam.condition(rows.cpu().numpy().copy(), data)
    np.testing.assert_allclose(on_device.cpu().numpy(), on_host, rtol=1e-8, atol=1e-10 * np.abs(on_host).max())
    assert np.all(on_device.cpu().numpy()[:, sam.dofs] == data)
    # through the generator, filled in place on the device
    rng = ngsrng.default_rng(fes, seed=1, device="cuda", dtype="float64",
                             precision=ngsrng.PDEPrecision(form, power=power))
    buf = torch.empty((6, fes.ndof), device="cuda", dtype=torch.float64)
    rng.normal(bnd="left", bnd_values=data, out=buf)
    assert torch.equal(buf[:, torch.as_tensor(sam.dofs, device="cuda")].cpu(),
                       torch.from_numpy(data).expand(6, -1))
