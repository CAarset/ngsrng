# ngsrng

Function-space random variables in NGSolve finite element spaces.

ngsrng draws random *functions* (Gaussian, log-normal, gamma, Student-t, Wishart, ... random fields) in an FE space,
with numpy's sampling syntax. A Gaussian random function $f\sim\mathcal N(m,\mathcal C)$ is not obtained by drawing FE
coefficients as a Gaussian vector: the coefficients of its FE representative have covariance
$\Sigma_h = M^{-1}K_{\mathcal C}M^{-1}$, and sampling it needs FE white noise $z\sim\mathcal N(0,M)$ and the right
operator layer. Every other family is built on such a Gaussian random function. For every family ngsrng implements
and benchmarks several algorithms and picks one automatically.

**Tutorial:** [examples/demo.ipynb](examples/demo.ipynb) walks through the features with pictures: Gaussian fields,
the numpy-style distributions, representations, a PDE with a random coefficient, a Bayesian inverse problem,
low-rank models, covariances given as operators (variance maps, traces, sample covariances), and 1D / curved / DG /
3D settings. FE functions are shown with NGSolve's `Draw` (interactive);
side-by-side panels use the local `ngsplot` package, which paints the same pictures as `Draw` with matplotlib.

**Advanced walkthrough:** [examples/stokes_inverse.ipynb](examples/stokes_inverse.ipynb) solves a 2D Bayesian linear
inverse problem end to end: buoyancy inferred from noisy surface velocities of a Stokes flow, with a Matérn prior whose
Robin boundary condition keeps the variance constant up to the boundary (Daon & Stadler 2018; Villa &
O'Leary-Roseberry 2024), an adjoint-built observation operator, exact posterior variance maps, a calibration check,
posterior predictive checks, the informed directions for several noise levels, low-rank updates, a comparison of
measurement designs, and an independent cross-check. The noise level is a parameter at the top.

**Advanced walkthrough II:** [examples/hydraulic_tomography_3d.ipynb](examples/hydraulic_tomography_3d.ipynb) is a
nonlinear 3D problem with heavy-tailed noise: the log-conductivity of an aquifer block from heads measured in nine
wells during four pumping tests, with Student-t measurement errors. It covers a Robin-tuned Matérn prior in 3D, an
adjoint Jacobian, the MAP point by Gauss–Newton (every step an exact conditioning, with re-weighting for the outliers),
the Laplace approximation with its exact variance map and low-rank structure, and MCMC checks of it (pCN, gpCN with
Laplace proposals, and a Gibbs sampler for the noise scales). It runs in about four minutes.

```python
from ngsolve import *
import numpy as np
import ngsrng

mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
X = H1(mesh, order=3, dirichlet="left")

rng = ngsrng.default_rng(X, seed=0, correlation=ngsrng.matern_correlation(0.2, nu=1))  # device="cuda" too
u = rng.normal()                        # GridFunction
U = rng.normal(1.0, 0.5, size=100)      # (100, ndof) coefficients: NumPy on the CPU, torch on a GPU
k = rng.lognormal(0.0, 0.5, size=10)    # (10, npoints) values at quadrature points
g = rng.sample("gamma", 2.0, size=10)   # same as rng.gamma(2.0, size=10); aliases "gaussian", "chi2", "t", ...
W = rng.wishart(5, S, size=4)           # (4, npoints, p, p) SPD-matrix fields
f = rng.uniform(representation="bernstein")  # an FE function with 0 < f(x) < 1 at every point
rng.info("beta", 2, 5)                  # the law and the algorithms chosen

buf = np.empty((5000, X.ndof), np.float32)   # or a torch tensor, also on the GPU
for _ in range(10):
    rng.normal(out=buf)                  # large batches, filled in place (also lognormal, uniform, ...)
```

### Distributions and the Generator
Method names and parameters follow `numpy.random.Generator`: `normal`, `standard_normal`, `uniform`, `lognormal`,
`exponential`, `gamma`, `beta`, `chisquare`, `standard_t`, `f`, `wald`, `weibull`, `laplace`, `gumbel`, `logistic`,
`rayleigh`, `pareto`, `power`, `triangular`, `vonmises`, `standard_cauchy`, `poisson`, `binomial`,
`negative_binomial`, `geometric`, `hypergeometric`, `logseries`, `zipf`, `dirichlet`, plus `bernoulli`,
`halfnormal`, `wishart`, `inverse_wishart`, `matrix_lognormal`, `normal_inverse_gaussian`, `variance_gamma`,
`indicator`, `white_noise`, `marginal(scipy_dist)` and `transformed(func)`. `sample(name, ...)` accepts numpy,
scipy and torch.distributions names; unknown names fall back to scipy.stats distributions.

- `correlation` (per generator or per call) is the spatial structure: a correlation length, `matern_correlation`
  (unit variance), any precision specification below, an `OperatorCovariance`, or any Gaussian object, e.g. a
  posterior: `rng = ngsrng.default_rng(X, correlation=post)`.
- `size=None` gives one sample (a GridFunction for scalar fields), otherwise arrays of shape
  `(*size, nvalues, *event_shape)`. Gaussian-type fields (normal, Student-t process, NIG / variance-gamma, white
  noise) are FE coefficient vectors. All other fields are values at points: by default the quadrature points of the
  mesh (usable directly as coefficients in forms; integrate with the same `order=`), or any coordinates you pass
  with `points=`. For an FE function use `representation="bernstein"`: it stays inside the distribution's range at
  every point (see limitation 3). `rng.points()` gives the quadrature point coordinates,
  `rng.as_gridfunction(values)` a GridFunction.
- So the last axis has different lengths: `ndof` for coefficients (Gaussian-type fields, `representation="bernstein"`
  or `"projection"`), the number of quadrature points for the default representation (several times `ndof`: about
  10x for P2 triangles at the default `order`), and the number of `points=` otherwise. `as_gridfunction` of
  quadrature-point values is a GridFunction on an IntegrationRuleSpace: a coefficient for forms integrated with
  `rng.dx()`, not an FE function (evaluating or drawing it elsewhere crashes NGSolve,
  known issue 11).
- Output arrays are float32 by default. Every engine, the GPU included, computes in float64 and casts at the end:
  float32 solves lost the samples on fine meshes (pathwise errors up to O(1) in the GPU benchmark), so
  `method={"work_dtype": "float32"}` is an explicit opt-in. Pass `dtype="float64"` for double-precision output.
- Parameters may vary in space (CoefficientFunctions, GridFunctions or arrays of point values).
- `construction=` picks between genuinely different fields with the same single-point distribution (e.g.
  `chisquare(df, construction="squares" | "copula")`, `wishart(..., construction="process" | "bartlett")`,
  `standard_t(..., construction="process" | "copula")`). Within one construction the fastest algorithm is chosen
  automatically (`method=` overrides) and never changes the result's distribution. `rng.info(name, ...)` says what
  was chosen.
- `rng.distribution(name, ...)` freezes a family (set-up done once); `rng.spawn(n)` gives independent generators.

#### What "uniform" (or "gamma", ...) means here
There is no such thing as a uniformly random function. What ngsrng gives you is a random field whose value **at each
single point** follows the named distribution: `rng.uniform()` is a field whose value at any one point is uniformly
distributed between 0 and 1, and whose values at nearby points move together according to the correlation you
chose. Every field is built by transforming one or more Gaussian fields. In practice this means accepting the
following:

1. **Only values at single points follow the named distribution.** Averages, integrals and other combinations of
   values do not. The average of a uniform field over the domain is not uniform: with a short correlation length it
   is close to 0.5, with a long one it spreads out towards a uniform value.
2. **The correlation you choose is that of the hidden Gaussian field.** The field you get is somewhat less
   correlated: for a uniform field a correlation of 0.5 becomes 0.48, and the change can be larger for strongly
   skewed distributions such as the lognormal.
3. **Values follow the named distribution exactly only where they are stored.** By default that is at the
   quadrature points, which is exactly where NGSolve evaluates coefficients when it assembles a form; with `points=`
   it is wherever you ask. If you need an FE function, use `representation="bernstein"`. Its values stay inside the
   distribution's range at **every** point, for any mesh, element order, dimension or curved geometry: a uniform
   field is strictly between 0 and 1 everywhere, a gamma field positive everywhere, a Dirichlet field on the
   simplex, a Wishart field positive definite. At the mesh vertices its values follow the named distribution
   exactly. Between vertices they are weighted averages of nearby values, so they are a little less spread out (in
   our test the variance between vertices was 11% too low with a mesh size one third of the correlation length);
   this disappears as the mesh is refined. It needs meshes of one element type (triangles, tetrahedra,
   quadrilaterals, hexahedra or segments) and spaces of uniform order. `representation="projection"` (plain L2
   projection) has **no** such guarantee: a projected uniform field went down to -0.06 and up to 1.06 on a
   first-order H1 mesh in our tests, and to -0.23 and 1.17 on a first-order discontinuous (L2) mesh.
4. **The correlation must be smooth enough for point values to exist.** For Matérn fields that means smoothness
   ν > 0. Rougher fields would collapse to a constant as the mesh is refined, so ngsrng refuses them for these
   families. For other kinds of covariance it cannot check this; you have to make sure yourself.
5. **Exact values at each point need the field's pointwise standard deviation.** It is computed exactly for up to
   4000 unknowns, and at any size for low-rank models (KL expansions, sample covariances, covariances given as
   operators). Above that it is estimated: the dominant part exactly, the rest from 64 samples. ngsrng prints the
   estimated error; it is typically below 0.5% for smooth fields and 1–2% for rough ones (short correlation length).
   `std="exact"` removes it, at the cost of one solve per unknown.
6. **Posteriors.** The transformation is defined by the prior, so `rng.uniform(correlation=posterior)` is the
   posterior of the prior's uniform field, given data that observe the hidden Gaussian field through linear
   measurements. Data that observe the transformed field itself (say, measured uniform values) are not covered;
   that needs MCMC. Fields built from several copies (sum-of-squares chi-square, Wishart, matrix lognormal,
   Dirichlet) and the Student-t process are meant for priors (ngsrng warns otherwise).
7. **The default Student-t field multiplies a whole Gaussian sample by one random factor.** Its heavy tails show up
   across samples; each individual sample looks Gaussian. For spiky individual samples use the t copula
   (`construction="copula"`) or the NIG / variance-gamma fields.
8. **Discrete fields and indicators** (Poisson, Bernoulli, excursion sets, ...) are piecewise constant with jumps.
9. **A Wishart "process" with few degrees of freedom can be singular on small parts of the domain** (when df is at
   most p + d − 1). This never happens exactly at a stored point, but its inverse is very large near such places.
   The Bartlett construction is invertible everywhere.
10. **Gaussian fields themselves are exact** for the chosen mesh and element order; what remains is the usual
    finite-element discretisation error (plus 1e-7 for fractional powers and the solver tolerance when an iterative
    solver is used). Covariances given only as an operator (`OperatorCovariance`, an NGSolve operator, a function)
    are approximated by dropping the finest scales, so those samples are smoother than the true field. An NGSolve
    `BilinearForm` is never a valid covariance by itself (it describes a local operator, which gives white-noise-like
    "fields"); ngsrng warns and suggests using it as a precision instead.
11. Values of bounded distributions always stay strictly inside their range, also in single precision (a uniform
    value is never exactly 0 or 1).
12. On a GPU (A100), cuDSS samples 25-80x faster than the threaded CPU engine; `python bench/calibrate.py`
    measures it on your machine and installs the fastest accurate choices.
13. **Large batches:** on the CPU, samples are computed in parallel threads (one sample per thread; set the number
    with `method={"threads": k}`). Results do not depend on the number of threads. A batch and the same samples drawn
    one at a time agree to rounding (about 1e-14 relative), not bit for bit.
14. **In-place sampling:** every sampler has an in-place form next to the allocating one, as torch has `normal_`
    next to `normal`: `rng.<family>(..., out=buf)`, `rng.distribution(...).sample(out=buf)`, and
    `Gaussian.sample(n, out=buf)` / `LowRankGaussian.sample(n, out=buf)` / `MatheronPosterior.sample(n, out=buf)` for
    an (ndof x n) buffer. Refilling one buffer takes no memory proportional to the batch: the work runs in chunks of
    `work_values` values (`default_rng(..., work_values=...)`; 2^20 on the CPU, 2^24 on a GPU), and on the CPU each
    sample of a Gaussian field is written into its row by its worker thread. Refilling a 57 MB float32 buffer on a
    2D P1 mesh added 0, 31, 67 and 0 MB at peak for normal(loc, scale), uniform, gamma and chi-square (114, 471,
    1112 and 407 MB before this was introduced). On the CPU the result does not depend on the chunk size or the
    thread count (to rounding for low-rank models and FE coefficients).

#### Distributions, algorithms, accuracy and speed
Speed is the cost of one sample relative to one sample of the underlying Gaussian field. It comes from a preliminary
CPU smoke run (2D/3D, 10^4–10^5 points, on a shared machine), so treat it as indicative until the full benchmark.
Pointwise families also have a one-off set-up per correlation (mostly the exact pointwise standard deviation: seconds
to tens of seconds), which is reused afterwards.

| Distribution | Algorithms (chosen automatically) | Accuracy | Speed | Notes |
|---|---|---|---|---|
| Gaussian (`normal`), integer Matérn power | white noise plus a sparse direct solve (CPU: Cholesky / UMFPACK / PARDISO; GPU: cuDSS) or preconditioned CG (BDDC, AMG, multigrid; GPU CG) | exact with direct solvers; 1e-8 with iterative ones | reference (power 2: about one sparse solve per sample) | which solver wins depends on dimension and size; the full benchmark sets the switch points |
| Gaussian, fractional power | rational approximation | relative 1e-7 | 7–25 solves per sample (up to ~50 for extremely wide spectra) | |
| Gaussian, precision given only as an operator | multi-shift CG | 1e-7 | grows with the square root of the operator's condition number | impractical for very ill-conditioned operators |
| Gaussian, covariance given only as an operator (`OperatorCovariance`, NGSolve operator, function) | adaptive truncated expansion | missing variance at most `tol` (default 1%) | cheap after a one-off set-up | drops the finest scales |
| any covariance, non-Gaussian (`karhunen_loeve(coefficients=...)`) | Karhunen–Loève expansion with independent non-Gaussian coefficients (uniform, Rademacher, Laplace, any scipy law) | the covariance is exactly the given one (low-rank models) or that of the truncation to `rank` modes | one product with the modes, after a one-off expansion | the covariance belongs to the field itself (unlike the transformed fields below); bounded coefficients give bounded fields; values at a point are not of a named distribution |
| Gaussian posterior | low-rank update (FE or noise space), randomised low rank, pathwise (Matheron) | exact (randomised: approximate) | about one prior sample | |
| white noise (`white_noise`) | local noise plus a mass solve | exact | about one Gaussian sample | not a function: values grow under refinement |
| uniform, lognormal, exponential, Weibull, Laplace, Gumbel, logistic, Rayleigh, Pareto, power, Cauchy, triangular, half-normal | closed formula | exact at every point (see limitation 5) | 1–2× | parameters may vary in space; runs on GPU; FE function inside the range everywhere: `representation="bernstein"` |
| gamma, beta, F, Wald, von Mises, chi-square / t copulas, any scipy distribution | lookup table (checked to 1e-10) for large draws and on GPU; exact quantile otherwise | 1e-10 or exact (see limitation 5) | table 1.3–7×; exact 3–18× | the table needs shape parameters that do not vary in space |
| Poisson, binomial, Bernoulli, negative binomial, geometric, log-series, Zipf, hypergeometric | exact lookup table; scipy otherwise | exact | table 1.2–2.7×; scipy 13× or more | integer values |
| chi-square | sum of `df` squared fields (default for integer `df`) or one transformed field (copula) | exact at every point | sum: 1.4–2.5× for df=3, growing with df; copula 1.5–3× | the two have different spatial behaviour |
| Student-t | one random factor per sample (default) or transformed field (copula) | exact | 2–8×; copula 0.6–4× | see limitation 7 |
| Wishart, inverse Wishart | process (`df`·p fields, default) or Bartlett (p(p+1)/2 fields, any real `df`) | exact at every point | p=2, df=5: Bartlett 3–7.5×, process 5–14×; the process gets slower with `df`, Bartlett does not | see limitation 9 |
| matrix lognormal | exact formula for 2×2; eigendecomposition for larger matrices | exact | 2×2: 2.5–10×; 3×3: about 10× | always symmetric positive definite |
| Dirichlet | normalised gamma fields with lookup tables | 1e-10 | 1.5–7× | parameters below ~0.02 are not supported |
| NIG, variance-gamma | the Gaussian pipeline with randomly rescaled white noise on each element | exact (for the element-scale noise model) | 2–4× | priors only; same covariance as the Gaussian, heavier tails |
| indicator, `transformed(func)`, `marginal(scipy_dist)` | as above | as above | as above | |
| any of the point-value families above as an FE function (`representation="bernstein"`) | Bernstein–Bézier form: the FE function is a weighted average of values at the element nodes, with nonnegative weights | inside the range at every point (guaranteed); exact at mesh vertices; slightly smoothed in between, vanishing under refinement | 1.1–6× a Gaussian sample (quadrature values: 1.1–2.3×) | one element type per mesh, uniform order; coefficients in float64 |

The Gaussians behind the generator can also be used directly:

```python
prior = ngsrng.Gaussian(X, precision=ngsrng.matern(kappa=10, power=2), seed=0)  # device="cuda" or "auto"
print(prior)  # chosen algorithms (plan) and why
f = prior.sample()  # GridFunction
F = prior.sample(100)  # torch tensor (ndof x 100), float32 by default
```
Algorithms can be overridden with `method={...}` (here or in `default_rng`), e.g. `{"engine": "torch", "solver": "cg",
"noise": "quadrature"}`; the valid keys and names are in `ngsrng.gaussian.METHOD_KEYS`, and anything else raises.
The approximate white-noise samplers (`"chebyshev"`, `"lumped"`) are never chosen automatically and warn with their
error when requested.

Precisions:
- `matern(kappa, power)` (H1) or `matern_dg(fes, kappa, power)` (L2 with `dgjumps=True`): $\mathcal C = \mathcal A^{-\text{power}}$.
  `robin=beta` (or `"auto"`: $\beta=\kappa/1.42$) adds a Robin boundary condition. With the default natural
  condition the variance is about twice as large at edges and four times as large in corners;
  `matern_correlation(..., boundary="robin")` keeps it within a few percent of the interior value except near corners
  (up to about 12% lower there), as measured in the Stokes walkthrough.
  `power` is any positive real (Matérn smoothness $\nu=\text{power}-d/2$). Integer powers are exact, except odd powers of forms with facet terms (`matern_dg`), which take the
  rational route below with a relative accuracy of `rtol` (no global factorisation is ever formed). Real powers use
  a rational approximation with relative accuracy `rtol` (default 1e-7) on the discrete spectrum, i.e. 7–25 shifted
  solves $(K+tM)^{-1}$ per sample.
- `PDEPrecision(form, power)`: any bilinear form.
- `AssembledPrecision(form)`: $\Sigma_h^{-1}=Q$, e.g. non-decoupling fourth-order operators. Exact via local Q-noise plus one solve.
- `OperatorPrecision(apply, convention="primal->dual" | "dual->dual")`: a precision known only through its action.
  Sampled as $(M^{-1}Q)^{-1/2}M^{-1}z$ by a rational approximation whose shifted systems $Q+t_lM$ are all solved in one
  multi-shift CG run per sample. The spectral interval comes from inverse iteration and Lanczos, or is user-given.
- `OperatorCovariance(apply, convention, tol)`: a covariance known only through its action: a function on
  coefficient blocks, an NGSolve `BaseMatrix` (any operator expression, e.g. `Ainv @ M.mat @ Ainv`) or a
  `BilinearForm`. `convention`: "dual->primal" ($\Sigma$ itself; default), "primal->primal" ($\Sigma M$) or
  "primal->dual" ($M\Sigma M$, the covariance form $b(u,v)=\mathbb E[(f,u)(f,v)]$; default for a BilinearForm).
  Sampled by an adaptive truncated KL expansion (`operator_kl`): the rank doubles until the uncaptured trace,
  estimated with white-noise Hutchinson probes, is below `tol` times the total. `default_rng(X, covariance=...)` also
  takes the operator, form or function directly.

Boundary conditions belong to the covariance: Dirichlet through the space, Robin through the form, Neumann implicitly.

### Posteriors (linear Gaussian Bayesian inverse problems)
```python
from ngsrng.posterior import LinearObservations

# any finite set of bounded linear observation functionals: local averages, LinearForms (fluxes, ds, derivatives),
# matrices, matrix-free maps via their adjoint (e.g. through a PDE solve), point values
obs = LinearObservations(prior, averages=(centers, 0.05), functionals=[flux_form], noise=1e-4)
post = obs.posterior(data=y)  # exact; a Gaussian with the prior's pipeline + a low-rank update
post.sample(100)  # costs about one prior sample each
# set-up keeps dense nrand x m and nfree x m factors (m observations): it raises MemoryError up front when they do not
# fit, and LinearObservations(..., rank=k) is the randomised low-rank alternative

post = prior.downdate(B, K)  # a known posterior C0 - B K^-1 B*, given directly

# re-weighted noise (precision of observation i times w[i]) without refactorising: an l x l eigendecomposition and
# shared NGSolve data, a few ms per re-weighted posterior. The Gaussian step of iteratively reweighted least squares
# and of Gibbs samplers for Student-t / Laplace noise
post_w = obs.posterior(data=y, noise_scaling=w)
laplace = obs.posterior(mean=m_map, noise_scaling=h)  # the same covariance centred elsewhere, e.g. at a MAP point
```
Low-rank models: `ngsrng.karhunen_loeve(gaussian, rank)` gives the truncated KL expansion of any Gaussian (prior or
posterior, L2-orthonormal modes) as a `LowRankGaussian`, which supports exact conditioning with r x r algebra. It uses
randomised subspace iteration on the Gaussian's factor (Aarset 2024, Algorithm 2; Halko, Martinsson & Tropp 2011):
the sketch is re-orthonormalised after every application, so small eigenvalues stay accurate and fast-decaying
spectra cannot break it (`method="covariance"`, a range finder on the covariance, is kept for comparison). The
randomised posterior update (`LinearObservations(..., rank=k)`) uses the same algorithm on the prior-preconditioned
adjoint of the observation map. All of it is in `ngsrng/randomized.py`.

`prior.condition(...)` is a shortcut for `LinearObservations(prior, ...).posterior(...)`.

### Covariances: variance maps, traces, sample covariances
Every Gaussian (prior, posterior, low-rank model) and every covariance given as an operator can be inspected without
drawing samples. The variance at a point x is the diagonal k(x, x) of the covariance kernel, and on each element it
only needs the covariance between that element's unknowns. ngsrng computes those entries and returns the variance
as an FE function (of twice the order), exact everywhere. The integral of the variance is the trace of the
covariance (the expected squared L2 norm of a sample).

```python
var = ngsrng.pointwise_variance(prior)          # GridFunction: Draw(var), or the std as ngs.sqrt(var)
ngsrng.pointwise_variance(post)                 # posterior: the prior's variance plus an exact low-rank update
ngsrng.covariance_trace(prior)                  # tr C = int k(x, x) dx, with its error and cost
k0 = ngsrng.covariance_function(prior, (0.5, 0.5))  # k(x0, .): covariance of the value at x0 with the field
S = ngsrng.functional_covariance(post, points=pts)  # exact covariance of point values, averages, LinearForms, ...
ens = ngsrng.sample_covariance(prior, 50)       # ensemble model: the sample covariance of 50 samples (Wishart)
C = ngsrng.OperatorCovariance(Ainv @ M.mat @ Ainv)  # any NGSolve operator expression as a covariance
rng = ngsrng.default_rng(X, covariance=C)       # ... and all distributions with it
rng.karhunen_loeve("uniform", rank=100)         # a non-Gaussian field whose covariance is exactly that of the KL
```

| Quantity | Method (`method=`) | Accuracy | Cost (applications of the covariance or its factor) |
|---|---|---|---|
| variance map | `"exact"`: from all columns | exact | one per unknown (default up to 4000 unknowns) |
| variance map | low-rank models, and posteriors of any prior | exact | free (low rank); posterior: the prior's (once, cached) plus about twice the number of observations |
| variance map | `"lowrank"`: dominant part exact, rest sampled (default above 4000 unknowns) | error printed; 2D tests: 0.2% (smooth) to 3% (rough) with 1400 applications | rank × (2 × iterations + 3) + samples |
| variance map | `"mc"`: sample variance | about sqrt(2 / samples): 7% with 400 samples | samples |
| trace | `"exact"` or `"variance"` (integral of the variance map) | as the variance map | as the variance map |
| trace | `"xnystrace"` (default when no exact method is cheap) | best of the three in our tests: often 10–1000× better than Hutchinson at equal cost | probes (× 2 with a factor) |
| trace | `"hutch++"` | good for fast-decaying spectra | about 5/3 × probes |
| trace | `"hutchinson"` | sqrt(2 tr(C²) / probes) relative to tr(C): fine for rough fields, poor for smooth ones | probes |

What to keep in mind:
- The variance maps and traces are those of the **finite element** field: exact for the chosen mesh and order,
  including the higher variance near Neumann boundaries.
- `"lowrank"` pays off for smooth fields (long correlation lengths, fast eigenvalue decay). For rough fields it is
  about as good as plain sampling at the same cost, and `"exact"` may be the better choice.
- A **sample covariance** with n samples is unbiased, but its error decreases only like 1/sqrt(n), and with few
  samples it shows spurious correlations between distant points. It needs more samples than the number of
  independent features of the field (about the domain size divided by the correlation length, to the power of the
  dimension).
- A **Karhunen–Loève field** with uniform coefficients has exactly the requested covariance and bounded values, but
  its value at a point is not uniform (a sum of many independent terms is close to Gaussian). For a field that is
  uniform at every point, use `rng.uniform()`; its covariance is then that of the hidden Gaussian field, not the
  given one.

## Repository layout
| Path | Contents |
|---|---|
| `ngsrng/` | The package. `gaussian.py` (sampler and automatic selection), `noise.py` (white noise), `solvers.py`, `rational.py` (fractional powers), `observations.py`, `posterior.py`, `lowrank.py`, `randomized.py` (randomised subspace iteration), `covariance.py` (variance maps, traces, covariance functions, sample covariances), `quadrature.py`, `generator.py` (numpy-style API), `correlation.py`, `distributions/` (inversion, copula marginals, quadratic forms, scale mixtures, KL expansions), `tuning.py`, `reference.py` (dense ground truth), `space.py`, `backend.py`, `specs.py` |
| `tests/` | One test file per module; `python -m pytest` |
| `bench/` | Calibration: `calibrate.py` (one command), `run_all.py` (orchestrator), `cases.py` (candidates), `worker.py`, `report.py`; optional `boundary_scaling.py`, `gpu_batch.py` |
| `examples/` | `demo.ipynb`: the user tutorial; `stokes_inverse.ipynb`: advanced walkthrough of a linear Bayesian inverse problem with a Stokes forward model; `hydraulic_tomography_3d.ipynb`: nonlinear 3D problem with heavy-tailed noise, Laplace approximation and MCMC. All executed (interactive `Draw` views and static panels included) |
| `results/` | `calibrate/` and `runs/` (generated, git-ignored), `gpu_box/` (the 2026-11-02 A100 run), `prototype/` (archived early results) |

## Installation
```bash
uv pip install -e ".[test,bench]"     # core + tests + benchmark tooling
uv pip install -e ".[gpu]"            # optional: cuDSS GPU direct solver (CUDA 13)
uv pip install -e ".[examples]"       # optional: Jupyter, anywidget (for Draw) and matplotlib, for the tutorial
uv pip install -e /path/to/ngsplot     # optional: local package for the tutorial's side-by-side panels
```

## Calibration (one command)
```bash
python bench/calibrate.py                  # tests, validation, calibration, boundary and GPU checks (~30 min)
python bench/calibrate.py --skip boundary,throughput   # tests, validation and calibration only (~20 min)
python bench/calibrate.py --quick          # a few minutes: harness check (installs no tuning)
python bench/calibrate.py --large          # calibrate up to 1e6 dofs (an hour or more)
python bench/calibrate.py --resume results/calibrate/<run>   # continue an interrupted run
python bench/run_all.py --install          # the calibration step alone
python bench/run_all.py --profile research # posteriors, distributions, low rank, ...: research data, hours
```
- **Purpose:** choose this machine's white-noise sampler and linear solver, and reject the candidates that cannot win.
  The calibration measures only exact samplers (diagonal, local Gram, quadrature) and the solvers that can win (NGSolve
  sparse Cholesky and UMFPACK, CG with BDDC or h1amg, cuDSS and batched PCG on a GPU), in 2D and 3D at 1e4 and 1e5
  dofs plus 3D P1 at 2.5e5 (1e6 with `--large`), on every available device. One recommendation serves single draws
  and batches (one sample per CPU thread, 256 on a GPU), so candidates are ranked by their worst ratio to the best
  time in either mode, and near-ties go to exact solvers. On the A100 box (2026-11-03) cuDSS was 16-94x faster than
  the best CPU solver at 1e5 dofs.
- **Accuracy first:** every measurement is checked, also at large sizes: white noise must reproduce the mass matrix
  to 1e-10, priors must match a float64 direct reference sample for sample to 1e-6. A candidate is rejected for
  larger sizes once it is inaccurate, fails, exceeds its time budget, or is 20x slower than the best accurate one on
  its platform (iterative solvers in 3D excepted). The earlier exhaustive benchmark took days and ranked by speed
  alone; the measured losers -- Chebyshev and lumped noise, Jacobi- and multigrid-preconditioned CG, PARDISO,
  float32 GPU arithmetic -- are no longer candidates ([bench/cases.py](bench/cases.py)).
- **Portability:** available devices and libraries are detected. Crashes, hangs and out-of-memory cases run in
  isolated worker processes and are recorded as results. A memory watchdog keeps workers below 80% of available RAM.
- **Output:** `results/calibrate/<host>_<date>/` holds `SUMMARY.md`, the step logs and, under `out/calibrate/`, the
  raw JSON lines, `report.md` (accuracy, fastest accurate method per case, rejected candidates), `plots/`,
  `recommendations.json` and `rejected.json`.
- **Tuning:** the calibration installs its recommendations (`python -m ngsrng.tuning install <run dir>` does it by
  hand): `Gaussian` then uses the measured fastest accurate noise sampler, solver and, with `device="auto"`, device
  for the nearest calibrated case. `python -m ngsrng.tuning show` lists the active tuning. Without tuning, built-in
  rules apply.

## Development
```bash
python -m pytest                  # GPU tests are skipped automatically without a usable CUDA device
ruff check . && ruff format .     # lint and formatting (configured in pyproject.toml)
```

**Every sampler must have an in-place form** ([ngsrng/inplace.py](ngsrng/inplace.py)): inherit `InPlace` (chunked
`sample` calls) or implement a direct `sample_into(rows)`, draw randomness sample by sample, and add new families to
`ARGS` in `tests/test_inplace.py`. That test fails for any registered family whose `out=` fill differs from the
allocating call, and for any class with a `sample` method but no in-place form.
