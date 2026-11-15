"""Statistical assertions with a stated false-failure rate: the test suite's policy for randomised code.

A test must never fail by chance. ngsrng's tests follow these rules:

1. Every draw is seeded, so a test's outcome is a fixed function of the code and the platform.
2. A law is checked exactly where possible: the same noise through two routes (common random numbers), exact moments
   and covariances of affine maps, maps evaluated on deterministic inputs.
3. A remaining statistical check is a statement about i.i.d. values with a known law, and its threshold comes from a
   bound that holds for every sample size, at the false-failure rate ``ALPHA``: a correct implementation fails it
   with probability at most ``ALPHA`` whatever the seed, so a new realisation (another platform, whose mesher or
   libraries round differently; a library update that changes a stream) cannot flip it in practice. The bounds are
   distribution-free (Dvoretzky-Kiefer-Wolfowitz-Massart, Hoeffding) or exact (Gaussian means, chi-square laws,
   Laurent-Massart for Gaussian quadratic forms, Davidson-Szarek for Gaussian matrices). Every assertion reports
   the smallest deviation it detects, so its power is visible.
4. Values that are not i.i.d. with a known law are first made so: whitening a Gaussian vector with its exact
   covariance, transforming copula values back to their latent Gaussians, standardising by an exact variance, the
   probability integral transform.
5. An estimator without a tractable law is checked deterministically: fixed seeds on a structured mesh (exact
   coordinates and topology, unlike netgen's mesher, which turns a one-ulp change into another mesh), so its
   realisation, hence the outcome, is the same on every machine.

Values are never pooled across points of one draw unless whitened: they are correlated, not i.i.d.
"""

import math

import numpy as np
import scipy.stats as st

ALPHA = 1e-12  # false-failure rate of every statistical assertion
MARGINS = None  # a list to record (assertion, statistic / threshold) in, e.g. to check that a law model is right


def _note(what, kind, ratio):
    if MARGINS is not None:
        MARGINS.append((what or kind, kind, float(ratio)))


def _flat(x):
    return np.asarray(x, dtype=np.float64).ravel()


def _fail(what, message):
    raise AssertionError(f"{what + ': ' if what else ''}{message}")


# ------------------------------------------------------------------ laws of i.i.d. values


def ks_distance(x, cdf, left_cdf=None):
    """sup_t |F_n(t) - F(t)| of the empirical CDF of x against ``cdf`` (exact also for discrete laws: ``left_cdf`` is
    F(t-), the CDF just below t; None for continuous laws)."""
    x = np.sort(_flat(x))
    n = x.size
    f = cdf(x)
    fl = f if left_cdf is None else left_cdf(x)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - f), np.max(fl - (i - 1) / n)))


def dkw_threshold(n, alpha=ALPHA):
    """epsilon with P(sup |F_n - F| > epsilon) <= alpha for n i.i.d. values (Dvoretzky-Kiefer-Wolfowitz with
    Massart's constant: 2 exp(-2 n epsilon^2))."""
    return math.sqrt(math.log(2 / alpha) / (2 * n))


def assert_law(x, cdf, *, left_cdf=None, alpha=ALPHA, what=""):
    """The values x (i.i.d.) have the CDF ``cdf`` (``left_cdf``: F(t-) for discrete laws). Fails a correct sampler
    with probability at most ``alpha``; detects CDF errors larger than twice the threshold."""
    x = _flat(x)
    d, eps = ks_distance(x, cdf, left_cdf), dkw_threshold(x.size, alpha)
    _note(what, "law", d / eps)
    if d > eps:
        _fail(what, f"empirical CDF off by {d:.4g} > {eps:.4g} (DKW-Massart, n = {x.size}, alpha = {alpha:g})")


def assert_same_law(x, y, *, alpha=ALPHA, what=""):
    """Two i.i.d. samples x, y have the same law: sup |F_x - F_y| <= eps_x + eps_y (each DKW-Massart at alpha / 2,
    by the triangle inequality through the common CDF)."""
    x, y = np.sort(_flat(x)), np.sort(_flat(y))
    grid = np.concatenate([x, y])
    d = float(np.max(np.abs(np.searchsorted(x, grid, side="right") / x.size
                            - np.searchsorted(y, grid, side="right") / y.size)))
    eps = dkw_threshold(x.size, alpha / 2) + dkw_threshold(y.size, alpha / 2)
    _note(what, "same law", d / eps)
    if d > eps:
        _fail(what, f"empirical CDFs differ by {d:.4g} > {eps:.4g} (two DKW-Massart bands, n = {x.size}, "
                    f"{y.size}, alpha = {alpha:g})")


def assert_uniform_pit(x, cdf, *, alpha=ALPHA, what=""):
    """Independent values x with (possibly different, continuous) CDFs ``cdf(x)`` evaluated elementwise: the
    probability integral transform u = F(x) is i.i.d. uniform, checked by :func:`assert_law`."""
    assert_law(cdf(x), st.uniform().cdf, alpha=alpha, what=what)


def assert_normal(x, mean, sd, *, alpha=ALPHA, what=""):
    """The values x are i.i.d. N(mean, sd^2): their law (DKW-Massart), their mean (exactly normal) and their spread
    (sum of squares exactly chi-square), each at alpha / 3."""
    u = (_flat(x) - mean) / sd
    n = u.size
    assert_law(u, st.norm.cdf, alpha=alpha / 3, what=what)
    z = st.norm.isf(alpha / 6)
    _note(what, "normal mean", abs(u.mean()) / (z / math.sqrt(n)))
    if abs(u.mean()) > z / math.sqrt(n):
        _fail(what, f"mean off by {u.mean():.4g} standard deviations > {z / math.sqrt(n):.4g} (n = {n}, "
                    f"alpha = {alpha:g})")
    q = float(np.sum(u * u))
    lo, hi = st.chi2.ppf(alpha / 6, n), st.chi2.isf(alpha / 6, n)
    _note(what, "normal variance", max((n - q) / (n - lo), (q - n) / (hi - n)))
    if not lo <= q <= hi:
        _fail(what, f"variance ratio {q / n:.4g} outside [{lo / n:.4g}, {hi / n:.4g}] (n = {n}, alpha = {alpha:g})")


def assert_bounded_mean(x, mean, lo, hi, *, alpha=ALPHA, what=""):
    """The i.i.d. values x in [lo, hi] have the mean ``mean`` (Hoeffding)."""
    x = _flat(x)
    t = (hi - lo) * math.sqrt(math.log(2 / alpha) / (2 * x.size))
    _note(what, "bounded mean", abs(x.mean() - mean) / t)
    if abs(x.mean() - mean) > t:
        _fail(what, f"mean {x.mean():.6g}, expected {mean:.6g} +- {t:.3g} (Hoeffding, n = {x.size}, "
                    f"alpha = {alpha:g})")


def assert_gaussian(y, mean, cov, *, alpha=ALPHA, rank_rtol=1e-10, what=""):
    """The rows of y (n x m) are i.i.d. N(mean, cov). Whitened with the exact covariance (its eigenvectors with
    eigenvalues above ``rank_rtol`` times the largest), the n x r values must be i.i.d. N(0, 1): their pooled law
    (DKW-Massart), every coordinate's mean (exactly normal), and every direction's spread at once (Davidson-Szarek:
    the singular values of an n x r standard Gaussian matrix lie in sqrt(n) +- (sqrt(r) + t) with probability at least
    1 - 2 exp(-t^2 / 2)), each at alpha / 3."""
    y = np.asarray(y, dtype=np.float64)
    lam, v = np.linalg.eigh(0.5 * (np.asarray(cov) + np.asarray(cov).T))
    keep = lam > rank_rtol * lam.max()
    u = (y - np.asarray(mean)) @ (v[:, keep] / np.sqrt(lam[keep]))
    n, r = u.shape
    assert_law(u, st.norm.cdf, alpha=alpha / 3, what=what)
    z = st.norm.isf(alpha / (6 * r))
    worst = float(np.abs(u.mean(axis=0)).max())
    _note(what, "gaussian mean", worst / (z / math.sqrt(n)))
    if worst > z / math.sqrt(n):
        _fail(what, f"a whitened mean off by {worst:.4g} > {z / math.sqrt(n):.4g} (n = {n}, alpha = {alpha:g})")
    t = math.sqrt(2 * math.log(2 / (alpha / 3)))
    s = np.linalg.svd(u, compute_uv=False)
    lo, hi = math.sqrt(n) - math.sqrt(r) - t, math.sqrt(n) + math.sqrt(r) + t
    _note(what, "gaussian covariance", max((math.sqrt(n) - s.min()) / (math.sqrt(n) - lo),
                                           (s.max() - math.sqrt(n)) / (hi - math.sqrt(n))))
    if s.min() < lo or s.max() > hi:
        _fail(what, f"whitened covariance eigenvalues in [{s.min()**2 / n:.4g}, {s.max()**2 / n:.4g}], allowed "
                    f"[{lo**2 / n:.4g}, {hi**2 / n:.4g}] (Davidson-Szarek, n = {n}, r = {r}, alpha = {alpha:g})")


# ------------------------------------------------------------------ Gaussian quadratic forms


def assert_scaled_chi2(v, scale, df, *, alpha=ALPHA, what=""):
    """Every entry of v is scale * chi^2_df / df (e.g. a Monte Carlo variance from df Gaussian samples, scale the exact
    variance): df v / scale lies within the chi-square quantiles at alpha / (2 entries) each (union bound)."""
    v, scale = _flat(v), np.broadcast_to(_flat(scale), _flat(v).shape)
    a = alpha / (2 * v.size)
    lo, hi = st.chi2.ppf(a, df), st.chi2.isf(a, df)
    q = df * v / scale
    _note(what, "scaled chi2", float(np.max(np.maximum((df - q) / (df - lo), (q - df) / (hi - df)))))
    bad = (q < lo) | (q > hi)
    if bad.any():
        k = int(np.argmax(np.abs(np.log(q / df))))
        _fail(what, f"{int(bad.sum())} of {v.size} values outside the chi-square band: ratio {q[k] / df:.4g}, allowed "
                    f"[{lo / df:.4g}, {hi / df:.4g}] (df = {df}, alpha = {alpha:g})")


def assert_quadratic_form(value, mean, sum_a2, max_a, *, alpha=ALPHA, what=""):
    """``value`` = sum_i a_i Z_i^2 with i.i.d. standard normal Z_i and weights a_i >= 0 (E value = ``mean`` = sum a_i),
    given sum a_i^2 and max a_i, or upper bounds of both (Laurent-Massart: P(X - E X >= 2 sqrt(sum a^2 x) + 2 max(a) x)
    and P(E X - X >= 2 sqrt(sum a^2 x)) are each at most exp(-x))."""
    x = math.log(2 / alpha)
    up, down = 2 * math.sqrt(sum_a2 * x) + 2 * max_a * x, 2 * math.sqrt(sum_a2 * x)
    _note(what, "quadratic form", (value - mean) / up if value >= mean else (mean - value) / down)
    if not -down <= value - mean <= up:
        _fail(what, f"{value:.6g} outside [{mean - down:.6g}, {mean + up:.6g}] around {mean:.6g} (Laurent-Massart, "
                    f"alpha = {alpha:g})")
