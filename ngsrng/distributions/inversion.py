"""Inverse-transform maps z -> F^-1(Phi(z)) from standard normal values to a prescribed marginal F.

Fields with prescribed marginals must use inversion: the map has to be monotone in z so that the spatial
dependence of the Gaussian field carries over (a Gaussian copula). Rejection samplers, which give the fastest iid
variates for many families (Marsaglia-Tsang gamma, Cheng beta, PTRS Poisson), do not preserve that coupling, so the
relevant fast algorithms are fast numerical inversion. Methods, all exact or verified to a tolerance:

- ``closed``: closed-form quantile functions of Phi (uniform, exponential, Weibull, Laplace, Gumbel, ...), exact on
  any device, any spatially varying parameters (``transforms.CLOSED``);
- ``table``: piecewise cubic Hermite interpolation of q(z) = F^-1(Phi(z)) on a uniform z-grid with exact slopes
  q'(z) = phi(z) / f(q), refined until the relative error at all interval midpoints is below ``tol`` (the idea of
  Hoermann & Leydold's numerical inversion (PINV, UNU.RAN), set in the normal-score variable so the grid lookup is
  O(1) and the tails are resolved). Evaluation is a handful of flops per point on the CPU or GPU; points beyond the
  verified range (|z| > zmax, probability < 1e-15 by default) use exact ``ppf``. Spatially varying ``loc``/``scale``
  are supported; shape parameters must be constant;
- ``discrete-table``: exact inversion of a discrete CDF by binary search in its cumulative table (lower tail) and
  survival table (upper tail, so no precision is lost near u = 1); constant parameters, any device;
- ``ppf``: scipy's quantile function on the CPU (``isf`` in the upper tail), exact, any parameters.

``method="auto"`` picks ``table``/``discrete-table`` when the parameters allow it and the number of values to
transform pays for the table set-up, and ``ppf`` otherwise.
"""

import numpy as np
import scipy.special as sps
import scipy.stats as st
import torch
from scipy.interpolate import CubicHermiteSpline

TABLE_MIN_VALUES = 20_000  # below this many values per call, building a table costs more than calling ppf
# (placeholder until the benchmark fits it; table set-up is a few thousand ppf evaluations)
DISCRETE_TABLE_MAX = 1_000_000


def _support(dist):
    """Effective support. scipy's von Mises reports (-inf, inf) and its CDF leaves [0, 1] outside one period, so its
    support is declared here (``vonmises_line`` has the right support but a slow, quadrature-based CDF)."""
    if getattr(getattr(dist, "dist", None), "name", "") == "vonmises":
        loc, scale = dist.kwds.get("loc", 0.0), dist.kwds.get("scale", 1.0)
        if len(dist.args) > 1:
            loc = dist.args[1]
        if len(dist.args) > 2:
            scale = dist.args[2]
        return loc - np.pi * scale, loc + np.pi * scale
    a, b = dist.support()  # arrays (one value per point) when shape parameters vary in space
    if np.ndim(a) == 0 and np.ndim(b) == 0:
        return float(a), float(b)
    # scipy reports nan for degenerate parameters (e.g. Bernoulli with p = 0 or 1): use the family-wide bounds there
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.where(np.isnan(a), np.nanmin(a), a), np.where(np.isnan(b), np.nanmax(b), b)


def _native_ppf(dist) -> bool:
    """Whether scipy has a closed-form/special-function ppf for ``dist`` (otherwise it root-finds per element)."""
    gen = getattr(dist, "dist", None)
    if gen is None or isinstance(gen, st.rv_discrete):
        return True
    return type(gen)._ppf is not st.rv_continuous._ppf


def _q_exact(dist, z, loc=0.0, scale=1.0):
    """F^-1(Phi(z)) to full precision, using the survival function in the upper tail (no cancellation in 1 - u).

    Uses scipy's ppf/isf where they are special functions; otherwise a vectorised safeguarded Newton iteration on the
    CDF/SF, bracketed by a coarse ppf grid (scipy's generic ppf runs a scalar root finder per value).
    """
    z = np.asarray(z, dtype=float)
    if _native_ppf(dist) or z.size <= 64:
        with np.errstate(all="ignore"):
            q = np.where(z <= 0, dist.ppf(sps.ndtr(z)), dist.isf(sps.ndtr(-z)))
        q = np.array(q, dtype=float)
        bad = ~np.isfinite(q) & np.isfinite(z)
        if bad.any():
            if isinstance(getattr(dist, "dist", None), st.rv_discrete):
                q = _discrete_search(dist, z, q, bad)  # e.g. degenerate parameters (Bernoulli with p = 0 or 1)
            else:
                q[bad] = _q_newton(dist, z[bad])  # scipy's special functions can fail far in the tails
    else:
        q = _q_newton(dist, z)
    return loc + scale * q


def _discrete_search(dist, z, q, bad, max_steps=100_000):
    """The defining property min{k : F(k) >= Phi(z)} by stepping up from the lower end of the support, where scipy's
    ppf gives nan (parameters may vary per point, so whole arrays are evaluated)."""
    a = np.broadcast_to(np.asarray(dist.support()[0], dtype=float), q.shape)
    k = np.where(bad, np.where(np.isfinite(a), a, 0.0), q)
    u = sps.ndtr(z)
    for _ in range(max_steps):
        with np.errstate(all="ignore"):
            step = bad & (dist.cdf(k) < u)
        if not step.any():
            break
        k = k + step
    return k


def _q_newton(dist, z, iters=200):
    """Vectorised safeguarded Newton for F^-1(Phi(z)), relative accuracy ~1e-14 also deep in the tails.

    Solves g(y) = log F(x(y)) - log Phi(z) (or log Phi(-z) - log S(x(y)) for z > 0), increasing in y, with
    y = log x on positive supports and y = asinh(x / s) on the real line, so power-law and exponential tails become
    (close to) linear. Brackets come from a coarse ppf grid (scipy, |z| <= 9); ends that fail the sign check or are
    missing are found by geometric expansion in y. Converged entries leave the active set. ``z`` may have any shape.
    """
    shape = np.shape(z)
    z = np.asarray(z, dtype=float).ravel()  # the bracketing below works on flat indices
    zc = np.linspace(-9.0, 9.0, 37)
    with np.errstate(all="ignore"):
        qc = np.where(zc <= 0, dist.ppf(sps.ndtr(zc)), dist.isf(sps.ndtr(-zc)))
    a, b = (float(np.min(v)) if i == 0 else float(np.max(v)) for i, v in enumerate(_support(dist)))
    positive = a >= 0
    keep = np.isfinite(qc) & ((qc > 0) if positive else True)
    zc, qc = zc[keep], qc[keep]
    if positive:
        to_y, to_x, dxdy = np.log, np.exp, (lambda y: np.exp(y))
    else:
        sc = float(np.interp(1.0, zc, qc) - np.interp(-1.0, zc, qc)) or 1.0
        to_y, to_x, dxdy = (lambda x: np.arcsinh(x / sc)), (lambda y: sc * np.sinh(y)), (lambda y: sc * np.cosh(y))
    yc = to_y(qc)
    upper = z > 0
    log_target = np.where(upper, sps.log_ndtr(-z), sps.log_ndtr(z))

    def log_tail(x, up):
        """log F(x) (lower) or log S(x) (upper); log of cdf/sf where they do not underflow (scipy's logcdf is a slow
        generic fallback for many distributions), logcdf/logsf only where they do."""
        with np.errstate(all="ignore"):
            p = np.where(up, dist.sf(x), dist.cdf(x))
            out = np.log(p)
            tiny = ~(p > 1e-280)
            if tiny.any():
                out[tiny] = np.where(up[tiny], dist.logsf(x[tiny]), dist.logcdf(x[tiny]))
        return out

    def g(y, sel):
        lt = log_tail(to_x(y), upper[sel])
        return np.where(upper[sel], log_target[sel] - lt, lt - log_target[sel])

    def slope(y, sel):
        x = to_x(y)
        with np.errstate(all="ignore"):
            return np.exp(dist.logpdf(x) - log_tail(x, upper[sel])) * dxdy(y)

    j = np.searchsorted(zc, z)  # zc[j-1] < z <= zc[j]
    lo = np.where(j > 0, yc[np.clip(j - 1, 0, len(yc) - 1)], np.nan)
    hi = np.where(j < len(yc), yc[np.clip(j, 0, len(yc) - 1)], np.nan)
    for side, sign in ((lo, -1.0), (hi, 1.0)):  # scipy's ppf is not reliable far out: check every bracket end
        fin = np.flatnonzero(np.isfinite(side))
        gv = g(side[fin], fin)
        side[fin[(gv > 0) if sign < 0 else (gv <= 0)]] = np.nan
    ymax = np.log(np.finfo(float).max) - 1.0  # |y| beyond which x overflows / underflows
    ylo = -ymax if (positive and a == 0) or not np.isfinite(a) else max(-ymax, float(to_y(a)))
    yhi = ymax if not np.isfinite(b) else min(ymax, float(to_y(b)))
    for side, ref, sign in ((lo, yc[0], -1.0), (hi, yc[-1], 1.0)):
        idx = np.flatnonzero(np.isnan(side))
        width = 1.0
        while len(idx):
            cand = np.clip(ref + sign * width, ylo, yhi)
            gv = g(np.full(len(idx), cand), idx)
            done = ((gv > 0) if sign > 0 else (gv <= 0)) | (cand <= ylo) | (cand >= yhi)
            side[idx[done]] = cand
            idx = idx[~done]
            width *= 2.0
    y = 0.5 * (lo + hi)
    active = np.arange(len(z))
    for _ in range(iters):
        ya = y[active]
        gy = g(ya, active)
        hi[active] = np.where(gy > 0, ya, hi[active])
        lo[active] = np.where(gy <= 0, ya, lo[active])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            step = ya - gy / slope(ya, active)
        la, ha = lo[active], hi[active]
        ok = np.isfinite(step) & (step >= la) & (step <= ha)
        y_new = np.where(gy == 0, ya, np.where(ok, step, 0.5 * (la + ha)))
        tol = 1e-15 * (np.abs(y_new) + 1.0)
        conv = (np.abs(y_new - ya) <= tol) | (ha - la <= tol) | (np.abs(gy) <= 1e-14)  # CDFs are ~1e-15 accurate
        y[active] = y_new
        active = active[~conv]
        if not len(active):
            break
    return np.clip(to_x(y), a, b).reshape(shape)


class Inversion:
    """z -> loc + scale * F^-1(Phi(z)) for a frozen scipy distribution ``dist``.

    ``method``: "auto" | "table" | "discrete-table" | "ppf" (see the module docstring). With "auto" the table is
    built lazily, on the first call that transforms at least ``TABLE_MIN_VALUES`` values or runs on a GPU; shape
    parameters that vary in space force "ppf" (``loc``/``scale`` may vary with any method).
    """

    def __init__(self, dist, loc=0.0, scale=1.0, method="auto", tol=1e-10, zmax=8.0):
        if method not in ("auto", "table", "discrete-table", "ppf"):
            raise ValueError(f"unknown inversion method {method!r}")
        self.dist, self.loc, self.scale, self.tol, self.zmax_request = dist, loc, scale, tol, zmax
        self.discrete = isinstance(getattr(dist, "dist", None), st.rv_discrete)
        self.varying = _varying_shapes(dist)
        tabular = "discrete-table" if self.discrete else "table"
        if method not in ("auto", "ppf", tabular):
            raise ValueError(f"{method} inversion does not apply to this distribution")
        if method == tabular and self.varying:
            raise ValueError(f"{method} inversion needs constant shape parameters")
        self.requested = method
        self.method = None if method in ("auto", tabular) else "ppf"
        self.info = {"inversion": method}
        if method == "auto":
            pending = f"auto ({tabular} from {TABLE_MIN_VALUES} values or on GPU)"
            self.info["inversion"] = "ppf" if self.varying else pending
        if method == tabular:
            self.method = tabular
            self._build()

    def _build(self):
        if self.discrete:
            self._build_discrete(self.zmax_request)
        else:
            self._build_table(self.zmax_request)

    def _resolve(self, count, on_torch):
        if self.method is None:
            if self.varying or not (on_torch or count >= TABLE_MIN_VALUES):
                self.info["inversion"] = "ppf"
                if not self.varying:
                    self.info["note"] = f"auto: ppf below {TABLE_MIN_VALUES} values per call, table above"
                return "ppf"
            self.method = "discrete-table" if self.discrete else "table"
            self.info = {"inversion": self.method}
            self._build()
        return self.method

    @property
    def support(self):
        """(lo, hi) of the values, or None for discrete distributions (no clamping)."""
        if self.discrete:
            return None
        a, b = _support(self.dist)
        return self.loc + self.scale * a, self.loc + self.scale * b

    @property
    def closed_range(self):
        """(lo, hi) of a discrete distribution's values (attained), else None."""
        if not self.discrete:
            return None
        a, b = _support(self.dist)
        return self.loc + a, self.loc + b

    @property
    def torch_capable(self) -> bool:
        return not self.varying and self.requested != "ppf"

    # ------------------------------------------------------------------ continuous: Hermite table

    def _build_table(self, zmax):
        """Hermite table of a smoothing transform y = T(q): log q for positive supports, asinh(q / s) otherwise,
        which tames power-law and exponential tails; the tolerance is checked on q itself."""
        d = self.dist
        positive = _support(d)[0] >= 0
        for zm in (zmax, 6.0, 4.0):
            for m in (2**k for k in range(8, 16)):
                z = np.linspace(-zm, zm, m + 1)
                q = _q_exact(d, z)
                if not np.all(np.isfinite(q)) or np.any(np.diff(q) < 0) or (positive and q[0] <= 0):
                    break
                if m == 256:
                    s = float(abs(q[np.searchsorted(z, 1.0)] - q[np.searchsorted(z, -1.0)])) or 1.0
                    fwd, inv, slope = _transform(positive, s)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    dq = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi) / d.pdf(q)
                bad = ~np.isfinite(dq)
                if bad.any():
                    dq[bad] = np.gradient(q, z)[bad]
                spline = CubicHermiteSpline(z, fwd(q), slope(q, dq))
                mid = np.concatenate([z[:-1] + 0.25 * (z[1] - z[0]), z[:-1] + 0.5 * (z[1] - z[0])])
                exact = _q_exact(d, mid)
                with np.errstate(over="ignore", invalid="ignore"):
                    err = np.max(np.abs(inv(spline(mid)) - exact) / (np.abs(exact) + s))
                if err <= self.tol:
                    self.z0, self.h, self.zmax = -zm, z[1] - z[0], zm
                    self.coef = np.ascontiguousarray(spline.c.T)  # (m, 4): c3 t^3 + c2 t^2 + c1 t + c0
                    self.positive, self.s = positive, s
                    self.info.update(intervals=m, zmax=zm, variable="log q" if positive else "asinh(q/s)",
                                     max_rel_error=float(err))
                    self._tcache = {}
                    return
        self.method = "ppf"  # tolerance not reachable: fall back to exact
        self.info = {"inversion": "ppf", "note": "table tolerance not reached"}

    def _table(self, z, xp):
        if xp.torch:
            key = (z.device, z.dtype)
            if key not in self._tcache:
                self._tcache[key] = torch.as_tensor(self.coef, device=z.device, dtype=z.dtype)
            coef = self._tcache[key]
            m = coef.shape[0]
            idx = ((z - self.z0) / self.h).floor().clamp(0, m - 1).long()
            t = z - (self.z0 + idx.to(z.dtype) * self.h)
            c = coef[idx]
            y = ((c[..., 0] * t + c[..., 1]) * t + c[..., 2]) * t + c[..., 3]
            q = torch.exp(y) if self.positive else self.s * torch.sinh(y)
            out_of_range = z.abs() > self.zmax
            if out_of_range.any():
                zo = z[out_of_range].double().cpu().numpy()
                q[out_of_range] = torch.as_tensor(_q_exact(self.dist, zo), device=z.device, dtype=z.dtype)
            return q
        m = self.coef.shape[0]
        idx = np.clip(np.floor((z - self.z0) / self.h).astype(np.int64), 0, m - 1)
        t = z - (self.z0 + idx * self.h)
        c = self.coef[idx]
        y = ((c[..., 0] * t + c[..., 1]) * t + c[..., 2]) * t + c[..., 3]
        q = np.exp(y) if self.positive else self.s * np.sinh(y)
        out_of_range = np.abs(z) > self.zmax
        if out_of_range.any():
            q[out_of_range] = _q_exact(self.dist, z[out_of_range])
        return q

    # ------------------------------------------------------------------ discrete: cumulative tables

    def _build_discrete(self, zmax):
        d = self.dist
        p_tail = sps.ndtr(-zmax - 1)
        a, b = _support(d)
        # bounds from the accurate cdf/sf by doubling (scipy's discrete isf is ppf(1 - q): off far in the upper tail)
        mid = float(d.ppf(0.5))
        lo, hi, step = mid, mid, 1.0
        while lo > a and d.cdf(lo - 1) >= p_tail and step < 2 * DISCRETE_TABLE_MAX:
            lo, step = max(a, mid - step), 2 * step
        step = 1.0
        while hi < b and d.sf(hi) > p_tail and step < 2 * DISCRETE_TABLE_MAX:
            hi, step = min(b, mid + step), 2 * step
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo > DISCRETE_TABLE_MAX:
            self.method = "ppf"
            self.info = {"inversion": "ppf", "note": "support too wide for a table"}
            return
        self.k = np.arange(lo, hi + 1)
        self.cdf = d.cdf(self.k)  # increasing: ppf(u) = k[first index with cdf >= u]
        self.neg_sf = -d.sf(self.k)  # increasing: the same index from the survival side, for u near 1
        self.info.update(table_size=len(self.k))
        self._tcache = {}

    def _discrete(self, z, xp):
        if xp.torch:
            key = (z.device, z.dtype)
            if key not in self._tcache:
                as_t = lambda a: torch.as_tensor(a, device=z.device, dtype=torch.float64)  # noqa: E731
                self._tcache[key] = (as_t(self.k), as_t(self.cdf), as_t(self.neg_sf))
            k, cdf, neg_sf = self._tcache[key]
            zd = z.double()
            lower = torch.searchsorted(cdf, torch.special.ndtr(zd).contiguous())
            # ppf(u) = first k with sf(k) <= 1 - u; side="left" on -sf finds first -sf(k) >= -(1-u)
            upper = torch.searchsorted(neg_sf, (-torch.special.ndtr(-zd)).contiguous())
            idx = torch.where(zd <= 0, lower, upper).clamp(0, len(k) - 1)
            return k[idx].to(z.dtype)
        lower = np.searchsorted(self.cdf, sps.ndtr(z))
        upper = np.searchsorted(self.neg_sf, -sps.ndtr(-z))
        idx = np.clip(np.where(z <= 0, lower, upper), 0, len(self.k) - 1)
        return self.k[idx]

    # ------------------------------------------------------------------

    def __call__(self, z, xp):
        method = self._resolve(z.numel() if xp.torch else z.size, xp.torch)
        if method == "table":
            q = self._table(z, xp)
        elif method == "discrete-table":
            q = self._discrete(z, xp)
        elif xp.torch:  # only reached if a table could not be built
            q = torch.as_tensor(_q_exact(self.dist, z.double().cpu().numpy()), device=z.device, dtype=z.dtype)
        else:
            q = _q_exact(self.dist, z)
        loc, scale = xp.const(self.loc, q), xp.const(self.scale, q)
        return loc + scale * q


def _transform(positive, s):
    """(forward, inverse, slope) of the table variable y = log q or asinh(q / s); slope maps (q, dq/dz) to dy/dz."""
    if positive:
        return np.log, np.exp, lambda q, dq: dq / q
    return (lambda q: np.arcsinh(q / s), lambda y: s * np.sinh(y),
            lambda q, dq: dq / (s * np.sqrt(1.0 + (q / s) ** 2)))


def _varying_shapes(dist) -> bool:
    return any(np.ndim(a) > 0 for a in list(getattr(dist, "args", ())) + list(getattr(dist, "kwds", {}).values()))
