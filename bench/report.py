"""Turn a run directory into report.md, recommendations.json and plots.

    python bench/report.py results/runs/<run>

Validity rules used for "best" choices:
- noise: the method must be exact by construction and pass its small-N covariance check (float64: 1e-10, float32: 1e-5);
- prior: status ok, and the pathwise error against the float64 direct reference must be <= 1e-6 (float64 runs) or
  <= 1e-4 (float32 runs); the law error, where measured, must satisfy the same bound.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cases import CPU_SOLVERS, GPU_SOLVERS  # noqa: E402

SOLVER_OPTIONS = {**CPU_SOLVERS, **GPU_SOLVERS}
TOL = {"float64": 1e-6, "float32": 1e-4}
NOISE_TOL = {"float64": 1e-10, "float32": 1e-5}


def load(run_dir):
    out = {}
    for suite in ("noise", "prior", "posterior", "distributions", "lowrank", "covariance", "continuum"):
        path = os.path.join(run_dir, f"{suite}.jsonl")
        if os.path.exists(path):
            with open(path) as fh:
                rows = [json.loads(line) for line in fh]
            df = pd.DataFrame(rows)
            if suite != "continuum":
                df = df.drop_duplicates("task_id", keep="last")
            out[suite] = df
    with open(os.path.join(run_dir, "environment.json")) as fh:
        out["env"] = json.load(fh)
    cfg = os.path.join(run_dir, "config.json")
    out["config"] = json.load(open(cfg)) if os.path.exists(cfg) else {}
    return out


def platform(df):
    return np.where(
        df.engine == "ngsolve",
        "cpu-ngsolve",
        np.where(df.device == "cpu", "cpu-torch", "cuda-" + df.dtype.str.replace("float", "f")),
    )


def md(df, floatfmt=".3g"):
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    return df.to_markdown(floatfmt=floatfmt) + "\n"


def fmt_time(s):
    if pd.isna(s):
        return ""
    return f"{s * 1e3:.3g} ms" if s < 1 else f"{s:.3g} s"


# ------------------------------------------------------------------ noise


def noise_section(df, plots_dir, recs):
    out = ["## 1. White-noise layer: z ~ N(0, M)\n"]
    df = df.assign(platform=platform(df))
    ok = df[df.status == "ok"].copy()
    ok["valid"] = ok.exact.astype(bool) & ok.apply(
        lambda r: pd.isna(r.get("cov_rel_err")) or r["cov_rel_err"] <= NOISE_TOL[r["dtype"]], axis=1
    )

    out.append(
        "### 1.1 Validation: worst small-N covariance error (max|HH^T - M|/max|M| for nrand <= 2000, else the "
        "probe residual max_k |HH^T v_k - M v_k|/|M v_k| via the exact adjoint)\n"
    )
    if "cov_rel_err" in ok:
        piv = ok.pivot_table(index="method", columns="platform", values="cov_rel_err", aggfunc="max")
        out.append(md(piv.map(lambda v: f"{v:.1e}" if pd.notna(v) else ""), floatfmt="s"))
        bad = ok[ok.exact.astype(bool) & (ok.cov_rel_err > ok.dtype.map(NOISE_TOL))]
        out.append(
            f"\n**{len(bad)} exact-method checks failed.**\n"
            if len(bad)
            else "\nAll exact methods reproduce M to the precision tolerance on every platform.\n"
        )
        if len(bad):
            out.append(md(bad[["dim", "kind", "order", "ndof", "platform", "method", "cov_rel_err"]]))

    out.append("\n### 1.2 Fastest exact method per space and size (time per sample, single sample)\n")
    rows, best_map = [], {}
    for key, sub in ok[ok.valid].groupby(["dim", "kind", "order", "target"]):
        row = dict(zip(["dim", "kind", "order", "target"], key))
        row["ndof"] = int(sub.ndof.max())
        for plat, s in sub.groupby("platform"):
            b = s.loc[s.per_sample_s.idxmin()]
            row[plat] = f"{b.method} ({fmt_time(b.per_sample_s)})"
            best_map.setdefault(plat, {})[f"{key[0]}|{key[1]}|{key[2]}|{int(key[3])}"] = {
                "method": b.method,
                "per_sample_s": float(b.per_sample_s),
            }
        rows.append(row)
    recs["noise"] = best_map
    out.append(md(pd.DataFrame(rows).set_index(["dim", "kind", "order", "target"]), floatfmt="s"))

    for plat, sub in ok.groupby("platform"):
        out.append(f"\n### 1.3 [{plat}] time per sample at the largest size reached\n")
        last = sub.sort_values("ndof").groupby(["dim", "kind", "order", "method"]).tail(1)
        piv = last.pivot_table(index=["dim", "kind", "order"], columns="method", values="per_sample_s")
        out.append(md(piv.map(fmt_time), floatfmt="s"))
        if "per_sample_batch_s" in sub and sub.per_sample_batch_s.notna().any():
            out.append(f"\nBatched (per sample, batch {int(sub.batch.max())}):\n")
            piv = last.pivot_table(index=["dim", "kind", "order"], columns="method", values="per_sample_batch_s")
            out.append(md(piv.map(fmt_time), floatfmt="s"))
        noise_plot(sub, plat, plots_dir)
    out.append("\nRandom numbers per dof (largest size):\n")
    last = ok.sort_values("ndof").groupby(["dim", "kind", "order", "method"]).tail(1)
    out.append(
        md(
            last.assign(r=last.nrand / last.nfree)
            .pivot_table(index=["dim", "kind", "order"], columns="method", values="r")
            .round(2)
        )
    )
    return "\n".join(out)


def noise_plot(sub, plat, plots_dir):
    methods = sorted(sub.method.unique())
    colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(methods)}
    for (dim, kind), s in sub.groupby(["dim", "kind"]):
        orders = sorted(s.order.unique())
        fig, axes = plt.subplots(1, len(orders), figsize=(3.6 * len(orders) + 1.6, 3.2), sharey=True, squeeze=False)
        for ax, p in zip(axes[0], orders):
            for m, c in s[s.order == p].groupby("method"):
                c = c.sort_values("ndof")
                ax.loglog(c.ndof, c.per_sample_s, "o-", ms=3, color=colors[m], label=m)
            ax.set_title(f"{dim}D {kind} p={p}", fontsize=9)
            ax.set_xlabel("ndof")
            ax.grid(True, which="both", alpha=0.3)
        axes[0][0].set_ylabel("time per sample [s]")
        handles = [
            plt.Line2D([], [], color=colors[m], marker="o", ms=3, label=m) for m in methods if m in set(s.method)
        ]
        fig.legend(handles=handles, loc="center right", fontsize=7)
        fig.suptitle(f"white noise, {plat}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1 - 1.6 / (3.6 * len(orders) + 1.6), 1))
        fig.savefig(os.path.join(plots_dir, f"noise_{plat}_{dim}D_{kind}.png"), dpi=110)
        plt.close(fig)


# ------------------------------------------------------------------ prior


def prior_section(df, plots_dir, recs):
    out = ["## 2. Prior samplers: C = A^-power (Matern-type), noise chosen automatically\n"]
    df = df.assign(platform=platform(df))
    df["config"] = df.platform + ":" + df.solver
    ok = df[df.status == "ok"].copy()
    tol = ok.dtype.map(TOL)
    path_ok = ok.pathwise_rel_err.isna() | (ok.pathwise_rel_err <= tol) if "pathwise_rel_err" in ok else True
    law_ok = ok.law_rel_fro.isna() | (ok.law_rel_fro <= tol) if "law_rel_fro" in ok else True
    ok["valid"] = path_ok & law_ok

    out.append("### 2.1 Fastest valid configuration per case (time per sample; batched where faster)\n")
    ok["best_time"] = (
        ok[["per_sample_s", "per_sample_batch_s"]].min(axis=1) if "per_sample_batch_s" in ok else ok.per_sample_s
    )
    rows, best_map = [], {}
    for key, sub in ok[ok.valid].groupby(["dim", "kind", "order", "power", "target"]):
        row = dict(zip(["dim", "kind", "order", "power", "target"], key))
        row["ndof"] = int(sub.ndof.max())
        for label, s in (("cpu", sub[sub.device == "cpu"]), ("gpu", sub[sub.device == "cuda"])):
            if len(s):
                b = s.loc[s.best_time.idxmin()]
                row[f"best {label}"] = f"{b.config} ({fmt_time(b.best_time)})"
                row[f"{label}_s"] = b.best_time
        if "cpu_s" in row and "gpu_s" in row:
            row["gpu speedup"] = f"{row['cpu_s'] / row['gpu_s']:.2g}x"
        for plat, s in sub.groupby("platform"):
            b = s.loc[s.best_time.idxmin()]
            best_map.setdefault(plat, {})[f"{key[0]}|{key[1]}|{key[2]}|{key[3]:g}|{int(key[4])}"] = {
                "solver": b.solver,
                "method": {**SOLVER_OPTIONS.get(b.solver, {}), "engine": b.engine},
                "noise": b.get("noise"),
                "device": b.device,
                "dtype": b.dtype,
                "per_sample_s": float(b.per_sample_s),
                "per_sample_batch_s": float(b.get("per_sample_batch_s", np.nan)),
            }
        rows.append({k: v for k, v in row.items() if not k.endswith("_s")})
    recs["prior"] = best_map
    out.append(md(pd.DataFrame(rows).set_index(["dim", "kind", "order", "power", "target"]), floatfmt="s"))

    out.append("\n### 2.2 Time per sample by configuration (single sample)\n")
    piv = ok.pivot_table(index=["dim", "kind", "order", "power", "ndof"], columns="config", values="per_sample_s")
    out.append(md(piv.map(fmt_time), floatfmt="s"))
    out.append("\n### 2.3 Setup time\n")
    piv = ok.pivot_table(index=["dim", "kind", "order", "power", "ndof"], columns="config", values="setup_s")
    out.append(md(piv.map(fmt_time), floatfmt="s"))
    if "pathwise_rel_err" in ok:
        out.append("\n### 2.4 Accuracy: worst pathwise error vs float64 direct reference (same noise)\n")
        piv = ok.pivot_table(index=["dim", "kind", "power"], columns="config", values="pathwise_rel_err", aggfunc="max")
        out.append(md(piv.map(lambda v: f"{v:.1e}" if pd.notna(v) else ""), floatfmt="s"))
    if "law_rel_fro" in ok:
        out.append("\nWorst exact discrete-law error (small N), relative Frobenius:\n")
        piv = ok.pivot_table(index=["dim", "kind", "power"], columns="config", values="law_rel_fro", aggfunc="max")
        out.append(md(piv.map(lambda v: f"{v:.1e}" if pd.notna(v) else ""), floatfmt="s"))
    if "cg_iterations" in ok:
        out.append("\n### 2.5 Iterative solver iterations (mean per solve)\n")
        it = ok[ok.cg_iterations.notna()]
        piv = it.pivot_table(index=["dim", "kind", "order", "power", "ndof"], columns="config", values="cg_iterations")
        out.append(md(piv.round(1)))
    prior_plot(ok, plots_dir)
    return "\n".join(out)


def prior_plot(ok, plots_dir):
    configs = sorted(ok.config.unique())
    colors = {c: plt.cm.tab20(i % 20) for i, c in enumerate(configs)}
    for (dim, kind, power), s in ok.groupby(["dim", "kind", "power"]):
        orders = sorted(s.order.unique())
        fig, axes = plt.subplots(1, len(orders), figsize=(3.6 * len(orders) + 2.2, 3.2), sharey=True, squeeze=False)
        for ax, p in zip(axes[0], orders):
            for c, d in s[s.order == p].groupby("config"):
                d = d.sort_values("ndof")
                ax.loglog(d.ndof, d.best_time, "o-", ms=3, color=colors[c])
            ax.set_title(f"{dim}D {kind} p={p} A^-{power}", fontsize=9)
            ax.set_xlabel("ndof")
            ax.grid(True, which="both", alpha=0.3)
        axes[0][0].set_ylabel("time per sample [s]")
        handles = [
            plt.Line2D([], [], color=colors[c], marker="o", ms=3, label=c) for c in configs if c in set(s.config)
        ]
        fig.legend(handles=handles, loc="center right", fontsize=6)
        fig.tight_layout(rect=(0, 0, 1 - 2.2 / (3.6 * len(orders) + 2.2), 1))
        fig.savefig(os.path.join(plots_dir, f"prior_{dim}D_{kind}_power{power}.png"), dpi=110)
        plt.close(fig)


# ------------------------------------------------------------------ posterior


def posterior_section(df, recs):
    out = [
        "## 3. Posteriors under linear observations (m local averages, A^-2 prior)\n",
        "fe-space / noise-space = the exact low-rank update applied to the prior sample / to the white noise; "
        "fe-space-reweighted = fe-space plus the cost of a re-weighted posterior and one sample from it "
        "(reweight_and_sample_s); fe-space-rank32 = randomised low-rank version; matheron = pathwise conditioning "
        "(reference implementation, CPU dense algebra).\n",
    ]
    df = df.assign(platform=platform(df))
    df["m"] = df.solver.str.replace("obs", "").astype(int)
    ok = df[df.status == "ok"].copy()
    cols = [c for c in ("setup_s", "posterior_s", "per_sample_s", "overhead_vs_prior", "reweight_and_sample_s")
            if c in ok]
    out.append("### 3.1 Cost (largest size per space)\n")
    last = ok.sort_values("ndof").groupby(["dim", "kind", "order", "platform", "method", "m"]).tail(1)
    tab = last.set_index(["dim", "kind", "order", "ndof", "platform", "method", "m"])[cols].sort_index()
    out.append(md(tab.map(lambda v: f"{v:.3g}" if pd.notna(v) else ""), floatfmt="s"))
    acc = [c for c in ("law_rel_fro", "pred_cov_rel_err", "mean_rel_err") if c in ok]
    if acc:
        out.append(
            "\n### 3.2 Accuracy (worst case): exact law at small N; predictive covariance and mean vs float64 "
            "Matheron for m <= 100\n"
        )
        piv = ok.groupby(["platform", "method"])[acc].max()
        out.append(md(piv.map(lambda v: f"{v:.1e}" if pd.notna(v) else ""), floatfmt="s"))
    best = {}
    for key, sub in ok.groupby(["dim", "kind", "order", "target", "m", "platform"]):
        valid = sub[
            (sub.get("pred_cov_rel_err", pd.Series(0, index=sub.index)).fillna(0) <= sub.dtype.map(TOL))
            & (~sub.method.str.startswith("fe-space-rank32"))
            & (sub.method != "fe-space-reweighted")  # the same sampler as fe-space
        ]
        if len(valid):
            b = valid.loc[valid.per_sample_s.idxmin()]
            best.setdefault(key[5], {})[f"{key[0]}|{key[1]}|{key[2]}|{key[4]}|{int(key[3])}"] = {
                "method": b.method,
                "per_sample_s": float(b.per_sample_s),
                "setup_s": float(b.setup_s),
            }
    recs["posterior"] = best
    return "\n".join(out)


# ------------------------------------------------------------------ distributions


def _cell(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return f"{v:.3g}" if isinstance(v, (float, int, np.floating, np.integer)) else str(v)


def distributions_section(df):
    out = [
        "## 3b. Distribution families (Matern A^-2 prior underneath, via ngsrng.Generator)\n",
        "Every algorithm variant of a family is timed: inversion (closed / table / ppf / discrete table) and "
        "construction (chi^2 squares vs copula, Wishart process vs Bartlett, t-process vs t-copula). Pointwise "
        "families live at quadrature points; Student-t processes and type-G fields are FE functions. Validity: KS "
        "p-value of known marginals (should not be tiny), Poisson mean error, variance ratio vs theory (~1).\n",
    ]
    df = df.assign(platform=platform(df))
    ok = df[df.status == "ok"]
    cols = [
        c
        for c in (
            "setup_s",
            "per_sample_s",
            "out_per_sample_s",
            "overhead_vs_gauss",
            "inversion",
            "standardisation",
            "points",
            "ks_min_pvalue",
            "mean_rel_err",
            "variance_ratio",
        )
        if c in ok
    ]
    last = ok.sort_values("ndof").groupby(["dim", "kind", "order", "platform", "method"]).tail(1)
    tab = last.set_index(["dim", "kind", "order", "ndof", "platform", "method"])[cols].sort_index()
    out.append(md(tab.map(_cell), floatfmt="s"))
    return "\n".join(out)


# ------------------------------------------------------------------ low rank


def lowrank_section(df):
    out = [
        "## 3c. Truncated Karhunen-Loeve expansions (Matern A^-2 prior)\n",
        "subspace = randomised subspace iteration on the factor L (Aarset 2024, Alg. 2; 2q + 2 applications); "
        "covariance = range finder on C = L L^T M with Rayleigh-Ritz (2q + 3 applications). Errors against the dense "
        "eigendecomposition (small problems only).\n",
    ]
    df = df.assign(platform=platform(df))
    ok = df[df.status == "ok"]
    cols = [c for c in ("setup_s", "applications", "rank", "eig_rel_err_max", "eig_rel_err_last",
                        "trace_captured_ratio") if c in ok]
    last = ok.sort_values("ndof").groupby(["dim", "kind", "order", "platform", "method", "solver"]).tail(1)
    tab = last.set_index(["dim", "kind", "order", "ndof", "platform", "solver", "method"])[cols].sort_index()
    out.append(md(tab.map(_cell), floatfmt="s"))
    return "\n".join(out)


# ------------------------------------------------------------------ covariance diagnostics


def covariance_section(df):
    out = [
        "## 3d. Covariance diagnostics: pointwise variance k(x, x) and trace tr(C) (Matern A^-2 prior, posterior)\n",
        "Variance: exact = local covariance from all columns (prior) or prior + exact low-rank update (posterior); "
        "lowrank = exact randomised low-rank part + Monte Carlo residual; mc = Monte Carlo. Relative errors at the "
        "quadrature points against the exact local covariance. Trace: relative error against the exact trace; "
        "applications = applications of L or L^T.\n",
    ]
    ok = df[df.status == "ok"]
    for kind in ("variance", "trace"):
        part = ok[ok.method.str.startswith(kind)]
        if part.empty:
            continue
        cols = [c for c in ("setup_s", "applications", "rel_err_rms", "rel_err_max", "stderr_rel_median", "rel_err",
                            "stderr_rel") if c in part and part[c].notna().any()]
        last = part.sort_values("ndof").groupby(["dim", "kind", "order", "method", "solver"]).tail(1)
        tab = last.set_index(["dim", "kind", "order", "ndof", "solver", "method"])[cols].sort_index()
        out.append(md(tab.map(_cell), floatfmt="s"))
    return "\n".join(out)


# ------------------------------------------------------------------ continuum


def continuum_section(df, plots_dir):
    out = [
        "## 4. Continuum accuracy: Var<f_h, e_k> vs exact Var<f, e_k> (Neumann eigenfunctions, unit cube)\n",
        "Independent of the sampling algorithm: this is what the choice of space, order and mesh costs.\n",
    ]
    ok = df[df.status == "ok"]
    rows = []
    for key, s in ok.groupby(["dim", "kind", "order", "power", "mode"]):
        s = s.sort_values("h")
        fine = s.head(3)
        # rates are meaningless once the error sits at round-off (e.g. the constant mode)
        usable = fine[fine.rel_err > 1e-10]
        rate = np.polyfit(np.log(usable.h), np.log(usable.rel_err), 1)[0] if len(usable) >= 2 else np.nan
        rows.append(
            {
                **dict(zip(["dim", "kind", "order", "power", "mode"], key)),
                "finest ndof": int(s.ndof.max()),
                "rel err (finest)": s.rel_err.iloc[0],
                "observed rate in h": rate,
            }
        )
    tab = pd.DataFrame(rows).set_index(["dim", "kind", "order", "power", "mode"])
    out.append(md(tab))
    for dim, s in ok.groupby("dim"):
        powers = sorted(s.power.unique())
        fig, axes = plt.subplots(1, len(powers), figsize=(4.2 * len(powers), 3.4), squeeze=False)
        mode = sorted(s["mode"].unique())[1]
        for ax, pw in zip(axes[0], powers):
            for (kind, order), d in s[(s.power == pw) & (s["mode"] == mode)].groupby(["kind", "order"]):
                d = d.sort_values("ndof")
                ax.loglog(d.ndof, d.rel_err, "o-" if kind == "H1" else "s--", ms=3, label=f"{kind} p={order}")
            ax.set_title(f"{dim}D, A^-{pw}, mode {mode}", fontsize=9)
            ax.set_xlabel("ndof")
            ax.grid(True, which="both", alpha=0.3)
        axes[0][0].set_ylabel("relative variance error")
        axes[0][-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"continuum_{dim}D.png"), dpi=110)
        plt.close(fig)
    return "\n".join(out)


# ------------------------------------------------------------------ main


def rejected_section(run_dir):
    """The candidates the calibration dropped (not measured at larger sizes), with the reason."""
    path = os.path.join(run_dir, "rejected.json")
    if not os.path.exists(path):
        return ""
    with open(path) as fh:
        rejected = json.load(fh)
    out = ["## Rejected candidates\n",
           "Dropped for larger sizes: failures and timeouts, misses of the accuracy threshold (noise: covariance error "
           "1e-10; priors: pathwise error 1e-6 against the float64 direct reference), and candidates 20x slower than "
           "the best accurate one on their platform (iterative solvers in 3D excepted).\n"]
    if not rejected:
        return out[0] + "None.\n"
    rows = []
    for lineage, reason in sorted(rejected.items()):
        suite, dim, kind, order, engine, device, dtype, method, solver, power = lineage.split("|")
        rows.append({"suite": suite, "space": f"{dim}D {kind} p={order}", "platform": f"{engine}/{device}",
                     "candidate": solver if suite == "prior" else method,
                     "power": power if suite == "prior" else "", "reason": reason})
    out.append(md(pd.DataFrame(rows).set_index(["suite", "platform", "candidate"]).sort_index(), floatfmt="s"))
    return "\n".join(out)


def failures_section(data):
    out = ["## 5. Failures, timeouts and inapplicable combinations\n"]
    frames = [d.assign(suite=s) for s, d in data.items() if isinstance(d, pd.DataFrame) and "status" in d]
    if not frames:
        return ""
    df = pd.concat(frames, ignore_index=True)
    bad = df[df.status != "ok"].copy()
    if not len(bad):
        return out[0] + "None.\n"
    bad["reason"] = bad.get("error", pd.Series(index=bad.index, dtype=str)).fillna("").str.slice(0, 90)
    tab = bad.groupby(["suite", "status", "method", "solver", "reason"]).size().rename("count").reset_index()
    out.append(md(tab.sort_values("count", ascending=False).set_index(["suite", "status"]), floatfmt="s"))
    return "\n".join(out)


def write(run_dir):
    data = load(run_dir)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    env, cfg = data["env"], data["config"]
    head = [
        f"# ngsrng benchmark report: {os.path.basename(os.path.abspath(run_dir))}\n",
        f"- profile: `{cfg.get('profile')}`, suites: {cfg.get('suites')}, devices: {cfg.get('devices')}",
        f"- machine: {env['cpu']} ({env['cpu_count']} threads), {env['mem_total_gb']:.0f} GB RAM; "
        f"GPU: {env['cuda']['device'] + ' %.0f GB' % env['cuda']['mem_total_gb'] if env.get('cuda') else 'none'}",
        f"- software: {env['versions']}, git {env.get('git')}, started {env['time']}",
        "- per-sample times are wall-clock medians; CPU engine = NGSolve kernels + threaded NumPy RNG; "
        "GPU engine = torch",
        "",
    ]
    counts = {
        s: d.status.value_counts().to_dict() for s, d in data.items() if isinstance(d, pd.DataFrame) and "status" in d
    }
    head.append(f"- record status counts: {counts}\n")
    recs = {"run": os.path.basename(os.path.abspath(run_dir)), "environment": env}
    parts = head
    if "noise" in data:
        parts.append(noise_section(data["noise"], plots_dir, recs))
    if "prior" in data:
        parts.append(prior_section(data["prior"], plots_dir, recs))
    if "posterior" in data:
        parts.append(posterior_section(data["posterior"], recs))
    if "distributions" in data:
        parts.append(distributions_section(data["distributions"]))
    if "lowrank" in data:
        parts.append(lowrank_section(data["lowrank"]))
    if "covariance" in data:
        parts.append(covariance_section(data["covariance"]))
    if "continuum" in data:
        parts.append(continuum_section(data["continuum"], plots_dir))
    parts.append(rejected_section(run_dir))
    parts.append(failures_section(data))
    plots = sorted(os.listdir(plots_dir))
    parts.append("## 6. Plots\n\n" + "\n".join(f"![{p}](plots/{p})" for p in plots))
    with open(os.path.join(run_dir, "report.md"), "w") as fh:
        fh.write("\n".join(parts) + "\n")
    with open(os.path.join(run_dir, "recommendations.json"), "w") as fh:
        json.dump(recs, fh, indent=1, default=str)


if __name__ == "__main__":
    write(sys.argv[1])
    print("wrote", os.path.join(sys.argv[1], "report.md"))
