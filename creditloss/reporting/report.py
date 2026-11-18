"""Charts, Excel workbook and a markdown summary of a run."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCEN_COL = {"baseline": "#2a78d6", "downside": "#eb6834", "upside": "#1baf7a", "severe": "#e34948"}
SCEN_ORDER = ["upside", "baseline", "downside", "severe"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SEG_LABEL = {"mortgage_uninsured": "Mortgages, uninsured", "mortgage_insured": "Mortgages, insured",
             "heloc": "HELOCs", "consumer": "Consumer", "commercial_re": "Commercial / CRE",
             "credit_cards": "Credit cards", "commercial": "Commercial + equipment"}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
        "lines.linewidth": 2})


def _save(fig, out: Path, name: str) -> str:
    p = out / f"{name}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p.name


def chart_pd_paths(res, cfg, out) -> str:
    segs = list(cfg["segments"])
    fig, axes = plt.subplots(len(segs), 1, figsize=(8, 2.3 * len(segs)), sharex=True)
    dcol = cfg["columns"]["default_rate"]
    for ax, seg in zip(axes, segs):
        h = res["portfolio"][seg][dcol]
        h = h[h.index >= "2004-01-01"]
        ax.plot(h.index, h.values, color=INK2, lw=1.5)
        p = res["paths"]
        for s in SCEN_ORDER:
            g = p[(p.segment == seg) & (p.scenario == s)]
            ax.plot(pd.to_datetime(g.date), 100 * g.pd_annual, color=SCEN_COL[s], lw=2)
            ax.annotate(s, (pd.to_datetime(g.date.iloc[-1]), 100 * g.pd_annual.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK2, va="center")
        ax.axvline(h.index[-1], color=GRID, lw=1)
        ax.set_title(f"{SEG_LABEL.get(seg, seg)}: annual default rate, %", loc="left")
    axes[-1].set_xlim(pd.Timestamp("2004-01-01"), None)
    fig.text(0.01, -0.01, "Grey: history (simulated). Coloured: projection by scenario, champion PD model.",
             color=INK2, fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "fig1_pd_paths")


def chart_loss_rates(res, cfg, out) -> str:
    s = res["summary"]
    s = s[s.horizon_q == cfg["horizon_quarters"]]
    segs = list(cfg["segments"])
    fig, axes = plt.subplots(1, len(segs), figsize=(11, 3.2))
    for ax, seg in zip(axes, segs):
        g = s[s.segment == seg].set_index("scenario").reindex(SCEN_ORDER)
        bars = ax.bar(range(4), g.loss_rate_pct, color=[SCEN_COL[x] for x in SCEN_ORDER], width=0.7,
                      edgecolor=SURF, linewidth=2)
        for b, v in zip(bars, g.loss_rate_pct):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(4), SCEN_ORDER, rotation=45, fontsize=8)
        ax.set_title(SEG_LABEL.get(seg, seg), loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Cumulative loss, % of avg EAD")
    fig.suptitle(f"Cumulative credit losses over {cfg['horizon_quarters']} quarters", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, out, "fig2_loss_rates")


def chart_lgd(res, cfg, out) -> str:
    segs = [s for s, c in cfg["segments"].items() if c["lgd_model"] in ("structural_ltv", "frye_jacobs")]
    fig, axes = plt.subplots(1, len(segs), figsize=(4 * len(segs), 3), sharey=False)
    axes = np.atleast_1d(axes)
    p = res["paths"]
    for ax, seg in zip(axes, segs):
        for s in SCEN_ORDER:
            g = p[(p.segment == seg) & (p.scenario == s)]
            ax.plot(g.q, 100 * g.lgd, color=SCEN_COL[s], label=s)
        ax.set_title(f"{SEG_LABEL.get(seg, seg)}: LGD, %", loc="left")
        ax.set_xlabel("Quarters from jump-off")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "fig3_lgd")


def chart_attribution(res, out) -> str:
    a = res["attribution"]
    tot = a[~a.component.str.startswith("  ")].groupby("component", sort=False)["amount"].sum() / 1e6
    order = ["Baseline loss", "Higher defaults (PD)", "Weaker collateral (LGD)", "Severe loss"]
    tot = tot.reindex(order)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    base = 0
    for i, k in enumerate(order):
        v = tot[k]
        if k in ("Baseline loss", "Severe loss"):
            ax.bar(i, v, color="#2a78d6" if i == 0 else "#e34948", width=0.6)
            base = v
            y = v
        else:
            ax.bar(i, v, bottom=base, color="#eb6834" if "PD" in k else "#1baf7a", width=0.6)
            base += v
            y = base
        ax.text(i, y, f"{v:,.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(4), ["Baseline", "+ higher PD", "+ weaker collateral", "Severe"])
    ax.set_ylabel("C$ millions, cumulative")
    ax.set_title("What drives severe-scenario losses (all segments)", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, out, "fig4_attribution")


def _model_colors(models, champion):
    return ["#e34948" if x == champion else "#9a9890" if "AR(1)" in x else "#2a78d6" for x in models]


def chart_model_range(res, out) -> str:
    from matplotlib.ticker import MaxNLocator
    m = res["model_range"]
    m = m[m.scenario == "severe"]
    segs = list(m.segment.unique())
    fig, axes = plt.subplots(1, len(segs), figsize=(12, 3.4))
    for ax, seg in zip(axes, segs):
        g = m[m.segment == seg].sort_values("loss_rate_pct")
        cols = _model_colors(g.model, res["champions"].get(seg))
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.scatter(g.loss_rate_pct, range(len(g)), c=cols, s=40, zorder=3)
        ax.set_yticks(range(len(g)), [x.replace(" (arrears regression)", "") for x in g.model], fontsize=7)
        ax.set_title(SEG_LABEL.get(seg, seg), loc="left", fontsize=9)
        ax.set_xlabel("Severe loss, % EAD", fontsize=8)
    fig.suptitle("Model risk: severe-scenario loss under each PD model (red = champion)", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, out, "fig5_model_range")


def chart_provisions(res, out) -> str:
    p = res["paths"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for s in SCEN_ORDER:
        g = p[p.scenario == s].groupby("q")[["pcl", "allowance_total"]].sum() / 1e6
        axes[0].plot(g.index[1:], g.pcl.iloc[1:], color=SCEN_COL[s], label=s)
        axes[1].plot(g.index, g.allowance_total, color=SCEN_COL[s], label=s)
    pw = res["pw_ecl_by_stage"]["allowance_total"].sum() / 1e6
    axes[1].scatter([0], [pw], color=INK, zorder=5, s=18)
    axes[1].annotate(" probability-weighted\n opening", (0, pw), fontsize=7, color=INK)
    axes[0].set_title("IFRS 9 provisions (change in ACL + write-offs), C$m/q", loc="left")
    axes[1].set_title("Allowance for credit losses, Stages 1-3, C$m", loc="left")
    for ax in axes:
        ax.set_xlabel("Quarters from jump-off")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "fig6_provisions")


def chart_stages(res, out) -> str:
    """Stage mix of the gross carrying amount and ECL by stage, baseline vs severe, all segments."""
    p = res["paths"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    cols = {"gca_s1": "#9ec5ee", "gca_s2": "#e9a23b", "gca_s3": "#c0392b"}
    for ax, s in zip(axes[:2], ["baseline", "severe"]):
        g = p[p.scenario == s].groupby("q")[list(cols)].sum()
        g = 100 * g.div(g.sum(axis=1), axis=0)
        ax.stackplot(g.index, *[g[c] for c in cols], colors=list(cols.values()),
                     labels=["Stage 1", "Stage 2", "Stage 3"])
        ax.set_ylim(0, 100)
        ax.set_title(f"Stage mix of gross loans, {s}, %", loc="left")
        ax.set_xlabel("Quarters from jump-off")
    axes[0].legend(fontsize=7, loc="lower left")
    for s in ["baseline", "severe"]:
        g = p[p.scenario == s].groupby("q")[["ecl_stage1", "ecl_stage2", "ecl_stage3"]].sum() / 1e6
        for c, ls in zip(g.columns, ["-", "--", ":"]):
            axes[2].plot(g.index, g[c], color=SCEN_COL[s], ls=ls, label=f"{s} {c[-6:].replace('stage', 'Stage ')}")
    axes[2].set_title("ECL by stage, C$m", loc="left")
    axes[2].legend(fontsize=6, ncol=2)
    axes[2].set_xlabel("Quarters from jump-off")
    fig.tight_layout()
    return _save(fig, out, "fig8_stages")


def chart_oos(oos_metrics: pd.DataFrame, stress: pd.DataFrame, out, champions: dict | None = None) -> str:
    m = oos_metrics[oos_metrics.h.astype(str).str.startswith("1-")]
    segs = list(m.segment.unique())
    fig, axes = plt.subplots(1, len(segs), figsize=(12, 3.4))
    for ax, seg in zip(np.atleast_1d(axes), segs):
        g = m[m.segment == seg].sort_values("rmse_pp", ascending=False)
        cols = _model_colors(g.model, (champions or {}).get(seg))
        ax.barh(range(len(g)), g.rmse_pp, color=cols, height=0.6)
        ax.set_yticks(range(len(g)), [x.replace(" (arrears regression)", "") for x in g.model], fontsize=7)
        ax.set_title(SEG_LABEL.get(seg, seg), loc="left", fontsize=9)
        ax.set_xlabel("RMSE, pp of annual PD", fontsize=8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Out-of-sample error, 1-12 quarters ahead, re-selected at every origin (red = champion)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, out, "fig7_oos_rmse")


def spec_table(res) -> pd.DataFrame:
    rows = []
    for seg, sel in res["sels"].items():
        for nm, sp in sel["specs"].items():
            rows.append({"segment": seg, "model": nm, "specification": sp.label(),
                         "coefficients": "; ".join(f"{k}={v:.4f} (t={sp.tstats.get(k, np.nan):.1f})"
                                                   for k, v in sp.params.items()),
                         "R2": sp.r2, "BIC": sp.bic, "dynamic_RMSE_pp": sp.dyn_rmse, "n": sp.nobs})
    return pd.DataFrame(rows)


def write_excel(res, out: Path, oos: dict | None = None) -> str:
    p = out / "credit_loss_results.xlsx"
    with pd.ExcelWriter(p, engine="openpyxl") as xw:
        res["summary"].to_excel(xw, sheet_name="summary", index=False)
        res["paths"].to_excel(xw, sheet_name="quarterly_paths", index=False)
        spec_table(res).to_excel(xw, sheet_name="pd_models", index=False)
        res["model_range"].to_excel(xw, sheet_name="model_risk_range", index=False)
        res["nco_check"].to_excel(xw, sheet_name="top_down_check", index=False)
        res["pw_ecl"].to_excel(xw, sheet_name="prob_weighted_ecl")
        res["attribution"].to_excel(xw, sheet_name="attribution", index=False)
        if "ifrs9" in res:
            d = res["ifrs9"]
            d["pw_by_stage"].to_excel(xw, sheet_name="ifrs9_pw_ecl_by_stage")
            d["coverage"].to_excel(xw, sheet_name="ifrs9_coverage_pct")
            d["sensitivity"].to_excel(xw, sheet_name="ifrs9_scenario_sensitivity")
            d["staging"].to_excel(xw, sheet_name="ifrs9_staging_sensitivity")
            d["rollforward"].to_excel(xw, sheet_name="ifrs9_acl_rollforward", index=False)
            d["gca"].to_excel(xw, sheet_name="ifrs9_gca_rollforward", index=False)
        res["pd_paths"].to_excel(xw, sheet_name="pd_paths_all_models", index=False)
        for seg, sel in res["sels"].items():
            sel["search"].head(200).assign(drivers=lambda d: d.drivers.astype(str)).to_excel(
                xw, sheet_name=f"search_{seg}"[:31], index=False)
        if oos:
            for k, v in oos.items():
                v.to_excel(xw, sheet_name=f"oos_{k}"[:31], index=False)
    return p.name
