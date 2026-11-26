"""Run every extension and write outputs/<config>/extensions/: CSV tables, PNG charts and one HTML page."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from creditloss.extensions import bvar, canada, history, panel, uncertainty  # noqa: E402
from creditloss.extensions.context import Context  # noqa: E402


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _table(df: pd.DataFrame, digits: int = 1) -> str:
    return df.round(digits).to_html(index=False, border=0, classes="t", na_rep="")


def run(config: str, n_draws: int = 400, n_bvar: int = 3000, log=print) -> Path:
    ctx = Context(config)
    out = Path(ctx.cfg["paths"]["outputs"]) / "extensions"
    out.mkdir(parents=True, exist_ok=True)
    sec: list[tuple[str, str, str]] = []           # (title, text, html body)

    # 1-3. Canadian data, payment shock, provincial panel ------------------------------------------
    log("[ext] provincial panel and payment shock")
    canada.write_sources()
    p = canada.panel(ctx.macro)
    coefs = panel.coefficients(p)
    rmse, dm, _ = panel.backtest(p)
    hold = panel.holdout(p)
    scen_pan = panel.scenario_losses(ctx, p)
    for name, df in {"panel_coefficients": coefs, "panel_oos_rmse": rmse, "panel_dm": dm,
                     "panel_holdout_2023_2026": hold.reset_index(names="date"), "panel_scenario_losses": scen_pan}.items():
        df.to_csv(out / f"{name}.csv", index=False)
    ps = canada.payment_shock(ctx.macro["R10Y"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ps.plot(ax=ax[0], color="#2a6f97")
    ax[0].axhline(0, color="grey", lw=0.6)
    ax[0].set_title("Renewal payment shock, % (5-year-old mortgage repriced today)")
    hold.plot(ax=ax[1])
    ax[1].set_title("Ontario arrears %, models estimated to 2022Q4")
    ax[1].legend(fontsize=7)
    sec.append(("Payment shock and provincial panel",
                "Pooled across regions the payment shock has the wrong sign; it matters in Ontario. Out of sample "
                "(origins every year from 2007, horizons 1-12) no model beats Ontario UR+HPI significantly, and every "
                "model misses the 2023-26 rise in Ontario arrears. Keep the champion; use the panel with the payment "
                "shock as a challenger and as the basis for a baseline renewal overlay.",
                f'<img src="data:image/png;base64,{_png(fig)}"><h4>Coefficients</h4>{_table(coefs, 4)}'
                f"<h4>Out-of-sample RMSE, Ontario arrears (pp)</h4>{_table(rmse, 3)}"
                f"<h4>Diebold-Mariano tests</h4>{_table(dm, 3)}"
                f"<h4>Mortgage losses, C$m: champion vs panel with payment shock</h4>{_table(scen_pan)}"))

    # 4. Loss distribution and reverse stress ------------------------------------------------------
    log(f"[ext] loss distribution ({n_draws} draws) and reverse stress test")
    s = ctx.res["summary"]
    point = (s[s.horizon_q == ctx.H].groupby("scenario").cum_loss.sum() / 1e6).to_dict()
    draws, bands = uncertainty.simulate(ctx, n=n_draws)
    dist = uncertainty.distribution_table(draws, point)
    rs = uncertainty.reverse_stress(ctx)
    grid = rs.pivot(index="unemployment_rise_pp", columns="house_price_fall_pct", values="loss_C$m")
    draws.to_csv(out / "loss_draws.csv", index=False)
    bands.to_csv(out / "proxy_fan_chart.csv", index=False)
    dist.to_csv(out / "loss_distribution.csv", index=False)
    rs.to_csv(out / "reverse_stress_grid.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for sname, g in draws.groupby("scenario", sort=False):
        ax[0].hist(g["loss_C$m"], bins=40, alpha=0.6, label=sname)
    ax[0].set_title("13-quarter loss, C$m: parameter + residual uncertainty")
    ax[0].legend()
    cs = ax[1].contourf(grid.columns, grid.index, grid.values, levels=12, cmap="Reds")
    ax[1].contour(grid.columns, grid.index, grid.values, levels=[point.get("severe", 0), 2000], colors="k",
                  linewidths=0.8)
    fig.colorbar(cs, ax=ax[1], label="C$m")
    ax[1].set_xlabel("house-price fall, %")
    ax[1].set_ylabel("unemployment rise, pp")
    ax[1].set_title("Reverse stress: lines at severe loss and C$2bn")
    sec.append(("Loss distribution and reverse stress test",
                "Coefficients drawn from their sampling distribution, quarterly credit shocks from the residual "
                "variance (common across segments). The reverse-stress grid adds an unemployment hump and a "
                "house-price fall to the baseline; losses are driven far more by unemployment than by house prices "
                "because only the uninsured mortgages carry house-price risk.",
                f'<img src="data:image/png;base64,{_png(fig)}"><h4>Percentiles, C$m</h4>{_table(dist, 0)}'
                f"<h4>Reverse-stress grid, C$m (rows: unemployment rise pp; columns: house-price fall %)</h4>"
                f"{grid.round(0).reset_index().to_html(index=False, border=0, classes='t')}"))

    # 5. BVAR scenarios ----------------------------------------------------------------------------
    log(f"[ext] BVAR scenarios ({n_bvar} paths)")
    m = bvar.fit(ctx.macro)
    sims = bvar.simulate(m, n=n_bvar)
    bsc, btab = bvar.scenarios(ctx, sims)
    bl = bvar.losses(ctx, bsc)
    tot = bl.groupby("scenario", sort=False)["loss_C$m"].sum().rename("total_C$m").reset_index()
    btab = btab.merge(tot, on="scenario")
    w = {k: v for k, v in zip(btab.scenario, btab.probability) if k != "tail"}
    btab.to_csv(out / "bvar_scenarios.csv", index=False)
    bl.to_csv(out / "bvar_losses.csv", index=False)
    for k, v in bsc.items():
        v.to_csv(out / f"bvar_path_{k}.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, v in bsc.items():
        ax[0].plot(v.index, v.LBR_CA, label=f"BVAR {k}")
        ax[1].plot(v.index, v.HPI_CA, label=f"BVAR {k}")
    for k in ("baseline", "severe"):
        ax[0].plot(ctx.scen[k].index, ctx.scen[k].LBR_CA, "k--" if k == "severe" else "k:", label=f"config {k}")
        ax[1].plot(ctx.scen[k].index, ctx.scen[k].HPI_CA, "k--" if k == "severe" else "k:")
    ax[0].set_title("Unemployment rate, %")
    ax[1].set_title("House prices (index)")
    ax[0].legend(fontsize=7)
    sec.append(("Model-based scenarios (Bayesian VAR)",
                f"Scenarios are the median paths within severity bands of {n_bvar:,} simulated paths; probabilities are the "
                f"band widths ({', '.join(f'{k} {v:.0%}' for k, v in w.items())}). The VAR's severe band is much milder "
                "than the configured severe scenario: Gaussian shocks rarely produce a 2008- or 2020-sized jump, so "
                "the judgemental severe scenario sits well beyond the VAR's 1-in-40 tail. The VAR's house prices also "
                "drift up at their 1994-2026 average growth, faster than the configured baseline. Use the BVAR to "
                "check the baseline and the weights, not to replace the severe narrative.",
                f'<img src="data:image/png;base64,{_png(fig)}">{_table(btab, 2)}'))

    # 7. Full chain vs reported PCL ----------------------------------------------------------------
    log("[ext] full-chain backtest vs reported PCL")
    hb = history.backtest(ctx)
    st = history.fit_stats(hb)
    hb.to_csv(out / "pcl_backtest.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = range(len(hb))
    ax.bar([i - 0.27 for i in x], hb.reported_PCL, 0.27, label="reported PCL")
    ax.bar(x, hb.model_macro, 0.27, label="model, actual macro")
    ax.bar([i + 0.27 for i in x], hb.model_proxy, 0.27, label="model, actual proxy")
    ax.set_xticks(list(x), hb.year)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title("PCL excluding cards, C$m")
    ax.legend(fontsize=8)
    sec.append(("Full-chain backtest against EQB's reported PCL",
                f"Over FY2020-FY2025 the model gives C${st['total_model_macro']:.0f}m against C${st['total_reported']:.0f}m "
                f"reported (correlation {st['corr_macro']:.2f}). It is close in FY2022-23 and, given the actual proxy "
                "path, catches the 2021 release, but explains less than half of FY2024-25: EQB's own losses rose faster than "
                "the industry proxies the PDs are scaled to, and the proxy-scaling calibration cannot see that. "
                "This is the case for bank-specific default data (loan-level PD) or a management overlay.",
                f'<img src="data:image/png;base64,{_png(fig)}">{_table(hb)}'))

    # page -----------------------------------------------------------------------------------------
    body = "".join(f"<h2>{html.escape(t)}</h2><p>{html.escape(x)}</p>{b}" for t, x, b in sec)
    page = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Model extensions — {html.escape(Path(config).stem)}"
            "</title><style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;"
            "color:#222}img{max-width:100%}table.t{border-collapse:collapse;font-size:12px;margin:.5em 0 1.5em}"
            "table.t td,table.t th{padding:3px 8px;border-bottom:1px solid #ddd;text-align:right}"
            "h2{border-top:2px solid #2a6f97;padding-top:.6em;margin-top:2em}p{line-height:1.5}</style></head><body>"
            f"<h1>Model extensions</h1><p>Config: {html.escape(config)}. Tables are also written as CSV next to this "
            f"page. Data sources: data/canada/SOURCES.md.</p>{body}</body></html>")
    f = out / "extensions.html"
    f.write_text(page, encoding="utf-8")
    log(f"[ext] wrote {f}")
    return f
