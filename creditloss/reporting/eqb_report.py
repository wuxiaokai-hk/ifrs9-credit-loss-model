"""Public-data report (HTML) for the EQB Inc. run: outputs/eqb_public/eqb_public_report.html."""
from __future__ import annotations

import base64
import html
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from creditloss import PROJECT_ROOT as ROOT  # noqa: E402
from creditloss.reporting import report as R  # noqa: E402

OUT = ROOT / "outputs/eqb_public"
CFG = yaml.safe_load(open(ROOT / "config" / "eqb_public.yaml"))
SEG = {"mortgage_uninsured": "Uninsured mortgages", "mortgage_insured": "Insured mortgages",
       "credit_cards": "Credit cards", "commercial": "Commercial + equipment"}
SC = ["upside", "baseline", "downside", "severe"]
REPORTED_ACL = {"mortgage_uninsured": 63.183, "mortgage_insured": 1.235, "credit_cards": 239.838,
                "commercial": 95.597}   # EQB Stage 1+2 ACL net of cash reserves, Jul 31 2026, C$m


def img(name, alt):
    p = OUT / name
    if not p.exists():
        return ""
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def table(df, fmt="{:.2f}", index=True):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    return d.to_html(index=index, border=0, classes="t", escape=True)


def chart_proxies(r):
    """Proxy history and its projection (bank PD divided by the calibration scale)."""
    R._style()
    port, p = r["portfolio"], r["paths"]
    segs = ["mortgage_uninsured", "credit_cards", "commercial"]
    titles = {"mortgage_uninsured": "Ontario mortgages 90+ days in arrears, % (CBA)",
              "credit_cards": "US credit-card charge-off rate, % (FRED)",
              "commercial": "US business-loan charge-off rate, % (FRED)"}
    fig, axes = plt.subplots(len(segs), 1, figsize=(8, 7.2), sharex=True)
    for ax, seg in zip(axes, segs):
        h = port[seg]["default_rate"]
        h = h[h.index >= "1995-01-01"]
        ax.plot(h.index, h.values, color=R.INK2, lw=1.5)
        last = float(h.dropna().iloc[-1])
        for s in SC:
            g = p[(p.segment == seg) & (p.scenario == s)]
            proxy = 100 * g.pd_annual / (r["jumpoff_pd"][seg] / last)
            ax.plot(pd.to_datetime(g.date), proxy, color=R.SCEN_COL[s], lw=2)
            ax.annotate(s, (pd.to_datetime(g.date.iloc[-1]), proxy.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=R.INK2, va="center")
        ax.axvline(h.index[-1], color=R.GRID, lw=1)
        ax.set_title(titles[seg], loc="left")
    fig.text(0.01, -0.01, "Grey: actual history. Coloured: champion satellite under each scenario (2026Q3 onward).",
             color=R.INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig0_proxies.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def chart_macro(r):
    R._style()
    m, sc = r["macro"], r["scenarios"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    spec = [("LBR_CA", "Unemployment rate, Canada, %", None),
            ("HPI_CA", "House prices, Canada (BIS, 2010=100)", None),
            ("R3M", "3-month rate, %", None)]
    for ax, (c, t, _) in zip(axes, spec):
        h = m[c][m.index >= "2004-01-01"]
        ax.plot(h.index, h.values, color=R.INK2, lw=1.5)
        for s in SC:
            ax.plot(sc[s].index[:13], sc[s][c].iloc[:13], color=R.SCEN_COL[s], lw=2, label=s)
        ax.set_title(t, loc="left", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig00_macro.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: str | None = None) -> Path:
    global OUT
    OUT = Path(out_dir) if out_dir else OUT
    r = pickle.loads((OUT / "run.pkl").read_bytes())
    xl = pd.ExcelFile(OUT / "credit_loss_results.xlsx")
    chart_proxies(r)
    chart_macro(r)
    H = CFG["horizon_quarters"]
    s = r["summary"]
    s = s[s.horizon_q == H]
    loss_rate = s.pivot(index="segment", columns="scenario", values="loss_rate_pct")[SC].rename(index=SEG)
    loss_amt = (s.pivot(index="segment", columns="scenario", values="cum_loss")[SC] / 1e6).rename(index=SEG)
    tot = s.groupby("scenario")[["cum_loss", "cum_pcl", "avg_ead"]].sum().loc[SC]
    tot_tbl = pd.DataFrame({"Credit losses, C$m": tot.cum_loss / 1e6, "PCL, C$m": tot.cum_pcl / 1e6,
                            "Losses, % of avg exposure": 100 * tot.cum_loss / tot.avg_ead}).T
    loss_amt.loc["Total"] = loss_amt.sum()
    pk = s.pivot(index="segment", columns="scenario", values="peak_pd_pct")[SC].rename(index=SEG)
    jpd = pd.Series(r["jumpoff_pd"]).rename(index=SEG)
    pk.insert(0, "jump-off", jpd)

    # reconciliation
    pw = r["pw_ecl"]
    rec = pd.DataFrame({"Model, baseline only": pw["baseline"] / 1e6, "Model, probability-weighted": pw["prob_weighted"] / 1e6,
                        "EQB reported (Stage 1+2)": pd.Series(REPORTED_ACL)}).rename(index=SEG)
    rec["Model / reported"] = rec["Model, probability-weighted"] / rec["EQB reported (Stage 1+2)"]
    rec_tot = rec.drop(index=[SEG["credit_cards"]])[["Model, probability-weighted", "EQB reported (Stage 1+2)"]].sum()
    gap_pct = 100 * (rec_tot.iloc[0] / rec_tot.iloc[1] - 1)

    # IFRS 9 tables
    d = r["ifrs9"]
    pws, g0, cov = d["pw_by_stage"], d["gca_jumpoff"], d["coverage"]
    i9 = pd.DataFrame({"Gross loans, Stage 1": g0.gca_s1 / 1e6, "Stage 2": g0.gca_s2 / 1e6, "Stage 3": g0.gca_s3 / 1e6,
                       "ECL, Stage 1": pws.ecl_stage1 / 1e6, "Stage 2 ": pws.ecl_stage2 / 1e6, "Stage 3 ": pws.ecl_stage3 / 1e6,
                       "Total ACL": pws.allowance_total / 1e6}).rename(index=SEG)
    i9.loc["Total"] = i9.sum()
    covt = cov.rename(index=SEG)
    sens = d["sensitivity"][SC + ["probability_weighted"]] / 1e6
    sens.loc["Total"] = sens.sum()
    sens = sens.rename(index=SEG)
    stg = (d["staging"][["all_stage1", "reported_staging", "all_stage2"]] / 1e6).rename(
        columns={"all_stage1": "All performing in Stage 1", "reported_staging": "As staged",
                 "all_stage2": "All performing in Stage 2"})
    stg.loc["Total"] = stg.sum()
    stg = stg.rename(index=SEG)
    rf = d["rollforward"]
    rf_sev = rf[rf.scenario == "severe"].groupby("line", sort=False)[["Stage 1", "Stage 2", "Stage 3", "Total"]].sum() / 1e6
    rf_base = rf[rf.scenario == "baseline"].groupby("line", sort=False)[["Total"]].sum() / 1e6
    rf_tbl = rf_sev.rename(columns={"Total": "Severe, total"}).assign(**{"Baseline, total": rf_base["Total"]})
    wo = s.groupby("scenario")[["cum_net_writeoffs", "closing_acl"]].sum().loc[SC] / 1e6

    # champions and models
    mt = []
    for seg, sel in r["sels"].items():
        if seg == "mortgage_insured":
            continue  # same proxy and satellite as uninsured
        for nm, sp in sel["specs"].items():
            mt.append({"Segment": SEG[seg] + (" (and insured)" if seg == "mortgage_uninsured" else ""), "Model": nm,
                       "Drivers": ", ".join(sp.drivers) or "-", "AR(1)": sp.params.get("AR1", np.nan),
                       "Coefficients (t)": "; ".join(f"{sp.params[d]:+.3f} ({sp.tstats[d]:.1f})" for d in sp.drivers) or "-",
                       "Signs OK": "yes" if sp.sign_ok else "NO",
                       "Champion": "★" if nm == r["champions"][seg] else ""})
    mt = pd.DataFrame(mt).set_index(["Segment", "Model"])
    m = xl.parse("oos_metrics")
    m = m[m.h.astype(str).str.startswith("1-")]
    oos = m.pivot(index="model", columns="segment", values="rmse_pp").drop(columns="mortgage_insured").rename(columns=SEG)
    st = xl.parse("oos_stress")
    cap = st.pivot_table(index="model", columns=["segment", "window"], values="cum_capture_pct")
    cap = cap.drop(columns="mortgage_insured", level=0)
    cap.columns = [f"{SEG[a]}, {b[:4]}–{b[-10:-6]}" for a, b in cap.columns]

    att = r["attribution"]
    att_tot = att[~att.component.str.startswith("  ")].groupby("component", sort=False).amount.sum() / 1e6
    gap = att_tot["Severe loss"] - att_tot["Baseline loss"]
    att_seg = (att.pivot_table(index="component", columns="segment", values="amount", aggfunc="sum", sort=False) / 1e6).rename(columns=SEG)
    sub = [c for c in att_seg.index if c.startswith("  ")]
    att_seg = att_seg.reindex(["Baseline loss", "Higher defaults (PD)"] + sub + ["Weaker collateral (LGD)", "Severe loss"])

    snap = r["snapshot"].copy()
    snap["Balance, C$m"] = snap.balance / 1e6
    book = snap.groupby("segment").agg(**{"Balance, C$m": ("Balance, C$m", "sum")}).rename(index=SEG)
    book["Jump-off PD, %"] = jpd
    book["LGD rule"] = pd.Series({k: CFG["segments"][k]["lgd_model"] for k in SEG}).rename(index=SEG)
    book["Proxy for PD dynamics"] = pd.Series({k: CFG["segments"][k]["proxy"] for k in SEG}).rename(index=SEG)

    sc_tbl = []
    mh = r["macro"]
    for k in SC:
        d = r["scenarios"][k].iloc[:H]
        sc_tbl.append({"Scenario": k, "Peak unemployment, %": d.LBR_CA.max(),
                       "House prices, trough vs jump-off, %": 100 * (d.HPI_CA.min() / mh.HPI_CA.iloc[-1] - 1),
                       "3-month rate, trough, %": d.R3M.min(), "10-year yield, trough, %": d.R10Y.min(),
                       "Real GDP, trough vs jump-off, %": 100 * (d.GDPR_CA.min() / mh.GDPR_CA.iloc[-1] - 1)})
    sc_tbl = pd.DataFrame(sc_tbl).set_index("Scenario")
    sc_tbl.loc["jump-off (2026Q2)"] = [mh.LBR_CA.iloc[-1], 0, mh.R3M.iloc[-1], mh.R10Y.iloc[-1], 0]

    sev, base = tot.loc["severe"], tot.loc["baseline"]
    css = """
    :root{--ink:#1b1b1a;--ink2:#52514e;--rule:#e6e5e0;--bg:#fcfcfb;--accent:#2a78d6;--note:#f3f1ea}
    @media (prefers-color-scheme: dark){:root{--ink:#f0efec;--ink2:#c3c2b7;--rule:#3a3936;--bg:#1a1a19;--note:#262624}
      figure img{filter:brightness(.92)}}
    body{font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);margin:0}
    main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
    h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px;padding-top:12px;border-top:1px solid var(--rule)}
    h3{font-size:16px;margin:24px 0 6px}.sub{color:var(--ink2);margin:0 0 24px}
    table.t{border-collapse:collapse;font-size:13px;margin:8px 0 16px;display:block;overflow-x:auto;max-width:100%}
    table.t th,table.t td{padding:5px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
    table.t th:first-child,table.t td:first-child{text-align:left}table.t thead th{color:var(--ink2);font-weight:600}
    figure{margin:16px 0}figure img{max-width:100%;background:#fcfcfb;border-radius:6px}
    figcaption{font-size:12px;color:var(--ink2)}code,.eq{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}
    .eq{background:var(--note);padding:10px 12px;border-radius:6px;white-space:pre-wrap;display:block;margin:8px 0}
    .note{background:var(--note);padding:10px 14px;border-radius:6px;font-size:14px}
    ul{padding-left:20px}li{margin:3px 0}a{color:var(--accent)}
    """
    body = f"""
<h1>Scenario credit losses on real data: EQB Inc.</h1>
<p class="sub">creditloss · jump-off 2026Q2 (EQB book at July 31, 2026) · horizon {H} quarters · public data only</p>
<p class="note">This is an outside-in exercise from public disclosures and public macro data, not EQB's own model.
Where EQB does not disclose something (LTV distribution, card limits, amortization), the assumption is stated.
Scenarios are author-built; Moody's paths are not public.</p>

<h2>1. Results</h2>
<p>Over {H} quarters, modelled credit losses on EQB's $49.2bn on-balance-sheet book are
<strong>C${base.cum_loss/1e6:,.0f}m in the baseline ({100*base.cum_loss/base.avg_ead:.2f}% of exposure)</strong> and
<strong>C${sev.cum_loss/1e6:,.0f}m in the severe scenario ({100*sev.cum_loss/sev.avg_ead:.2f}%)</strong>.
Severe-scenario IFRS 9 provisions (write-offs plus the change in the allowance, opening from the probability-weighted
allowance) are C${sev.cum_pcl/1e6:,.0f}m; section 7 has the stage-by-stage view.
The newly acquired PC Financial card book is 9% of loans but carries
{100*loss_amt.loc['Credit cards','baseline']/loss_amt.loc['Total','baseline']:.0f}% of baseline losses;
uninsured mortgages, at a 69% average LTV, move most in relative terms, from C${loss_amt.loc['Uninsured mortgages','baseline']:.0f}m to
C${loss_amt.loc['Uninsured mortgages','severe']:.0f}m, as a 30% house-price fall pushes higher-LTV loans into collateral shortfall.</p>
{table(tot_tbl, "{:,.1f}")}
<h3>Cumulative credit losses by segment, C$m</h3>
{table(loss_amt, "{:,.1f}")}
<h3>Same, % of average exposure</h3>
{table(loss_rate)}
{img('fig2_loss_rates.png', 'Cumulative credit losses by segment and scenario, % of average exposure')}

<h2>2. Data</h2>
<table class="t"><thead><tr><th>Item</th><th>Series</th><th>Source</th><th>Coverage</th><th>Treatment</th></tr></thead><tbody>
<tr><td>Mortgage default dynamics</td><td>Mortgages 90+ days in arrears, Ontario, % of count</td><td>Canadian Bankers Association (major banks)</td><td>1995Q1–2026Q2 (May)</td><td>Quarter-end months; all 126 rows checked against count ratio</td></tr>
<tr><td>Card default dynamics</td><td>Credit-card charge-off rate, all US commercial banks</td><td>FRED CORCCACBS</td><td>1990Q1–2026Q2</td><td>Proxy: no long Canadian card loss series is public</td></tr>
<tr><td>Commercial default dynamics</td><td>Business-loan charge-off rate, all US commercial banks</td><td>FRED CORBLACBS</td><td>1990Q1–2026Q2</td><td>Proxy, as above</td></tr>
<tr><td>Unemployment, Canada</td><td>LFS rate, 15+, quarterly</td><td>FRED LRHUTTTTCAQ156N (StatCan via OECD)</td><td>1994Q1–2026Q2</td><td>Seasonally adjusted; 2020Q2–2021Q3 interpolated</td></tr>
<tr><td>Unemployment, US</td><td>Rate, 15–74, quarterly</td><td>FRED LRUN74TTUSQ156N</td><td>1990Q1–2025Q3</td><td>Same; last value carried to 2026Q2</td></tr>
<tr><td>House prices</td><td>Residential property prices, Canada</td><td>BIS via FRED QCAN628BIS</td><td>1993Q1–2026Q1</td><td>2026Q1 carried to jump-off; drives LTV and LGD</td></tr>
<tr><td>Rates</td><td>3-month interbank; 10-year GoC yield</td><td>FRED IR3TIB01CAQ156N, IRLTLT01CAQ156N</td><td>1993Q1–2026Q3</td><td>Quarterly averages</td></tr>
<tr><td>Activity</td><td>Real GDP growth, q/q</td><td>FRED NAEXKP01CAQ657S</td><td>1993Q1–2026Q2</td><td>Cumulated to an index; 2020–21 interpolated</td></tr>
<tr><td>Jump-off book</td><td>Loans by product, ACL by stage and type, impaired loans, LTV, Stage 2 shares</td><td>EQB Q3 2026 Report to Shareholders; Q2 2026 MD&A</td><td>Jul 31, 2026 (Apr 30 for Stage 2 and impaired split)</td><td>See section 3</td></tr>
</tbody></table>
<p><strong>COVID bridge.</strong> Unemployment hit 13% in 2020Q2 while arrears and charge-offs <em>fell</em>, because of deferrals
and income support. Left in, that episode drags the unemployment elasticity toward zero, so the macro series are
interpolated through 2020Q2–2021Q3, the usual treatment in bank IFRS 9 models.</p>
<p><strong>Data gaps.</strong> StatCan Ontario unemployment (table 14-10-0287), the BoC
5-year mortgage rate and CREA/Teranet Ontario house prices would replace the national proxies. OSB Ontario consumer
and business insolvencies would replace the US charge-off proxies. This workspace was blocked from those sites.</p>
{img('fig00_macro.png', 'Macro history and the four scenarios (first 13 quarters)')}
<h3>Scenarios</h3>
{table(sc_tbl, "{:.1f}")}
<p>The severe path matches the size of the Bank of England 2025 test (house prices −28%) and Ontario's early-1990s
downturn (unemployment near 11%). Card and commercial satellites, which were estimated on US data, are driven by the
Canadian unemployment change: the US jump-off rate plus the Canadian scenario's deviation.</p>

<h2>3. Jump-off book and calibration</h2>
{table(book)}
<span class="eq">PD_bank,t = PD_bank,0 × proxy_t / proxy_0        (proxy_t from the champion satellite under the scenario)</span>
<ul>
<li><strong>Mortgages and commercial.</strong> PD<sub>0</sub> comes from EQB's impaired data. Q3 impaired formations were about
$205m a quarter (Q2 formations less the disclosed $39m decline), or roughly $0.8bn a year, on $35.7bn of uninsured residential and commercial
loans. Split by the Apr 30 impaired mix (personal 1.9% of uninsured personal loans, commercial 5.3%), that gives
1.7% for uninsured mortgages and 3.4% for commercial. Insured: 1.3%, from insured impaired loans of $147m.</li>
<li><strong>Credit cards.</strong> EQB gives no card default data one month after the acquisition. PD<sub>0</sub> is solved so the
model's probability-weighted Stage 1+2 allowance equals the $239.8m EQB reports, giving {r['jumpoff_pd']['credit_cards']:.2f}% a year.</li>
<li><strong>LTV.</strong> The disclosed 69% average is spread over four buckets (45/60/71/78%, assumed weights 10/20/35/35%).
Decumulation loans are set at 35%.</li>
<li><strong>Stage 2.</strong> Calibrated to EQB's Apr 30 shares (personal 20.4%, commercial 11.7%); cards 12% assumed.</li>
</ul>
<h3>Check: model allowance at jump-off vs EQB's reported allowance, C$m</h3>
{table(rec)}
<p>Only the card line was calibrated. For the three other books together, the modelled probability-weighted allowance is
{gap_pct:+.0f}% against what EQB reports. The model discounts expected shortfalls at the loan rate and measures lifetime ECL only
to the end of the contractual term (renewal for mortgages, assumed 2.5–3 years), as IFRS 9 requires, and it carries no
management overlay. A longer behavioural life, a lower cure rate or an overlay would each close the gap; EQB's own
inputs on these points are not public, so the gap is left visible rather than calibrated away.</p>
<p>Stage 3 is a separate check. EQB's total allowance at July 31, 2026 was $485.4m, which implies about
${485.4 - sum(REPORTED_ACL.values()):,.0f}m on impaired loans; the model's Stage 3 allowance is
${r['ifrs9']['pw_by_stage']['ecl_stage3'].sum()/1e6:,.0f}m, and its total ${r['ifrs9']['pw_by_stage']['allowance_total'].sum()/1e6:,.0f}m.
EQB also notes impaired loans rose because of "extended workout timelines", which is why the workout period is set to six quarters
for mortgages and commercial loans.</p>

<h2>4. PD satellites</h2>
<p>Same engine as the methodology build. The candidates are each series × transform × lag 0–4. A sign-constrained search
over 1–3 drivers with an AR(1) term is run alongside literature specifications and BMA. The champion must carry the
literature signs and capture 80–200% of crisis defaults out of sample. Among those, the lowest out-of-sample error wins.</p>
{table(mt, "{:.3f}")}
{img('fig0_proxies.png', 'Proxy histories and champion projections by scenario')}
<p>Three literature equations come out with the <em>wrong</em> sign on unemployment on real data. They are the ECB/EBA-style
mortgage equation and the unemployment-level card and commercial equations: with a near-unit-root lag, the level of unemployment
adds nothing and its sign flips. They are kept in the model-risk range but cannot be champion. The mortgage champion is the TDSA-style
equation (change in unemployment and house-price growth). Card and commercial champions both reduce to the change in
unemployment.</p>

<h2>5. Validation</h2>
<h3>Out-of-sample RMSE, pp, 1–12 quarters ahead, re-selected at every origin</h3>
{table(oos, "{:.3f}")}
<h3>Share of actual defaults captured in each stress window, forecasting from the last origin before it, %</h3>
{table(cap, "{:.0f}")}
<p>Mortgages: the champion captured 124% of the 2008–10 arrears rise and 65% of the 2023–26 rise. It underpredicts the
recent episode, which was driven by payment shock at renewal, a channel unemployment and house prices only partly carry.
Cards: 88% and 84%. Commercial: no model meets the 80–200% band. The change-in-unemployment model over-shoots the GFC
3.4× from a 2007 origin, while the alternatives under-shoot badly. The over-shooter is kept as champion, and the commercial
severe loss should be read as conservative.</p>

<h2>6. What drives the severe scenario</h2>
<p>Severe minus baseline is C${gap:,.0f}m: higher default rates account for C${att_tot['Higher defaults (PD)']:,.0f}m and weaker collateral
for C${att_tot['Weaker collateral (LGD)']:,.0f}m (Shapley split; commercial collateral effects sit inside the PD leg via the Frye-Jacobs LGD).</p>
{img('fig4_attribution.png', 'Severe-scenario loss bridge')}
{table(att_seg, "{:,.1f}")}
<h3>Peak annual PD by scenario, %</h3>
{table(pk)}
{img('fig3_lgd.png', 'Loss given default by scenario')}
{img('fig6_provisions.png', 'IFRS 9 provisions and allowance (Stages 1-3)')}
<h3>Model risk: severe loss under every PD model</h3>
{img('fig5_model_range.png', 'Severe loss under each PD model (red = champion)')}

<h2>7. IFRS 9 view</h2>
<p>The stage engine tracks gross loans in Stages 1–3 every quarter. Loans move to Stage 2 when their PD passes {CFG['ifrs9']['sicr_multiple']:.0f}×
the origination PD (calibrated to EQB's reported Stage 2 shares), default into Stage 3, and leave Stage 3 after a workout period, either cured
or written off. Stage 1 carries 12-month ECL, Stages 2 and 3 lifetime ECL, discounted at the loan rate. Scenario PDs are used for
{CFG['ifrs9']['reasonable_supportable_quarters']} quarters, then revert to the long-run average over {CFG['ifrs9']['reversion_quarters']}.
Every scenario opens from the probability-weighted allowance, so the first quarter's provision includes the move to that scenario's own
forward-looking ECL.</p>
<h3>Allowance at jump-off by stage, probability-weighted, C$m</h3>
{table(i9, "{:,.1f}")}
<h3>Coverage ratio (allowance / gross loans), %</h3>
{table(covt, "{:.2f}")}
<h3>Sensitivity to forward-looking information: total ACL with 100% weight on each scenario, C$m</h3>
{table(sens, "{:,.1f}")}
<h3>Sensitivity to staging: total ACL if every performing loan were in Stage 1 or Stage 2, C$m</h3>
{table(stg, "{:,.1f}")}
<h3>Allowance roll-forward, jump-off to quarter {H}, all segments, C$m (IFRS 7.35H format)</h3>
{table(rf_tbl, "{:,.1f}")}
<p>In the severe scenario the allowance builds immediately (the day-one line), loans migrate into Stage 2 and Stage 3, and
write-offs then draw it down as defaults are resolved. In the baseline, the allowance is <em>released</em>: the opening allowance
carries the 30% downside and 20% severe weights, which fall away as the baseline unfolds.</p>
<h3>Write-offs and closing allowance, C$m</h3>
{table(wo.rename(columns={"cum_net_writeoffs": f"Net write-offs over {H}q", "closing_acl": f"ACL at quarter {H}"}).T, "{:,.1f}")}
{img('fig8_stages.png', 'Stage mix and ECL by stage, baseline and severe')}

<h2>8. Limitations</h2>
<ul>
<li>Mortgage dynamics come from big-bank Ontario arrears. EQB's alternative-lending borrowers may be more sensitive than
the scaling assumes, since the scale is fixed at jump-off.</li>
<li>Card and commercial sensitivities are US-estimated. Canadian card losses rose less than US losses in 2008–10, so the card
stress is likely conservative; commercial depends on construction and CRE exposure that US C&amp;I charge-offs only roughly represent.</li>
<li>The structural mortgage LGD uses a national house-price index. GTA condo prices have fallen more than the national index
since 2022, which matters for EQB.</li>
<li>Static balance sheet; no management actions; second-round effects excluded.</li>
<li>PC Financial was acquired on July 1, 2026, so the card book's own loss history is not yet in EQB's disclosures.</li>
</ul>

<h2>Sources</h2>
<ul><li>EQB Inc., <a href="https://www.placera.se/pressmeddelanden/eqb-eqb-reports-third-quarter-2026-results-including-one-month-of-results-from-pc-financial-and-announces-dividend-increase-20260826">Q3 2026 results release</a> (total allowance $485.4m; impaired loans and workout timelines).</li>
<li>IFRS 9 <em>Financial Instruments</em>, section 5.5 and Appendix B5.5; IFRS 7 paragraphs 35A–35N.</li></ul>
<ul>
<li>Canadian Bankers Association, <a href="https://votreargent.cba.ca/Assets/CanadianBankersAssociation/Documents/Articles/Statistics/stat-mortgages-arrears-may-2026-en.pdf">Number of residential mortgages in arrears, May 2026</a>.</li>
<li>EQB Inc., <a href="https://eqb.investorroom.com/download/EQB+Report+to+Shareholders+Q3+2026+SEDAR.pdf">Report to Shareholders Q3 2026</a>; <a href="https://newsfile.futunn.com/public/NN-PersistNoticeAttachment/7781/20260528/SEDAR_PLUS/CSA_SEDAR_PLUS_NOTICE_RECORD_ID_2418769.pdf">Q2 2026 MD&amp;A</a>; <a href="https://eqb.investorroom.com/2026-08-26-EQB-reports-third-quarter-2026-results-including-one-month-of-results-from-PC-Financial-and-announces-dividend-increase">Q3 2026 press release</a>.</li>
<li>FRED: <a href="https://fred.stlouisfed.org/data/LRHUTTTTCAQ156N">Canada unemployment</a>, <a href="https://fred.stlouisfed.org/data/QCAN628BIS">BIS residential property prices, Canada</a>, <a href="https://fred.stlouisfed.org/data/IR3TIB01CAQ156N">3-month interbank</a>, <a href="https://fred.stlouisfed.org/data/IRLTLT01CAQ156N">10-year yield</a>, <a href="https://fred.stlouisfed.org/data/NAEXKP01CAQ657S">real GDP</a>, <a href="https://fred.stlouisfed.org/data/CORCCACBS">US card charge-offs</a>, <a href="https://fred.stlouisfed.org/data/CORBLACBS">US business-loan charge-offs</a>, <a href="https://fred.stlouisfed.org/data/LRUN74TTUSQ156N">US unemployment</a>.</li>
<li>Methodology sources as in the methodology report (BoC SAP 2026-12, TR-128; Fed 2025 methodology; EBA 2025 note; ECB WP 1845; BoE 2025 test).</li>
</ul>
"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EQB Scenario Losses</title><style>{css}</style></head><body><main>{body}</main></body></html>"""
    p = OUT / "eqb_public_report.html"
    p.write_text(page, encoding="utf-8")
    print(f"Wrote {p}")
    return p

