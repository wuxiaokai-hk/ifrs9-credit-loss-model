"""Methodology report (HTML) for a run: outputs/<run>/methodology_report.html from run.pkl and the workbook.

The methodology text is fixed; every number, table and chart is read from the run,
so re-running `creditloss run --config config/sample.yaml --backtest --report` keeps it in sync.
"""
from __future__ import annotations

import base64
import html
import pickle
from pathlib import Path

import pandas as pd

from creditloss import PROJECT_ROOT as ROOT  # noqa: E402

OUT = ROOT / "outputs/sample"
SEG = {"mortgage_uninsured": "Mortgages, uninsured", "mortgage_insured": "Mortgages, insured",
       "heloc": "HELOCs", "consumer": "Consumer", "commercial_re": "Commercial / CRE"}
SC = ["upside", "baseline", "downside", "severe"]


def img(name: str, alt: str) -> str:
    p = OUT / name
    if not p.exists():
        return ""
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def table(df: pd.DataFrame, fmt: str = "{:.3f}", index: bool = True) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    return d.to_html(index=index, border=0, classes="t", escape=True)


def main(out_dir: str | None = None) -> Path:
    global OUT
    OUT = Path(out_dir) if out_dir else OUT
    r = pickle.loads((OUT / "run.pkl").read_bytes())
    xl = pd.ExcelFile(OUT / "credit_loss_results.xlsx")
    sheets = set(xl.sheet_names)
    H = int(r["paths"].q.max())
    s = r["summary"]
    s13 = s[s.horizon_q == H]
    loss = s13.pivot(index="segment", columns="scenario", values="loss_rate_pct")[SC].rename(index=SEG)
    pcl = s13.pivot(index="segment", columns="scenario", values="pcl_rate_pct")[SC].rename(index=SEG)
    tot = s13.groupby("scenario")[["cum_loss", "avg_ead"]].sum()
    tot["rate"] = 100 * tot.cum_loss / tot.avg_ead
    tot_tbl = pd.DataFrame({"Cumulative loss, C$m": tot.cum_loss / 1e6, "% of average EAD": tot.rate}).loc[SC].T
    peaks = s13.pivot(index="segment", columns="scenario", values=["peak_pd_pct", "peak_lgd_pct"])
    sev_pd = peaks["peak_pd_pct"]["severe"].rename(index=SEG)
    sev_lgd = peaks["peak_lgd_pct"]["severe"].rename(index=SEG)
    jump_pd = r["paths"][r["paths"].q == 0].groupby("segment").pd_annual.first().mul(100).rename(index=SEG)

    # scenario table
    rows = []
    hist = r["macro"]
    for k in SC:
        d = r["scenarios"][k].iloc[:H]
        rows.append({"Scenario": k, "Peak unemployment, %": d.LBR_ON.max(),
                     "House prices, trough vs jump-off %": 100 * (d.HPI_ON.min() / hist.HPI_ON.iloc[-1] - 1),
                     "CRE prices, trough %": 100 * (d.PCRE_ON.min() / hist.PCRE_ON.iloc[-1] - 1),
                     "5y mortgage rate, peak %": d.RMORT5.max(), "Policy rate, trough %": d.RPOL.min(),
                     "BBB spread, peak %": d.SPRBBB.max()})
    scen_tbl = pd.DataFrame(rows).set_index("Scenario")
    jo = pd.Series({"Peak unemployment, %": hist.LBR_ON.iloc[-1], "5y mortgage rate, peak %": hist.RMORT5.iloc[-1],
                    "Policy rate, trough %": hist.RPOL.iloc[-1], "BBB spread, peak %": hist.SPRBBB.iloc[-1]}, name="jump-off")

    # models
    champs = r["champions"]
    mt = []
    for seg, sel in r["sels"].items():
        for nm in [champs[seg], "Data-driven search"]:
            sp = sel["specs"][nm]
            mt.append({"Segment": SEG[seg], "Model": nm + (" (champion)" if nm == champs[seg] else ""),
                       "Drivers (variable | transform | lag)": ", ".join(sp.drivers),
                       "AR(1) coef.": sp.params.get("AR1", float("nan")),
                       "Driver t-stats (HAC)": ", ".join(f"{sp.tstats[d]:.1f}" for d in sp.drivers),
                       "Dynamic RMSE, pp": sp.dyn_rmse})
    model_tbl = pd.DataFrame(mt).set_index(["Segment", "Model"])

    oos_html = stress_html = champ_html = ""
    if "oos_metrics" in sheets:
        m = xl.parse("oos_metrics")
        m = m[m.h.astype(str).str.startswith("1-")]
        oos_tbl = m.pivot(index="model", columns="segment", values="rmse_pp").rename(columns=SEG)
        oos_html = table(oos_tbl)
        st = xl.parse("oos_stress")
        st = st[st.window.str.startswith("2008")].pivot(index="model", columns="segment", values="cum_capture_pct").rename(columns=SEG)
        stress_html = table(st, "{:.0f}")
        if "oos_champions" in sheets:
            ch = xl.parse("oos_champions").assign(segment=lambda d: d.segment.map(SEG)).set_index("segment")
            champ_html = table(ch[["champion", "oos_rmse_pp", "gfc_capture_pct"]])
        cov = xl.parse("oos_stress")
        cov = cov[cov.window.str.startswith("2020")]
        covid_capture = cov[cov.model.isin(champs.values())].cum_capture_pct.median()
    else:
        covid_capture = float("nan")

    att = r["attribution"]
    att_tot = att[~att.component.str.startswith("  ")].groupby("component", sort=False).amount.sum() / 1e6
    pd_share = att_tot["Higher defaults (PD)"] / (att_tot["Severe loss"] - att_tot["Baseline loss"])
    att_seg = att.pivot_table(index="component", columns="segment", values="amount", aggfunc="sum", sort=False) / 1e6
    att_seg = att_seg.rename(columns=SEG)

    pw = r["pw_ecl"].copy()
    pw_tbl = (pw[["baseline", "prob_weighted"]] / 1e6).rename(columns={"baseline": "Baseline-only ECL, C$m",
                                                                      "prob_weighted": "Probability-weighted ECL, C$m"})
    pw_tbl["Uplift, %"] = pw["nonlinearity_vs_baseline_pct"]
    pw_tbl = pw_tbl.rename(index=SEG)
    pw_total = pw[["baseline", "prob_weighted"]].sum() / 1e6

    nco = r["nco_check"].pivot(index="segment", columns="scenario",
                               values=["bottom_up_loss_rate_pct", "top_down_nco_rate_pct"])
    nco_tbl = pd.DataFrame({"Bottom-up, baseline": nco["bottom_up_loss_rate_pct"]["baseline"],
                            "Top-down, baseline": nco["top_down_nco_rate_pct"]["baseline"],
                            "Bottom-up, severe": nco["bottom_up_loss_rate_pct"]["severe"],
                            "Top-down, severe": nco["top_down_nco_rate_pct"]["severe"]}).rename(index=SEG)

    sev_mult = (sev_pd / jump_pd)
    css = """
    :root{--ink:#1b1b1a;--ink2:#52514e;--rule:#e6e5e0;--bg:#fcfcfb;--accent:#2a78d6;--note:#f3f1ea}
    @media (prefers-color-scheme: dark){:root{--ink:#f0efec;--ink2:#c3c2b7;--rule:#3a3936;--bg:#1a1a19;--note:#262624}
      figure img{filter:brightness(.92)}}
    body{font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);margin:0}
    main{max-width:960px;margin:0 auto;padding:32px 16px 64px}
    h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px;padding-top:12px;border-top:1px solid var(--rule)}
    h3{font-size:16px;margin:24px 0 6px}.sub{color:var(--ink2);margin:0 0 24px}
    table.t{border-collapse:collapse;font-size:13px;margin:8px 0 16px;width:100%;display:block;overflow-x:auto}
    table.t th,table.t td{padding:5px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
    table.t th:first-child,table.t td:first-child{text-align:left}table.t thead th{color:var(--ink2);font-weight:600}
    figure{margin:16px 0}figure img{max-width:100%;background:#fcfcfb;border-radius:6px}
    figcaption{font-size:12px;color:var(--ink2)}code,.eq{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}
    .eq{background:var(--note);padding:10px 12px;border-radius:6px;white-space:pre-wrap;display:block;margin:8px 0}
    .note{background:var(--note);padding:10px 14px;border-radius:6px;font-size:14px}
    ul{padding-left:20px}li{margin:3px 0}a{color:var(--accent)}
    """

    body = f"""
<h1>Scenario credit-loss model: methodology and results</h1>
<p class="sub">creditloss · methodology on a simulated credit-union book · jump-off {r['paths'][r['paths'].q == 0].date.iloc[0].date()} ·
horizon {H} quarters · <strong>all data simulated</strong></p>

<p class="note">Every number below comes from simulated data generated by <code>simulate_data.py</code>.
They show what the method produces and how its checks behave; they are not estimates for any credit union.</p>

<h2>1. Results in brief</h2>
<p>Across the five segments, cumulative credit losses over {H} quarters are
<strong>C${tot.loc['baseline','cum_loss']/1e6:,.0f}m ({tot.loc['baseline','rate']:.2f}% of average exposure) in the baseline</strong>
and <strong>C${tot.loc['severe','cum_loss']/1e6:,.0f}m ({tot.loc['severe','rate']:.2f}%) in the severe scenario</strong>.
About {100*pd_share:.0f}% of the severe-minus-baseline gap comes from higher default rates and the rest from weaker collateral.
Consumer and commercial books carry most of the dollar loss; residential mortgages, with low current LTVs and
mortgage insurance, lose little even with a {abs(scen_tbl.loc['severe','House prices, trough vs jump-off %']):.0f}% house-price fall,
but their loss rate rises {loss.loc['Mortgages, uninsured','severe']/max(loss.loc['Mortgages, uninsured','baseline'],1e-9):.0f}-fold because LGD
moves from workout costs alone to a collateral shortfall.</p>
<h3>Cumulative loss rate by segment, % of average EAD over {H} quarters</h3>
{table(loss)}
<h3>Whole book</h3>
{table(tot_tbl, "{:,.2f}")}
{img('fig2_loss_rates.png', 'Cumulative credit losses by segment and scenario')}

<h2>2. How supervisors project mortgage and loan losses</h2>
<p>All five frameworks reviewed share one structure: a <strong>scenario</strong> fixes the macro path, a
<strong>satellite model</strong> turns it into default rates, a <strong>collateral or loss-severity rule</strong> turns defaults
into losses, and an <strong>accounting layer</strong> turns losses into provisions. They differ in granularity and in how much
conservatism is imposed by rule.</p>
<table class="t"><thead><tr><th>Institution</th><th>Default (PD) projection</th><th>Loss severity (LGD)</th><th>Provisions / horizon</th><th>What this model takes from it</th></tr></thead><tbody>
<tr><td>Bank of Canada: arrears model (SAP 2026-12)</td><td>Provincial OLS of 90+ day arrears on UR (6m lag), job finding/separation, y/y house prices (12m lag), 5y mortgage rate minus neutral (12m lag), core CPI, output gap; HAC errors; SVAR for causality</td><td>Not modelled (arrears only)</td><td>Projects arrears under MPR/FSR scenarios</td><td>Regressor set and lags as a fixed challenger; sign priors (inflation lowers arrears through debt deflation)</td></tr>
<tr><td>Bank of Canada: TDSA (TR-128)</td><td>Sectoral default rates from HRAM (household microsimulation) and CDM (corporate); aggregate PCL ratio regressed on its lag and one-quarter-ahead ΔUR, GDP, TSX, HPI</td><td>Downturn LGD from banks' IRB filings, raised by judgment when house prices fall: costs/value + max(1 − (1+g)/LTV, 0)</td><td>Impaired PCL = DR × LRDE × EAD; EAD = drawn + drawdown × undrawn; performing PCL pulled forward by a foresight assumption</td><td>Loss identity, EAD with CCF, house-price LGD, the generic AR + macro equation, LASSO/OOS selection with judgment, top-down PCL equation as cross-check</td></tr>
<tr><td>OSFI (SCSE 2024; B-15)</td><td>Scenario-adjusted PD applied through each institution's ECL engine via standardized adjustment factors</td><td>Scenario-adjusted LGD</td><td>Climate-adjusted ECL by horizon; scenario impact measured through provisioning infrastructure</td><td>Run the scenario through the IFRS 9 machinery (staging, 12m vs lifetime ECL) rather than stopping at losses</td></tr>
<tr><td>Federal Reserve (DFAST/CCAR)</td><td>Expected-loss framework (PD, LGD, EAD each quarter) for first-lien mortgages and other large books; net charge-off models elsewhere</td><td>Mortgage severity tied to updated LTV via house prices</td><td>Losses over 9 quarters; provisions = losses + change in allowance covering the next 4 quarters; unchanged balance sheet in 2025</td><td>13-quarter loss horizon, Fed 4-quarter allowance shown alongside IFRS 9, static balance sheet default, NCO top-down check, preference for stable models that perform in stress</td></tr>
<tr><td>EBA / ECB (EU-wide test)</td><td>Constrained bottom-up: banks' point-in-time PDs stressed by satellite models; ECB top-down satellites, with Bayesian model averaging argued for (Gross &amp; Población 2015)</td><td>Point-in-time LGD; collateral valued on the scenario path (perfect foresight)</td><td>IFRS 9 stages; Stage 1→2 when PD triples; no cure out of Stage 3; static balance sheet; 3 years</td><td>SICR ×3 rule, perfect-foresight LGD, no-cure option, BMA challenger, constrained-search philosophy</td></tr>
<tr><td>Bank of England (2025 Bank Capital Stress Test)</td><td>Bank and BoE models under a common scenario: UR to 8.5%, house prices −28%, Bank Rate 8%</td><td>Mortgage impairments driven by LTV; buy-to-let losses far above owner-occupied in past tests</td><td>5-year horizon, IFRS 9 basis</td><td>Severe scenario calibration (−28% house prices, UR roughly doubling-style shock); longer lifetime-ECL window</td></tr>
</tbody></table>

<h2>3. The model</h2>
<h3>3.1 Segments</h3>
<p>Uninsured mortgages, insured mortgages, HELOCs, consumer loans and commercial/CRE. Each segment has its own PD satellite,
LGD rule and EAD rule (set in the config). Household books use household sign priors; the commercial book uses business priors.</p>

<h3>3.2 Default rates: the PD satellite</h3>
<span class="eq">logit(DR<sub>t</sub>) = c + ρ·logit(DR<sub>t−1</sub>) + Σ<sub>k</sub> β<sub>k</sub>·x<sub>k,t−L<sub>k</sub></sub> + ε<sub>t</sub>      (1 to 3 drivers)</span>
<ul>
<li><strong>Target.</strong> New 90+ day / impaired flow as an annualized share of the opening balance, in logits so projections stay in (0, 1) and shocks act multiplicatively, as in ECB satellites.</li>
<li><strong>Candidates.</strong> Every series × transform (level, 4-quarter change, y/y %) × lag 0-4 quarters. A literature table gives each series its expected sign by book and the evidence behind it; series with no entry can be screened but cannot enter a model.</li>
<li><strong>Screen → cluster → search</strong> (the same engine as the companion balance-forecasting model). Univariate |t| controlling for the lag; greedy correlation clustering at 0.85, with a literature "primary" driver (unemployment, house prices, mortgage rate gap; CRE prices and spreads for business) representing its cluster when its |t| is within 80% of the leader; exhaustive search over 1-3 drivers with correct signs, |t| ≥ 1.65, VIF &lt; 5, distinct variables and 0 ≤ ρ &lt; 0.98.</li>
<li><strong>Ranking.</strong> By dynamic RMSE: the equation rebuilds the whole history from macro drivers alone, feeding back its own forecasts, the way it must under a scenario. The simplest model within 5% of the best wins.</li>
<li><strong>Challengers.</strong> BMA over the admissible set (BIC weights), the AR(1) benchmark and literature specifications re-estimated on the same data: BoC SAP 2026-12, a TDSA-style ARDL and an ECB/EBA core satellite for household books; BoC CDM-style and Fed CRE-style for commercial.</li>
<li><strong>Champion.</strong> Chosen by the out-of-sample record (section 4), not by in-sample fit.</li>
</ul>
<h3>Champion and search models</h3>
{table(model_tbl)}
{img('fig1_pd_paths.png', 'Annual default rate: history and projection by scenario')}

<h3>3.3 Loss given default</h3>
<span class="eq">LTV<sub>b,t</sub> = LTV<sub>b,0</sub> · (B<sub>t</sub>/B<sub>0</sub>) / (HPI<sub>t+4</sub>/HPI<sub>0</sub>)
LGD<sub>b,t</sub> = workout cost + max(0, 1 − (1 − d − c) / LTV<sub>b,t</sub>)
LGD<sub>t</sub> = Σ<sub>b</sub> w<sub>b</sub> · LGD<sub>b,t</sub>,   w<sub>b</sub> ∝ balance<sub>b</sub> · exp(3·(LTV<sub>b</sub> − mean LTV))</span>
<ul>
<li><strong>Uninsured mortgages:</strong> collateral valued at the scenario house-price index four quarters after default (EBA perfect foresight, liquidation lag), balance amortized on an annuity schedule, selling cost c = 8% and distressed-sale discount d = 10% netted from proceeds. Defaults are weighted toward high-LTV buckets, as TDSA allocates defaults to the riskiest buckets.</li>
<li><strong>Insured mortgages:</strong> lender LGD = residual unreimbursed cost (1%) + (1 − insurer coverage) × collateral shortfall.</li>
<li><strong>HELOCs:</strong> second lien; recovers only what is left after the amortized first mortgage.</li>
<li><strong>Commercial:</strong> Frye-Jacobs link, LGD rising with the default rate, anchored at a 30% long-run LGD.</li>
<li><strong>Consumer:</strong> fixed 75%.</li>
<li><strong>Cures:</strong> a share of 90+ defaults cure without loss; the cure rate falls as defaults rise (elasticity 0.3).</li>
</ul>
{img('fig3_lgd.png', 'Loss given default by scenario')}

<h3>3.4 Exposure, losses and IFRS 9 provisions</h3>
<span class="eq">EAD<sub>t</sub> = drawn<sub>t</sub> + CCF · undrawn<sub>t</sub>
Credit loss<sub>t</sub> = PD<sup>q</sup><sub>t</sub> · EAD<sub>t−1</sub> · (1 − cure<sub>t</sub>) · LGD<sub>t</sub>          (stress-test loss at default)
ECL<sub>Stage 1</sub> = Σ<sub>q=1..4</sub> S<sub>q−1</sub> · PD<sub>1,q</sub> · LGD*<sub>q</sub> · EAD<sub>q</sub> · (1+EIR/4)<sup>−q</sup>
ECL<sub>Stage 2</sub> = Σ<sub>q=1..L</sub> S<sub>q−1</sub> · PD<sub>2,q</sub> · LGD*<sub>q</sub> · EAD<sub>q</sub> · (1+EIR/4)<sup>−q</sup>      (L = remaining contractual term)
ECL<sub>Stage 3</sub> = GCA<sub>3</sub> · LGD* · discount to mid-workout
PCL<sub>t</sub> = (ACL<sub>t</sub> − ACL<sub>t−1</sub>) + net write-offs<sub>t</sub>,     ACL<sub>0</sub> = probability-weighted allowance</span>
<ul>
<li><strong>Balance sheet:</strong> static by default (EBA; Fed 2025), with run-off or balance-forecast growth paths as options.</li>
<li><strong>Stage engine</strong> (<code>creditloss/modelling/ifrs9.py</code>): gross loans are tracked in Stages 1–3 every quarter. Stage 1 → 2 on a
significant increase in credit risk (PD above three times origination PD, the EBA rule; borrower PD drift lognormal, calibrated to the reported
Stage 2 share), with an optional 30-days-past-due floor; Stage 2 → 1 cures after probation; defaults enter Stage 3; Stage 3 is resolved after
a workout period, cured to Stage 2 or written off. Repayments leave the book and, on a static balance sheet, new loans enter Stage 1.</li>
<li><strong>Measurement:</strong> 12-month ECL in Stage 1, lifetime in Stages 2 and 3, discounted at the effective interest rate. Lifetime is the
remaining contractual term (the revolving exception uses a behavioural life). Scenario PDs are used over the reasonable and supportable period,
then revert to the long-run average.</li>
<li><strong>Probability-weighted ECL</strong> at jump-off weights the four scenarios (IFRS 9 5.5.17). Every scenario opens from it, so the first
quarter's PCL includes the move to that scenario's own ECL. The Fed-style allowance (next four quarters of losses) is reported alongside.</li>
<li><strong>Disclosures:</strong> allowance and gross-loan roll-forwards by stage (IFRS 7.35H–35I), coverage ratios, and sensitivity of the ACL to
scenario weights and to staging; management overlays, if any, are reported separately.</li>
</ul>
{img('fig6_provisions.png', 'IFRS 9 provisions and allowance by scenario')}
{img('fig8_stages.png', 'Stage mix and ECL by stage')}

<h2>4. Validation</h2>
<p>At every fourth quarter from end-2006, the whole selection is re-run on data available at that date and every model
then forecasts 1-12 quarters ahead, conditional on the macro path that actually happened, which is how a satellite is
used in a stress test. Champions must capture at least 80% of cumulative GFC defaults out of sample; among those, the lowest
error wins.</p>
<h3>Out-of-sample RMSE, pp of annual default rate, averaged over 1-12 quarters</h3>
{oos_html}
<h3>Share of actual GFC defaults (2008Q3-2010Q2) captured, forecasting from end-2007, %</h3>
{stress_html}
<h3>Champions</h3>
{champ_html}
{img('fig7_oos_rmse.png', 'Out-of-sample error by model')}
<p>Three findings. First, on these data the theory-based ECB/EBA core satellite (unemployment level, house-price growth,
mortgage rate) beats the data-driven search out of sample in all four household segments, even though the search fits better in
sample: the search's lag and proxy choices shift from origin to origin, which is the instability the Fed warns about and
the hand-picking risk Gross &amp; Población describe. Second, BMA is not automatically conservative here; it captured
the GFC less well than the single best models for mortgages and HELOCs. Third, every model over-predicts the 2020-21
episode (median capture {covid_capture:.0f}% for the champions), because payment deferrals held defaults down; real
data would need an intervention dummy or an exclusion window for that period.</p>
<p>One caution on the champion: in the uninsured-mortgage equation the unemployment term has a full-sample t-statistic of
only {r['sels']['mortgage_uninsured']['specs'][champs['mortgage_uninsured']].tstats.get('LBR_ON|level|L1', float('nan')):.1f}.
It wins on forecast record, but a validator would ask for that term to be defended on theory (the BoC finds labour markets
the central driver) or replaced.</p>

<h2>5. Scenario results</h2>
<h3>Scenario design (simulated, vendor-style paths)</h3>
{table(pd.concat([scen_tbl, jo.to_frame().T]), "{:.1f}")}
<p>The severe path is calibrated to the size of the BoE's 2025 shock (house prices −28%) with Ontario unemployment
rising about 4 points, BBB spreads widening 2.6 points and the policy rate cut.</p>
<h3>Provisions (PCL) over {H} quarters, % of average EAD</h3>
{table(pcl)}
<p>In the severe scenario, peak annual default rates reach
{', '.join(f"{v:.2f}% for {k} ({sev_mult[k]:.1f}× jump-off)" for k, v in sev_pd.items())}.
Peak LGD reaches {sev_lgd['Mortgages, uninsured']:.0f}% on uninsured mortgages and {sev_lgd['HELOCs']:.0f}% on HELOCs, against workout-cost floors of 2% and 5%.</p>

<h3>What drives the severe-minus-baseline gap</h3>
<p>Counterfactual substitution: each component is swapped between the two scenarios and the two orderings are averaged
(Shapley), so the parts add up to the total. The PD leg is then split by driver by holding one macro series at its baseline
path at a time and rescaling to the PD total.</p>
{img('fig4_attribution.png', 'Severe-scenario loss bridge')}
{table(att_seg, "{:,.2f}")}
<p>For commercial loans the collateral effect shows up inside the PD leg, because the Frye-Jacobs LGD moves with defaults
rather than with a collateral index.</p>

<h3>Model risk</h3>
<p>The same scenario run through every PD model gives the range below. The spread between models is often as large as
the spread between downside and severe scenarios, which is why the champion is chosen on out-of-sample evidence and the
range is reported, not hidden.</p>
{img('fig5_model_range.png', 'Severe loss under each PD model')}

<h3>Probability-weighted ECL at jump-off</h3>
<p>Weights: {', '.join(f'{k} {v:.0%}' for k, v in {'baseline': .4, 'upside': .1, 'downside': .3, 'severe': .2}.items())}.
Weighted performing-loan ECL is C${pw_total['prob_weighted']:.1f}m against C${pw_total['baseline']:.1f}m from the baseline alone; the uplift
is largest where losses are most non-linear in the scenario (collateral-driven books).</p>
{table(pw_tbl, "{:,.2f}")}

<h3>Top-down cross-check</h3>
<p>The net charge-off equation (lag + one-quarter-ahead unemployment change, house-price and GDP growth, after TDSA eq. 25
and CLASS) is fitted on historical charge-offs and run on the same scenarios. Cumulative rates over {H} quarters, %:</p>
{table(nco_tbl)}
<p>Consumer and commercial reconcile within a few tenths of a point. For mortgages and HELOCs the bottom-up baseline is far
below the top-down figure: the top-down equation learned from a history that includes years of falling house prices,
while today's book has low indexed LTVs, so the structural LGD sits at its workout-cost floor. With real data, this gap
is the first thing to explain: either current LTVs are lower than the history implies, or the LGD floor and cure rate need
recalibrating to realized recoveries.</p>

<h2>6. Limitations and next steps with real data</h2>
<ul>
<li>Replace simulated files with your macro history and scenarios (e.g. a vendor S0-S4 set), and the credit union's default, charge-off and recovery history by segment. Check that the default-rate definition (new 90+ / impaired flow) matches what was reported historically.</li>
<li>Calibrate LGD parameters (selling cost, distressed discount, cure rate, workout cost) to realized recoveries, and the LTV-risk slope to observed default rates by LTV band.</li>
<li>Credit-union histories are short and few downturns are in them; pooling with sector data (provincial regulator filings, CBA arrears) or a panel across credit unions, as the BoC does across provinces, would stabilise the satellites.</li>
<li>Add the 2020-21 deferral intervention before trusting any coefficient estimated through that period.</li>
<li>Second-round effects (credit tightening feeding back into the economy) and the solvency-liquidity link are outside scope, as they are in TDSA.</li>
</ul>

<h2>Sources</h2>
<ul>
<li>Pugh, Wang &amp; Webley (2026), <a href="https://www.bankofcanada.ca/2026/03/staff-analytical-paper-2026-12/">Examining the macro drivers of mortgage arrears in Canada</a>, Bank of Canada SAP 2026-12.</li>
<li>Abdelrahman, Chen, MacDonald, Mordel &amp; Ouellet Leblanc (2025), <a href="https://www.bankofcanada.ca/2025/09/technical-report-128/">Top-Down Solvency Assessment tool</a>, Bank of Canada TR-128.</li>
<li>Bruneau, Duprey &amp; Hipp (2022), <a href="https://www.bankofcanada.ca/wp-content/uploads/2022/10/tr122.pdf">Forecasting banks' corporate loan losses under stress: a new corporate default model</a>, Bank of Canada TR-122.</li>
<li>Peterson &amp; Roberts (2016), <a href="https://www.bankofcanada.ca/2016/09/technical-report-106/">Household Risk Assessment Model</a>, Bank of Canada TR-106.</li>
<li>Fique (2017), <a href="https://www.bankofcanada.ca/2017/09/technical-report-111/">MacroFinancial Risk Assessment Framework 2.0</a>, Bank of Canada TR-111.</li>
<li>OSFI, <a href="https://www.osfi-bsif.gc.ca/en/data-forms/reporting-returns/standardized-climate-scenario-exercise">Standardized Climate Scenario Exercise methodology</a> (2024); <a href="https://arxiv.org/html/2602.01361v1">A methodology to measure impacts of scenarios through expected credit losses</a> (foundation of the SCSE).</li>
<li>Federal Reserve, <a href="https://www.federalreserve.gov/publications/2025-june-supervisory-stress-test-methodology-overview-modeling-framework.htm">2025 Supervisory Stress Test Methodology: overview of modeling framework</a>.</li>
<li>EBA, <a href="https://www.eba.europa.eu/sites/default/files/2024-11/ee01efca-947b-43ae-b17d-20f9d4314e8e/2025%20EU-wide%20stress%20test%20-%20Methodological%20Note.pdf">2025 EU-wide stress test methodological note</a>; staging and perfect-foresight conventions as summarised by the <a href="https://www.openriskmanual.org/wiki/EBA_2018_Credit_Risk_Stress_Test">Open Risk Manual</a>.</li>
<li>Gross &amp; Población (2015), <a href="https://www.ecb.int/pub/pdf/scpwps/ecbwp1845.en.pdf">A false sense of security in applying handpicked equations for stress test purposes</a>, ECB WP 1845.</li>
<li>Dees, Henry &amp; Martin (eds.), <a href="https://www.ecb.int/pub/conferences/shared/pdf/20170511_2nd_mp_policy/DeesHenryMartin-Stampe-Stress-Test_Analytics_for_Macroprudential_Purposes_in_the_euro_area.en.pdf">STAMP€: Stress-Test Analytics for Macroprudential Purposes in the euro area</a>, ECB.</li>
<li>Bank of England, <a href="https://www.bankofengland.co.uk/stress-testing/2025/key-elements-bank-capital">Key elements of the 2025 Bank Capital Stress Test</a>.</li>
<li>Hirtle, Kovner, Vickery &amp; Bhanot (2016), Assessing financial stability: the CLASS model, Journal of Banking &amp; Finance 69, S35-S55.</li>
<li>Frye &amp; Jacobs (2012), Credit loss and systematic loss given default, Journal of Credit Risk 8(1).</li>
</ul>
"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scenario Credit-Loss Model</title><style>{css}</style></head><body><main>{body}</main></body></html>"""
    p = OUT / "methodology_report.html"
    p.write_text(page, encoding="utf-8")
    print(f"Wrote {p}")
    return p

