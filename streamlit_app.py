"""Credit-loss scenario & IFRS 9 ECL app: set up, run and read scenario projections in the browser.

    pip install streamlit
    streamlit run app.py

Tabs: ① Data  ② Book and assumptions  ③ Scenarios  ④ Models and run  ⑤ Results.
Everything the app saves is an ordinary config file (config/<name>.yaml), so
`creditloss run --config config/<name>.yaml --backtest` repeats the same run from the command line.
"""
from __future__ import annotations

import copy
import io
import pickle
import subprocess
import sys
import time
import uuid
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
BUNDLED = {"eqb_public", "sample"}            # shipped configurations: read-only in the app

SCEN_KEYS = ["baseline", "upside", "downside", "severe"]
SCEN_COLOR = {"baseline": "#2a78d6", "upside": "#2e8b57", "downside": "#d18a1f", "severe": "#c0392b"}
LGD_MODELS = ["structural_ltv", "insured", "frye_jacobs", "fixed"]
EAD_MODELS = ["amortizing", "revolving"]
SEG_PARAMS = {  # field: (label, help, step, format)
    "cure_rate": ("Cure rate", "Share of defaults that cure without loss", 0.05, "%.2f"),
    "contract_rate": ("Contract rate", "Loan rate, for discounting and amortization", 0.005, "%.3f"),
    "sale_cost": ("Sale cost", "Costs of selling the property, share of value", 0.01, "%.2f"),
    "forced_sale_discount": ("Forced-sale discount", "Haircut at liquidation", 0.01, "%.2f"),
    "workout_cost": ("Workout cost", "Legal and carrying costs, share of balance", 0.01, "%.2f"),
    "insurer_coverage": ("Insurer coverage", "Share of the loss the mortgage insurer pays", 0.05, "%.2f"),
    "residual_cost": ("Residual cost", "Lender's unrecovered cost on insured loans", 0.005, "%.3f"),
    "fixed_lgd": ("LGD (fixed, or long-run for Frye-Jacobs)", None, 0.01, "%.2f"),
    "asset_corr": ("Asset correlation (Frye-Jacobs)", None, 0.01, "%.2f"),
    "ccf": ("Credit conversion factor on undrawn", None, 0.05, "%.2f"),
    "lifetime_years": ("Lifetime for Stage 2 (years)", "Overrides the IFRS 9 setting for this segment", 1.0, "%.0f"),
}
LGD_FIELDS = {"structural_ltv": ["sale_cost", "forced_sale_discount", "workout_cost"],
              "insured": ["insurer_coverage", "residual_cost"],
              "frye_jacobs": ["fixed_lgd", "asset_corr"], "fixed": ["fixed_lgd"]}

st.set_page_config(page_title="IFRS 9 Credit-Loss Scenario Model", page_icon="📉", layout="wide",
                   menu_items={"About": "IFRS 9 credit-loss scenario model · Xiaokai Wu · MIT licence"})


# ------------------------------------------------------------------ helpers
NICE = {"mortgage_uninsured": "Uninsured mortgages", "mortgage_insured": "Insured mortgages",
        "credit_cards": "Credit cards", "commercial": "Commercial", "heloc": "HELOCs", "consumer": "Consumer",
        "commercial_re": "Commercial real estate", "Total": "Total"}


def nice(seg) -> str:
    return NICE.get(seg, str(seg).replace("_", " ").capitalize())


def m(v):
    return f"C${v / 1e6:,.1f}m"


def rel(p) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def configs() -> list[Path]:
    return sorted((ROOT / "config").glob("*.yaml"))


def read_raw(path: Path) -> dict:
    return yaml.safe_load(open(path)) or {}


def abs_path(raw: dict, key: str) -> Path:
    p = Path(raw["paths"][key])
    return p if p.is_absolute() else ROOT / p


@st.cache_data(show_spinner=False)
def read_table(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)


def table(raw, key) -> pd.DataFrame | None:
    p = abs_path(raw, key)
    return read_table(str(p), p.stat().st_mtime) if p.exists() else None


def user_dir(raw) -> Path:
    d = ROOT / "data" / "user" / f"{st.session_state.name}-{st.session_state.sid}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_table(raw, key, df: pd.DataFrame):
    f = user_dir(raw) / f"{key}.csv"
    df.to_csv(f, index=False)
    raw["paths"][key] = rel(f)
    read_table.clear()


def qend(s):
    return pd.to_datetime(s).dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()


def raw_to_cfg(raw: dict) -> dict:
    """Write the edited config to a temporary file and load it with the package's own loader."""
    from creditloss.utils.config import load_config
    tmp = ROOT / "config" / f".app_{st.session_state.name}.yaml"
    yaml.safe_dump(raw, open(tmp, "w"), sort_keys=False)
    try:
        return load_config(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def out_dir(raw) -> Path:
    p = Path(raw["paths"].get("outputs", "outputs"))
    return p if p.is_absolute() else ROOT / p


def model_names(raw, seg) -> list[str]:
    sc = raw["segments"][seg]
    lit = (raw.get("pd_model", {}).get("literature_specs") or {}).get(sc.get("sign_priors", "household"), {}) or {}
    return ["Data-driven search", "BMA"] + list(lit) + ["AR(1) benchmark"]


def last_champions(raw) -> dict:
    f = results_dir(raw) / "champions.yaml"
    if not f.exists():
        f = out_dir(raw) / "champions.yaml"
    return yaml.safe_load(open(f)) if f.exists() else {}


def preflight(raw) -> list[tuple[str, str]]:
    """(level, message) pairs; level is 'error' or 'warn'."""
    msgs = []
    for k in ["macro_history", "macro_scenarios", "portfolio_history", "portfolio_snapshot", "literature_table"]:
        if not abs_path(raw, k).exists():
            msgs.append(("error", f"File for `{k}` not found: {raw['paths'][k]}"))
    if any(lvl == "error" for lvl, _ in msgs):
        return msgs
    dcol = raw.get("columns", {}).get("date", "date")
    mh, ms = table(raw, "macro_history"), table(raw, "macro_scenarios")
    ph, sn = table(raw, "portfolio_history"), table(raw, "portfolio_snapshot")
    for name, df in [("macro history", mh), ("macro scenarios", ms), ("portfolio history", ph)]:
        if dcol not in df:
            msgs.append(("error", f"{name}: no `{dcol}` column"))
    if "scenario" not in ms:
        msgs.append(("error", "macro scenarios: no `scenario` column (long-by-scenario layout expected)"))
    else:
        names = set(ms.scenario.astype(str))
        for k, v in raw["scenarios"].items():
            if str(v) not in names:
                msgs.append(("error", f"scenario `{k}` maps to `{v}`, which is not in the scenarios file"))
    segs = raw["segments"]
    for s, sc in segs.items():
        if s not in set(ph.get("segment", [])):
            msgs.append(("error", f"portfolio history has no rows for segment `{s}`"))
        if s not in set(sn.get("segment", [])):
            msgs.append(("error", f"snapshot has no rows for segment `{s}`"))
        miss = [c for c in sc.get("candidates", []) if c not in mh.columns]
        if miss:
            msgs.append(("error", f"{s}: candidate drivers not in macro history: {', '.join(miss)}"))
        if sc.get("jumpoff_pd") is None and sc.get("target_acl") is None:
            msgs.append(("warn", f"{s}: no jump-off PD and no target ACL; the proxy's last value is used as the PD"))
    need = {"segment", "balance", "current_ltv", "undrawn", "remaining_amort_years", "prepay_rate", "stage2_share",
            "stage3_share"}
    if need - set(sn.columns):
        msgs.append(("error", f"snapshot missing columns: {', '.join(sorted(need - set(sn.columns)))}"))
    ci = raw.get("collateral_index")
    if ci not in mh.columns:
        msgs.append(("error", f"collateral index `{ci}` not in macro history"))
    w = raw.get("ifrs9", {}).get("scenario_weights", {})
    if abs(sum(w.values()) - 1) > 1e-6:
        msgs.append(("error", f"scenario weights sum to {sum(w.values()):.2f}, not 1"))
    jo = pd.Timestamp(raw.get("jump_off", "2000-01-01"))
    last = qend(mh[dcol]).max()
    if last < jo:
        msgs.append(("warn", f"macro history ends {last.date()}, before the jump-off {jo.date()}"))
    h = raw.get("horizon_quarters", 13)
    n = ms.groupby("scenario").size().min() if "scenario" in ms else 0
    if n < h:
        msgs.append(("error", f"scenario paths have {n} quarters; the horizon needs {h}"))
    return msgs


def results_dir(raw) -> Path:
    """This session's latest run if there is one, else the saved results of the configuration."""
    p = st.session_state.get("run_out")
    if p and (Path(p) / "run.pkl").exists():
        return Path(p)
    return out_dir(raw)


def load_results(raw):
    """Saved results for the configuration. Pickles are tied to library versions, so when the shipped
    file cannot be read here (e.g. an older pandas), rebuild it once with the validated champions."""
    f = results_dir(raw) / "run.pkl"
    if f.exists():
        try:
            return pickle.loads(f.read_bytes()), f.stat().st_mtime
        except Exception:  # noqa: BLE001 - any unpickling failure means "rebuild"
            if f.parent != out_dir(raw):
                return None, None
            with st.spinner("Rebuilding the saved results for your Python setup (about 15 seconds, once) …"):
                from creditloss.pipelines.projection import run_projection
                run_projection(st.session_state.cfg_file)
            return pickle.loads(f.read_bytes()), f.stat().st_mtime
    return None, None


# ------------------------------------------------------------------ state
if "raw" not in st.session_state:
    first = ROOT / "config" / "eqb_public.yaml"
    st.session_state.cfg_file = str(first if first.exists() else configs()[0])
    st.session_state.raw = read_raw(Path(st.session_state.cfg_file))
    st.session_state.name = Path(st.session_state.cfg_file).stem
    st.session_state.history = []
    st.session_state.sid = uuid.uuid4().hex[:8]

with st.sidebar:
    st.title("IFRS 9 credit-loss scenario model")
    st.caption("Macro scenarios → PD satellites → LTV-based LGD → IFRS 9 stages → losses, allowance and provisions")
    opts = [str(p) for p in configs() if not p.name.startswith(".")]
    pick = st.selectbox("Configuration", opts, index=opts.index(st.session_state.cfg_file)
                        if st.session_state.cfg_file in opts else 0, format_func=lambda p: Path(p).name)
    if pick != st.session_state.cfg_file:
        st.session_state.cfg_file = pick
        st.session_state.raw = read_raw(Path(pick))
        st.session_state.name = Path(pick).stem
        st.session_state.pop("res", None)
        st.session_state.pop("run_out", None)
        st.rerun()
    st.session_state.name = st.text_input("Save as (name)", st.session_state.name,
                                          help="Save configuration writes config/<name>.yaml and runs from it write "
                                               "outputs/<name>/. The two shipped configurations are read-only: "
                                               "choose a new name to save your own.")
    raw = st.session_state.raw
    if st.button("Discard unsaved edits", width="stretch"):
        st.session_state.raw = read_raw(Path(st.session_state.cfg_file))
        st.rerun()
    st.divider()
    st.caption("`eqb_public.yaml`: EQB Inc. loan book (Jul 31, 2026), public disclosures and public proxies.  \n"
               "`sample.yaml`: simulated credit-union book with known drivers.")
    st.divider()
    from creditloss import __version__
    st.caption(f"v{__version__} · Xiaokai Wu · independent project built on public data only; not affiliated with, "
               "or endorsed by, EQB Inc. or any regulator. Results are illustrative, not investment advice.")

raw = st.session_state.raw
t0, t1, t2, t3, t4, t5, t6 = st.tabs(["Overview", "① Data", "② Book and assumptions", "③ Scenarios", "④ Models and run",
                                      "⑤ Results", "⑥ IFRS 9"])

# ================================================================== Overview
with t0:
    st.header("How much could this loan book lose, and what should it provision?")
    st.markdown(
        "This app projects credit losses, the IFRS 9 allowance and provisions for a loan book under macroeconomic "
        "scenarios, the way banks and supervisors run stress tests (Bank of Canada, OSFI, Federal Reserve, EBA/ECB). "
        "It ships with **EQB Inc.'s loan book at July 31, 2026**, built entirely from public disclosures and public "
        "data, and with a simulated credit-union book whose true drivers are known.")
    ov, _ = load_results(raw)
    if ov is not None:
        s = ov["summary"]
        H = int(s.horizon_q.max())
        s = s[s.horizon_q == H]
        tot = s.groupby("scenario")[["cum_loss", "cum_pcl", "avg_ead"]].sum()
        k = st.columns(4)
        if "baseline" in tot.index:
            k[0].metric(f"Baseline loss, {H} quarters", m(tot.loc["baseline", "cum_loss"]))
        if "severe" in tot.index:
            k[1].metric(f"Severe loss, {H} quarters", m(tot.loc["severe", "cum_loss"]))
            k[2].metric("Severe provisions (IFRS 9)", m(tot.loc["severe", "cum_pcl"]))
        if "ifrs9" in ov:
            k[3].metric("Allowance today, probability-weighted", m(ov["ifrs9"]["pw_by_stage"]["allowance_total"].sum()))
        c1, c2 = st.columns([3, 2])
        lb = s[s.scenario.isin(raw["scenarios"])].copy()
        lb["C$m"] = lb.cum_loss / 1e6
        lb["segment"] = lb.segment.map(nice)
        order = ["upside", "baseline", "downside", "severe"]
        dom = [x for x in order if x in set(lb.scenario)] + [x for x in raw["scenarios"] if x not in order]
        c1.markdown(f"**Cumulative credit loss over {H} quarters, C$m**")
        c1.altair_chart(alt.Chart(lb).mark_bar(size=46).encode(
            x=alt.X("scenario:N", sort=dom, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12)),
            y=alt.Y("C$m:Q", title=None),
            color=alt.Color("segment:N", title=None, legend=alt.Legend(orient="bottom", columns=2),
                            scale=alt.Scale(range=["#2a78d6", "#9ec5ee", "#e9a23b", "#c0392b", "#6a4fb3"])),
            tooltip=["segment", "scenario", alt.Tooltip("C$m:Q", format=",.1f")]).properties(height=330), width="stretch")
        with c2:
            st.markdown("**What sits behind the numbers**")
            st.graphviz_chart("""
digraph { rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#eef4fb", color="#9ec5ee", fontname="Helvetica", fontsize=10];
  edge [color="#8a8a8a"];
  M [label="Macro scenarios\nunemployment, house prices, rates, GDP"];
  P [label="PD satellites\nscreen, cluster, constrained search,\nBMA, literature challengers"];
  L [label="LGD from indexed LTV\n(collateral at liquidation)"];
  S [label="IFRS 9 stage engine\nSICR, cures, defaults, write-offs"];
  E [label="12-month / lifetime ECL\ndiscounted at EIR, probability-weighted"];
  O [label="Losses, allowance, provisions,\nIFRS 7 roll-forwards"];
  V [label="Out-of-sample backtest\nre-selection, DM tests, stress capture", fillcolor="#fdf3e6", color="#e9a23b"];
  M -> P -> S -> E -> O; M -> L -> E; V -> P [style=dashed, label=" champion", fontsize=9];
}""", width="stretch")
    st.markdown("""
**Try it**
1. **③ Scenarios**: build your own downturn (for example, house prices −35% by quarter 8) and add it.
2. **② Book and assumptions**: change the book, the loss-severity settings or the IFRS 9 parameters.
3. **④ Models and run**: *Quick run* takes about 10 seconds; the *full run* re-validates every model out of sample.
4. **⑤ Results** and **⑥ IFRS 9**: losses by segment, drivers of the severe-vs-baseline gap, allowance by stage,
   roll-forwards and sensitivities. Every run in the session is kept for side-by-side comparison.

Runs in the app are private to your session and never change the shipped results.
""")

# ================================================================== ① Data
with t1:
    st.subheader("Input files")
    st.caption("Four tables drive a run. Upload a CSV or Excel file to replace any of them; uploads are saved "
               "under `data/user/` for this session and the configuration points to them.")
    spec = {
        "macro_history": ("Macro history", "date + one column per macro series, quarterly"),
        "macro_scenarios": ("Macro scenarios", "scenario, date + the same macro columns, one block per scenario"),
        "portfolio_history": ("Portfolio history", "date, segment, default_rate (the PD proxy, annual %), "
                                                   "optional balance, nco_rate"),
        "portfolio_snapshot": ("Jump-off book", "segment, ltv_bucket, balance, current_ltv, undrawn, "
                                                "remaining_amort_years, prepay_rate, stage2_share, stage3_share"),
        "literature_table": ("Literature / sign-prior table", "variable, sign per book type, allowed transforms"),
    }
    for key, (lab, desc) in spec.items():
        df = table(raw, key)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**{lab}** · `{raw['paths'][key]}`")
            st.caption(desc + (f" — {len(df):,} rows × {df.shape[1]} columns" if df is not None else " — not found"))
        up = c2.file_uploader(f"Replace {lab.lower()}", type=["csv", "xlsx"], key=f"up_{key}",
                              label_visibility="collapsed")
        if up is not None and st.session_state.get(f"done_{key}") != up.file_id:
            new = pd.read_excel(up) if up.name.endswith("xlsx") else pd.read_csv(up)
            save_table(raw, key, new)
            st.session_state[f"done_{key}"] = up.file_id
            st.rerun()
        if df is not None:
            with st.expander("Preview"):
                st.dataframe(df.head(12), width="stretch", hide_index=True)
    st.divider()
    st.subheader("Loss proxies by segment")
    ph = table(raw, "portfolio_history")
    if ph is not None and "default_rate" in ph:
        d = ph.copy()
        d["date"] = qend(d[raw["columns"]["date"]])
        d = d.drop_duplicates(["date", "segment"])
        ch = alt.Chart(d).mark_line().encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")), y=alt.Y("default_rate:Q", title="proxy rate, %"),
            color=alt.Color("segment:N", legend=alt.Legend(orient="bottom")),
            tooltip=["segment", alt.Tooltip("date:T"), alt.Tooltip("default_rate:Q", format=".3f")]).properties(height=280)
        st.altair_chart(ch, width="stretch")
        for s, sc in raw["segments"].items():
            if sc.get("proxy"):
                st.caption(f"**{s}**: {sc['proxy']}")

# ================================================================== ② Book and assumptions
with t2:
    st.subheader("Jump-off book")
    sn = table(raw, "portfolio_snapshot")
    if sn is not None:
        edited = st.data_editor(sn, num_rows="dynamic", width="stretch", hide_index=True, key="snap_ed")
        tot = edited.groupby("segment").agg(balance=("balance", "sum"), undrawn=("undrawn", "sum"))
        tot["avg LTV"] = edited.assign(w=edited.balance * edited.current_ltv).groupby("segment").w.sum() / tot.balance
        c1, c2 = st.columns([3, 1])
        c1.dataframe((tot.assign(balance=tot.balance / 1e6, undrawn=tot.undrawn / 1e6)
                      .rename(columns={"balance": "balance C$m", "undrawn": "undrawn C$m"})).round(2),
                     width="stretch")
        if c2.button("Use this book", type="primary", disabled=edited.equals(sn)):
            save_table(raw, "portfolio_snapshot", edited)
            st.success("Saved. Run again in ④ to see the effect.")
            st.rerun()
    st.divider()
    st.subheader("Segments")
    mh = table(raw, "macro_history")
    macro_cols = [c for c in (mh.columns if mh is not None else []) if c != raw["columns"]["date"]]
    for seg, sc in raw["segments"].items():
        with st.expander(f"**{seg}** — {sc.get('label', '')}"):
            c1, c2, c3 = st.columns(3)
            sc["lgd_model"] = c1.selectbox("LGD model", LGD_MODELS, LGD_MODELS.index(sc.get("lgd_model", "fixed")),
                                           key=f"lgd_{seg}")
            sc["ead_model"] = c2.selectbox("EAD model", EAD_MODELS, EAD_MODELS.index(sc.get("ead_model", "amortizing")),
                                           key=f"ead_{seg}")
            sc["sign_priors"] = c3.selectbox("Sign priors", ["household", "business", "us"],
                                             ["household", "business", "us"].index(sc.get("sign_priors", "household")),
                                             key=f"sp_{seg}", help="Which sign column of the literature table applies")
            mode = st.radio("Bank's jump-off PD", ["Fixed annual PD", "Solve so the model allowance matches the reported ACL"],
                            index=1 if sc.get("target_acl") is not None else 0, horizontal=True, key=f"mode_{seg}")
            c1, c2 = st.columns(2)
            if mode.startswith("Fixed"):
                sc["target_acl"] = None
                sc["jumpoff_pd"] = c1.number_input("Jump-off PD, annual %", 0.0, 50.0, float(sc.get("jumpoff_pd") or 1.0),
                                                   0.1, key=f"pd_{seg}")
            else:
                sc["jumpoff_pd"] = None
                sc["target_acl"] = c1.number_input("Target Stage 1+2 ACL, C$", 0.0, None,
                                                   float(sc.get("target_acl") or sc.get("reported_acl") or 0), 1e6,
                                                   format="%.0f", key=f"tacl_{seg}")
            sc["reported_acl"] = c2.number_input("Reported Stage 1+2 ACL, C$ (for reconciliation)", 0.0, None,
                                                 float(sc.get("reported_acl") or 0), 1e6, format="%.0f", key=f"racl_{seg}")
            fields = LGD_FIELDS[sc["lgd_model"]] + ["cure_rate", "contract_rate"] + \
                (["ccf", "lifetime_years"] if sc["ead_model"] == "revolving" else [])
            cols = st.columns(4)
            for i, f in enumerate(fields):
                lab, hlp, stp, fmt = SEG_PARAMS[f]
                default = {"cure_rate": 0.3, "contract_rate": 0.05, "ccf": 0.25, "lifetime_years": 3,
                           "fixed_lgd": 0.4, "asset_corr": 0.15}.get(f, 0.0)
                v = cols[i % 4].number_input(lab, value=float(sc.get(f, default)), step=stp, format=fmt, help=hlp,
                                             key=f"{f}_{seg}")
                sc[f] = int(v) if f == "lifetime_years" else v
            st.markdown("IFRS 9 measurement")
            c1, c2, c3, c4 = st.columns(4)
            if sc["ead_model"] != "revolving":
                v = c1.number_input("Remaining contractual term, years", 0.25, 30.0,
                                    float(sc.get("contractual_term_years") or raw.get("ifrs9", {}).get("lifetime_years", 5)),
                                    0.25, key=f"term_{seg}", help="Lifetime ECL horizon for Stage 2: the maximum contractual "
                                    "period (IFRS 9 5.5.19). For Canadian mortgages, the remaining term to renewal.")
                sc["contractual_term_years"] = v
            sc["workout_quarters"] = int(c2.number_input("Workout, quarters", 1, 16,
                                         int(sc.get("workout_quarters") or raw.get("ifrs9", {}).get("workout_quarters", 4)),
                                         key=f"wo_{seg}", help="Quarters from Stage 3 entry to resolution (cure or write-off)"))
            sc["dpd30_share"] = c3.number_input("30+ days past due share", 0.0, 1.0, float(sc.get("dpd30_share") or 0.0), 0.005,
                                                format="%.3f", key=f"dpd_{seg}",
                                                help="Backstop (IFRS 9 5.5.11): at least this share of performing loans is in Stage 2")
            sc["overlay_pct"] = c4.number_input("Management overlay, % of modelled ECL", -50.0, 200.0,
                                                100 * float(sc.get("overlay_pct") or 0.0), 1.0, key=f"ovp_{seg}") / 100
            sc["overlay_amount"] = c1.number_input("Management overlay, C$", None, None, float(sc.get("overlay_amount") or 0.0),
                                                   1e6, format="%.0f", key=f"ova_{seg}")
            sc["overlay_rationale"] = st.text_input("Overlay rationale (shown in the disclosures)",
                                                    sc.get("overlay_rationale", ""), key=f"ovr_{seg}")
            sc["candidates"] = st.multiselect("Candidate macro drivers for the PD satellite", macro_cols,
                                              [c for c in sc.get("candidates", []) if c in macro_cols], key=f"cand_{seg}")
    st.divider()
    st.subheader("Collateral and IFRS 9")
    c1, c2, c3, c4 = st.columns(4)
    if macro_cols:
        ci = raw.get("collateral_index")
        raw["collateral_index"] = c1.selectbox("House-price index for LTV", macro_cols,
                                               macro_cols.index(ci) if ci in macro_cols else 0)
    f9 = raw.setdefault("ifrs9", {})
    f9["sicr_multiple"] = c2.number_input("Stage 2 trigger: PD multiple", 1.0, 10.0, float(f9.get("sicr_multiple", 3.0)), 0.5)
    f9["lifetime_years"] = int(c3.number_input("Default lifetime (years)", 1, 30, int(f9.get("lifetime_years", 5)),
                                               help="Used where a segment gives no contractual term"))
    f9["discount"] = c4.checkbox("Discount at the effective interest rate", bool(f9.get("discount", True)))
    c1, c2, c3, c4 = st.columns(4)
    f9["reasonable_supportable_quarters"] = int(c1.number_input(
        "Reasonable and supportable period, quarters", 4, 40, int(f9.get("reasonable_supportable_quarters", 12)),
        help="Scenario PDs are used this far ahead, then revert to the long-run average"))
    f9["reversion_quarters"] = int(c2.number_input("Reversion period, quarters", 1, 20, int(f9.get("reversion_quarters", 8))))
    f9["stage2_pd_multiple"] = c3.number_input("Stage 2 PD vs Stage 1 (multiple)", 1.0, 10.0,
                                               float(f9.get("stage2_pd_multiple", 3.0)), 0.5)
    f9["stage2_cure_q"] = c4.number_input("Stage 2 cure rate per quarter", 0.0, 1.0, float(f9.get("stage2_cure_q", 0.10)),
                                          0.01, help="Return to Stage 1 after probation")
    st.markdown("**Scenario weights for the probability-weighted allowance**")
    w = f9.setdefault("scenario_weights", {})
    cols = st.columns(len(raw["scenarios"]))
    for i, k in enumerate(raw["scenarios"]):
        w[k] = cols[i].number_input(k, 0.0, 1.0, float(w.get(k, 0.0)), 0.05, key=f"w_{k}")
    s = sum(w.values())
    (st.success if abs(s - 1) < 1e-6 else st.error)(f"Weights sum to {s:.2f}")

# ================================================================== ③ Scenarios
with t3:
    ms, mh = table(raw, "macro_scenarios"), table(raw, "macro_history")
    if ms is None or mh is None:
        st.warning("Load the macro files in ①.")
    else:
        dcol = raw["columns"]["date"]
        names = sorted(ms.scenario.astype(str).unique())
        st.subheader("Which scenario is which")
        cols = st.columns(len(raw["scenarios"]))
        for i, k in enumerate(list(raw["scenarios"])):
            cur = str(raw["scenarios"][k])
            raw["scenarios"][k] = cols[i].selectbox(k, names, names.index(cur) if cur in names else 0, key=f"map_{k}")
        c1, c2 = st.columns(2)
        raw["horizon_quarters"] = int(c1.number_input("Loss horizon (quarters)", 4, 24, int(raw.get("horizon_quarters", 13))))
        raw["jump_off"] = str(c2.date_input("Jump-off (last actual quarter)", pd.Timestamp(raw.get("jump_off", "2026-06-30"))))
        st.divider()
        st.subheader("Paths")
        mcols = [c for c in ms.columns if c not in (dcol, "scenario")]
        show = st.multiselect("Variables", mcols, [c for c in [raw.get("collateral_index"), "LBR_CA", "R3M"] if c in mcols][:3]
                              or mcols[:2])
        h = mh.copy()
        h[dcol] = qend(h[dcol])
        h = h[h[dcol] >= h[dcol].max() - pd.DateOffset(years=10)]
        s = ms.copy()
        s[dcol] = qend(s[dcol])
        inv = {str(v): k for k, v in raw["scenarios"].items()}
        lt = table(raw, "literature_table")
        desc = dict(zip(lt.mnemonic, lt.description)) if lt is not None and "description" in lt else {}
        cols = st.columns(min(3, max(1, len(show))))
        for i, v in enumerate(show):
            hh = h[[dcol, v]].assign(scenario="history")
            ss = s[[dcol, "scenario", v]].copy()
            ss["scenario"] = ss.scenario.astype(str).map(lambda x: inv.get(x, x))
            dd = pd.concat([hh, ss]).rename(columns={dcol: "date", v: "value"})
            ch = alt.Chart(dd).mark_line().encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")), y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("scenario:N", scale=alt.Scale(domain=["history"] + list(SCEN_COLOR) + [n for n in dd.scenario.unique() if n not in SCEN_COLOR and n != "history"],
                                                              range=["#1b1b1a"] + list(SCEN_COLOR.values()) + ["#6a4fb3", "#8c564b", "#17becf"])),
                tooltip=["scenario", alt.Tooltip("date:T"), alt.Tooltip("value:Q", format=".2f")]).properties(height=240)
            ch = ch.properties(title=alt.TitleParams(v, subtitle=desc.get(v, ""), anchor="start"))
            ch = ch.configure_legend(orient="bottom", columns=3, title=None)
            cols[i % len(cols)].altair_chart(ch, width="stretch")
        st.divider()
        st.subheader("Build a scenario")
        st.caption("Start from an existing scenario and add shocks. A hump shock rises to its peak at the chosen "
                   "quarter and fades; a ramp builds to its full size and stays. Shocks to index series (house prices, "
                   "GDP, equity) are in %; shocks to rates are in percentage points.")
        c1, c2 = st.columns(2)
        base = c1.selectbox("Start from", names, names.index(str(raw["scenarios"].get("baseline", names[0])))
                            if str(raw["scenarios"].get("baseline")) in names else 0)
        new_name = c2.text_input("New scenario name", "custom_severe")
        shocks = st.data_editor(
            pd.DataFrame([{"variable": raw.get("collateral_index", mcols[0]), "units": "%", "shape": "hump", "size": -20.0,
                           "peak_quarter": 8, "width": 5.0}]),
            num_rows="dynamic", width="stretch", hide_index=True, key="shock_ed",
            column_config={"variable": st.column_config.SelectboxColumn(options=mcols, required=True),
                           "units": st.column_config.SelectboxColumn(options=["%", "points"], required=True),
                           "shape": st.column_config.SelectboxColumn(options=["hump", "ramp"], required=True),
                           "peak_quarter": st.column_config.NumberColumn("peak / full by quarter", min_value=1, step=1),
                           "width": st.column_config.NumberColumn("hump width (quarters)", min_value=0.5)})
        b = s[s.scenario.astype(str) == base].sort_values(dcol).reset_index(drop=True)
        new = b.copy()
        hh = np.arange(1, len(b) + 1)
        for r in shocks.dropna(subset=["variable"]).itertuples():
            k = max(1, int(r.peak_quarter or 1))
            prof = (np.exp(-0.5 * ((hh - k) / float(r.width or 4)) ** 2) if r.shape == "hump" else np.minimum(hh / k, 1.0))
            sh = float(r.size or 0) * prof
            new[r.variable] = new[r.variable] * (1 + sh / 100) if r.units == "%" else new[r.variable] + sh
        new["scenario"] = new_name
        prev = shocks.dropna(subset=["variable"]).variable.unique().tolist()
        if prev:
            cmp = pd.concat([b.assign(which=base), new.assign(which=new_name)])
            cmp = cmp.melt(id_vars=[dcol, "which"], value_vars=prev, var_name="variable").rename(columns={dcol: "date"})
            st.altair_chart(alt.Chart(cmp).mark_line().encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")), y=alt.Y("value:Q", scale=alt.Scale(zero=False), title=None),
                color="which:N", strokeDash="which:N").properties(height=200, width=260)
                .facet(column=alt.Column("variable:N", title=None)).resolve_scale(y="independent"))
        c1, c2 = st.columns(2)
        role = c1.selectbox("Use it as", ["an extra scenario (weight 0)"] + [f"the {k} scenario" for k in raw["scenarios"]])
        if c2.button("Add to the scenarios file", type="primary"):
            out = pd.concat([ms[ms.scenario.astype(str) != new_name], new.assign(**{dcol: new[dcol].dt.date.astype(str)})])
            save_table(raw, "macro_scenarios", out)
            key = new_name if role.startswith("an extra") else role.split()[1]
            raw["scenarios"][key] = new_name
            raw["ifrs9"]["scenario_weights"].setdefault(key, 0.0)
            st.success(f"Added `{new_name}` as {role.replace('an extra scenario', 'an extra scenario')}.")
            st.rerun()

# ================================================================== ④ Models and run
with t4:
    pm = raw.setdefault("pd_model", {})
    st.subheader("PD satellite selection")
    c1, c2, c3, c4 = st.columns(4)
    pm["max_drivers"] = int(c1.number_input("Most drivers", 1, 5, int(pm.get("max_drivers", 3))))
    pm["max_vif"] = float(c2.number_input("Largest VIF", 1.0, 20.0, float(pm.get("max_vif", 5)), 0.5))
    pm["require_sign_priors"] = c3.checkbox("Correct signs only", bool(pm.get("require_sign_priors", True)))
    pm.setdefault("bma", {})["enabled"] = c4.checkbox("Bayesian model averaging", bool(pm.get("bma", {}).get("enabled", True)))
    c1, c2 = st.columns(2)
    pm["lags"] = c1.multiselect("Lags (quarters)", list(range(9)), pm.get("lags", [0, 1, 2, 3, 4]))
    pm["variable_transforms"] = c2.multiselect("Transforms", ["level", "d4", "yoy"],
                                               pm.get("variable_transforms", ["level", "d4", "yoy"]))
    with st.expander("Literature specifications (challengers re-estimated on your data)"):
        txt = st.text_area("YAML", yaml.safe_dump(pm.get("literature_specs", {}), sort_keys=False), height=220,
                           label_visibility="collapsed")
        try:
            pm["literature_specs"] = yaml.safe_load(txt) or {}
        except yaml.YAMLError as e:
            st.error(f"Not valid YAML: {e}")
        st.caption("Drivers are written `SERIES|transform|Llag`, e.g. `LBR_CA|d4|L0`; `ar: true` adds the lagged PD.")
    st.subheader("Out-of-sample test and champion rule")
    o = raw.setdefault("oos", {})
    c1, c2, c3, c4 = st.columns(4)
    o["first_origin"] = str(c1.date_input("First origin", pd.Timestamp(o.get("first_origin", "2006-12-31"))))
    o["step_quarters"] = int(c2.number_input("Step (quarters)", 1, 8, int(o.get("step_quarters", 4))))
    o["champion_min_capture"] = float(c3.number_input("Stress capture from, %", 0.0, 200.0, float(o.get("champion_min_capture", 80))))
    o["champion_max_capture"] = float(c4.number_input("to, %", 100.0, 1000.0, float(o.get("champion_max_capture", 200))))
    sw = st.text_input("Stress windows (start:end, comma-separated)",
                       ", ".join(f"{a}:{b}" for a, b in o.get("stress_windows", [])))
    o["stress_windows"] = [[a.strip(), b.strip()] for a, b in (x.split(":") for x in sw.split(",") if ":" in x)]
    st.caption("The champion is the correctly-signed model with the lowest out-of-sample error whose cumulative "
               "capture of the first stress window falls in the band.")

    st.subheader("Model used for the projection")
    lc = last_champions(raw)
    ch = pm.setdefault("champion_by_segment", {})
    cols = st.columns(len(raw["segments"]))
    for i, seg in enumerate(raw["segments"]):
        opts = (["Last out-of-sample choice"] if lc.get(seg) else []) + model_names(raw, seg)
        cur = ch.get(seg)
        idx = opts.index(cur) if cur in opts else 0
        v = cols[i].selectbox(seg, opts, idx, key=f"champ_{seg}",
                              help=f"Last validated choice: {lc.get(seg, 'none yet; run the full test')}")
        ch[seg] = lc[seg] if v == "Last out-of-sample choice" else v
    st.divider()
    st.subheader("Check and run")
    msgs = preflight(raw)
    if not msgs:
        st.success("All checks passed.")
    for lvl, msg in msgs:
        (st.error if lvl == "error" else st.warning)(msg)
    blocked = any(lvl == "error" for lvl, _ in msgs)
    c1, c2, c3 = st.columns(3)

    def save_config(working: bool = False) -> Path:
        """Write the config. working=True writes config/.run_<name>.yaml for a run and leaves the named file alone."""
        r2 = copy.deepcopy(raw)
        if working:          # runs from the app never overwrite saved results: one output folder per session
            r2["paths"]["outputs"] = f"outputs/app/{st.session_state.name}-{st.session_state.sid}"
            r2["paths"]["cache"] = f".cache/{st.session_state.name}"
        elif st.session_state.name not in BUNDLED:
            r2["paths"]["outputs"] = f"outputs/{st.session_state.name}"
            r2["paths"]["cache"] = f".cache/{st.session_state.name}"
        name = f".run_{st.session_state.name}-{st.session_state.sid}" if working else st.session_state.name
        f = ROOT / "config" / f"{name}.yaml"
        yaml.safe_dump(r2, open(f, "w"), sort_keys=False)
        if working:
            st.session_state.run_out = str(ROOT / r2["paths"]["outputs"])
        else:
            st.session_state.raw = r2
            st.session_state.cfg_file = str(f)
        return f

    if c1.button("Save configuration", width="stretch"):
        if st.session_state.name in BUNDLED:
            st.warning("The shipped configurations are read-only. Enter a new name in the sidebar to save your own.")
        else:
            f = save_config()
            st.success(f"Saved `{rel(f)}`")
    quick = c2.button("Quick run (no out-of-sample test, ~10 s)", type="primary", disabled=blocked, width="stretch")
    full = c3.button("Full run with out-of-sample test (minutes)", disabled=blocked, width="stretch")
    if quick:
        f = save_config(working=True)
        from creditloss.reporting import report as R
        from creditloss.scenarios.run import run
        from creditloss.utils.config import load_config
        cfg = load_config(f)
        logs = []
        with st.status("Running …", expanded=True) as box:
            t0 = time.time()
            res = run(cfg, log=lambda *a: (logs.append(" ".join(map(str, a))), box.write(" ".join(map(str, a)))))
            out = Path(cfg["paths"]["outputs"])
            R._style()
            for fn in (R.chart_pd_paths, R.chart_loss_rates, R.chart_lgd):
                fn(res, cfg, out)
            for fn in (R.chart_attribution, R.chart_model_range, R.chart_provisions, R.chart_stages):
                fn(res, out)
            prior = {k: pd.read_csv(out / f) for k, f in [("metrics", "oos_metrics.csv"), ("stress", "oos_stress_capture.csv"),
                     ("dm", "oos_dm.csv"), ("champions", "oos_champions.csv")] if (out / f).exists()}
            R.write_excel(res, out, prior or None)          # keep the last out-of-sample record in the workbook
            (out / "run.pkl").write_bytes(pickle.dumps(res))
            box.update(label=f"Done in {time.time() - t0:.0f} s", state="complete")
        st.session_state.res = res
        s = res["summary"]
        s = s[s.horizon_q == s.horizon_q.max()].groupby("scenario").cum_loss.sum() / 1e6
        st.session_state.history.append({"run": f"{st.session_state.name} · {time.strftime('%H:%M:%S')}",
                                         **{k: s.get(k, np.nan) for k in raw["scenarios"]},
                                         **{f"model: {k}": v for k, v in res["champions"].items()}})
        st.success("Done. See ⑤ Results.")
    if full:
        f = save_config(working=True)
        box = st.empty()
        prog = st.progress(0.0, "Starting …")
        lines = []
        p = subprocess.Popen([sys.executable, "-u", "-m", "creditloss", "run", "--config", str(f), "--backtest"],
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        n_or = None
        for line in p.stdout:
            lines.append(line.rstrip())
            if "origin" in line and "/" in line:
                try:
                    a, b = line.split("(")[1].split(")")[0].split("/")
                    n_or = int(b)
                    prog.progress(min(int(a) / int(b), 1.0), f"Out-of-sample origin {a} of {b}")
                except (IndexError, ValueError):
                    pass
            box.code("\n".join(lines[-25:]))
        p.wait()
        prog.progress(1.0, "Finished" if p.returncode == 0 else "Failed")
        if p.returncode == 0:
            st.session_state.pop("res", None)
            st.success("Done. The validated champions are saved and used by default. See ⑤ Results.")
        else:
            st.error("The run failed; the log above shows where.")

# ================================================================== ⑤ Results
with t5:
    res = st.session_state.get("res")
    when = "this session"
    if res is None:
        res, mt = load_results(raw)
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)) if mt else None
    if res is None:
        st.info("No results yet for this configuration. Run it in ④.")
    else:
        segs = list(res["summary"].segment.unique())
        H = int(res["summary"].horizon_q.max())
        st.caption(f"Results from {when} · {H}-quarter horizon · output folder `{rel(results_dir(raw))}`")
        s = res["summary"][res["summary"].horizon_q == H]
        tot = s.groupby("scenario")[["cum_loss", "cum_pcl", "avg_ead"]].sum()
        pw = res["pw_ecl"]
        rep = {g: (raw["segments"].get(g, {}).get("reported_acl") or 0) for g in pw.index}
        k = st.columns(4)

        def kpi(col, label, value, note=None):
            col.metric(label, value)
            if note:
                col.caption(note)

        if "baseline" in tot.index:
            kpi(k[0], "Baseline loss", m(tot.loc["baseline", "cum_loss"]),
                f"{100 * tot.loc['baseline', 'cum_loss'] / tot.loc['baseline', 'avg_ead']:.2f}% of average exposure")
        if "severe" in tot.index:
            kpi(k[1], "Severe loss", m(tot.loc["severe", "cum_loss"]),
                f"{100 * tot.loc['severe', 'cum_loss'] / tot.loc['severe', 'avg_ead']:.2f}% of average exposure")
            kpi(k[2], "Severe provisions (PCL)", m(tot.loc["severe", "cum_pcl"]), "change in allowance + write-offs")
        kpi(k[3], "Probability-weighted ACL, Stage 1+2", m(pw["prob_weighted"].sum()),
            f"reported {m(sum(rep.values()))}" if sum(rep.values()) else None)

        st.subheader("Losses by segment")
        unit = st.radio("Show", ["C$m", "% of average exposure", "Provisions, C$m"], horizontal=True)
        hz = st.radio("Horizon", sorted(res["summary"].horizon_q.unique()), index=len(res["summary"].horizon_q.unique()) - 1,
                      horizontal=True, format_func=lambda x: f"{x} quarters")
        ss = res["summary"][res["summary"].horizon_q == hz]
        val = {"C$m": "cum_loss", "% of average exposure": "loss_rate_pct", "Provisions, C$m": "cum_pcl"}[unit]
        pv = ss.pivot(index="segment", columns="scenario", values=val)
        pv = pv[[c for c in raw["scenarios"] if c in pv.columns]]
        if val != "loss_rate_pct":
            pv = pv / 1e6
            pv.loc["Total"] = pv.sum()
        pv.index = pv.index.map(nice)
        st.dataframe(pv.style.format("{:,.1f}" if val != "loss_rate_pct" else "{:.2f}"), width="stretch")

        st.subheader("Paths")
        c1, c2 = st.columns([1, 3])
        seg = c1.selectbox("Segment", segs, format_func=nice)
        what = c1.radio("Measure", ["PD, annual %", "LGD, %", "Credit loss, C$m per quarter", "Cumulative loss, C$m",
                                    "Stage 2 share, %", "Allowance, C$m", "Provisions (PCL), C$m per quarter",
                                    "Write-offs, C$m per quarter"])
        P = res["paths"][(res["paths"].segment == seg) & (res["paths"].q > 0)].copy()
        P["cum"] = P.groupby("scenario").credit_loss.cumsum()
        col, scale = {"PD, annual %": ("pd_annual", 100), "LGD, %": ("lgd", 100),
                      "Credit loss, C$m per quarter": ("credit_loss", 1e-6), "Cumulative loss, C$m": ("cum", 1e-6),
                      "Stage 2 share, %": ("stage2_share", 100), "Allowance, C$m": ("allowance_total", 1e-6),
                      "Provisions (PCL), C$m per quarter": ("pcl", 1e-6), "Write-offs, C$m per quarter": ("net_writeoffs", 1e-6)}[what]
        P["value"] = P[col] * scale
        dom = [x for x in raw["scenarios"] if x in P.scenario.unique()]
        c2.altair_chart(alt.Chart(P).mark_line(point=False).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")), y=alt.Y("value:Q", title=what),
            color=alt.Color("scenario:N", scale=alt.Scale(domain=dom, range=[SCEN_COLOR.get(x, "#6a4fb3") for x in dom]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["scenario", alt.Tooltip("date:T"), alt.Tooltip("value:Q", format=",.2f")]).properties(height=300),
            width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Why severe is worse than baseline")
            a = res["attribution"][res["attribution"].segment == seg][["component", "amount"]].copy()
            a["C$m"] = a.amount / 1e6
            st.dataframe(a[["component", "C$m"]].style.format({"C$m": "{:,.1f}"}), hide_index=True, width="stretch")
            st.caption("PD vs collateral split by Shapley averaging; the PD part split by holding one driver at its "
                       "baseline path.")
        with c2:
            st.subheader("Allowance at jump-off")
            t = pw[[c for c in list(raw["scenarios"]) + ["prob_weighted"] if c in pw.columns]] / 1e6
            t["reported"] = [rep.get(g, 0) / 1e6 for g in t.index]
            t.index = t.index.map(nice)
            st.dataframe(t.style.format("{:,.1f}"), width="stretch")
            st.caption("Stage 1+2 allowance, C$m, under each scenario and probability-weighted, against the reported "
                       "Stage 1+2 ACL. Stages, coverage and roll-forwards are in ⑥ IFRS 9.")

        st.subheader("Model risk: severe loss under every PD model")
        mr = res["model_range"]
        if len(mr):
            mm = mr[(mr.segment == seg)].pivot(index="model", columns="scenario", values="loss_rate_pct")
            mm["used"] = ["◀ used" if x == res["champions"].get(seg) else "" for x in mm.index]
            st.dataframe(mm.style.format({c: "{:.2f}" for c in mm.columns if c != "used"}), width="stretch")
            st.caption("Cumulative loss, % of average exposure.")
        sel = res["sels"].get(seg, {})
        if sel.get("specs"):
            rows = []
            for nm, sp in sel["specs"].items():
                rows.append({"model": nm, "drivers": ", ".join(sp.drivers) or "(AR only)",
                             "coefficients (t)": ", ".join(f"{k} {v:+.3f} ({sp.tstats.get(k, np.nan):.1f})"
                                                           for k, v in sp.params.items() if k != "const"),
                             "signs ok": sp.sign_ok, "BIC": sp.bic, "dyn. RMSE, pp": sp.dyn_rmse, "obs": sp.nobs})
            with st.expander("PD models estimated for this segment"):
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        od = results_dir(raw)
        if (od / "oos_metrics.csv").exists():
            st.subheader("Out-of-sample record (last full run)")
            om = pd.read_csv(od / "oos_metrics.csv")
            om = om[om.segment == seg].pivot(index="model", columns="h", values="rmse_pp")
            om.columns = [f"RMSE {c}q, pp" for c in om.columns]
            if (od / "oos_stress_capture.csv").exists():
                sc_ = pd.read_csv(od / "oos_stress_capture.csv")
                sc_ = sc_[sc_.segment == seg].pivot_table(index="model", columns="window", values="cum_capture_pct")
                sc_.columns = [f"capture {c[:4]}–{c[-10:-6]}, %" for c in sc_.columns]
                om = om.join(sc_)
            st.dataframe(om.style.format("{:.2f}"), width="stretch")
            if (od / "oos_champions.csv").exists():
                st.dataframe(pd.read_csv(od / "oos_champions.csv"), hide_index=True, width="stretch")

        if st.session_state.history:
            st.subheader("Runs this session")
            st.dataframe(pd.DataFrame(st.session_state.history).style.format(
                {k: "{:,.1f}" for k in raw["scenarios"]}, na_rep="–"), hide_index=True, width="stretch")
            st.caption("Total cumulative loss by scenario, C$m: compare what-if runs side by side.")

        st.subheader("Downloads")
        c = st.columns(4)
        x = od / "credit_loss_results.xlsx"
        if x.exists():
            c[0].download_button("Excel results", x.read_bytes(), x.name, width="stretch")
        c[1].download_button("Configuration (YAML)", yaml.safe_dump(raw, sort_keys=False), f"{st.session_state.name}.yaml",
                             width="stretch")
        buf = io.StringIO()
        res["paths"].to_csv(buf, index=False)
        c[2].download_button("Quarterly paths (CSV)", buf.getvalue(), "quarterly_paths.csv", width="stretch")
        for rep_f in [od / "eqb_public_report.html", od / "methodology_report.html",
                      out_dir(raw) / "eqb_public_report.html", out_dir(raw) / "methodology_report.html"]:
            if rep_f.exists():
                c[3].download_button("Report (HTML)", rep_f.read_bytes(), rep_f.name, width="stretch")
                break
        with st.expander("Static charts from the run"):
            for fig in sorted(od.glob("fig*.png")):
                st.image(str(fig), width="stretch")

# ================================================================== ⑥ IFRS 9
with t6:
    res = st.session_state.get("res")
    if res is None:
        res, _ = load_results(raw)
    if res is None or "ifrs9" not in res:
        st.info("Run the model in ④ to see the IFRS 9 tables (results from before the IFRS 9 engine lack them).")
    else:
        d = res["ifrs9"]
        segs = list(res["summary"].segment.unique())
        H = int(res["summary"].horizon_q.max())
        st.subheader("Expected credit loss at the reporting date (jump-off)")
        st.caption("Probability-weighted across scenarios (IFRS 9 5.5.17). C$m; coverage = allowance / gross carrying "
                   "amount.")
        pw, g0, cov = d["pw_by_stage"], d["gca_jumpoff"], d["coverage"]
        tb = pd.DataFrame({
            "Gross loans S1": g0.gca_s1 / 1e6, "S2": g0.gca_s2 / 1e6, "S3": g0.gca_s3 / 1e6,
            "ECL S1": pw.ecl_stage1 / 1e6, "S2 ": pw.ecl_stage2 / 1e6, "S3 ": pw.ecl_stage3 / 1e6,
            "Overlay": pw.overlay / 1e6, "Total ACL": pw.allowance_total / 1e6,
            "Coverage S1 %": cov["Stage 1"], "S2 %": cov["Stage 2"], "S3 %": cov["Stage 3"],
            "Reported S1+2": [(raw["segments"].get(s, {}).get("reported_acl") or np.nan) / 1e6 for s in pw.index]})
        tb.loc["Total"] = tb.sum(numeric_only=True)
        tb.index = tb.index.map(nice)
        for c in ["Coverage S1 %", "S2 %", "S3 %"]:
            tb.loc["Total", c] = np.nan
        tb = tb.astype(float)
        fmt = {c: ("{:.2f}" if c in ("Coverage S1 %", "S2 %", "S3 %") else "{:,.1f}") for c in tb.columns}
        st.dataframe(tb.style.format(fmt, na_rep="–"), width="stretch")
        ov = {s: raw["segments"][s].get("overlay_rationale") for s in raw["segments"]
              if raw["segments"][s].get("overlay_rationale")}
        for s, why in ov.items():
            st.caption(f"Overlay, **{s}**: {why}")

        c1 = c2 = st.container()
        with c1:
            st.subheader("Sensitivity to forward-looking information")
            sens = d["sensitivity"]
            keep = [c for c in raw["scenarios"] if c in sens.columns] + ["probability_weighted"]
            x = (sens[keep] / 1e6).rename(columns={"probability_weighted": "probability-weighted"})
            x.loc["Total"] = x.sum()
            x.index = x.index.map(nice)
            st.dataframe(x.style.format("{:,.1f}"), width="stretch")
            st.caption("Total ACL, C$m, if each scenario had 100% weight, against the probability-weighted ACL "
                       "(the disclosure most Canadian banks give for their downside scenario).")
        with c2:
            st.subheader("Sensitivity to staging")
            sg = d["staging"][["all_stage1", "reported_staging", "all_stage2"]] / 1e6
            sg.loc["Total"] = sg.sum()
            sg.index = sg.index.map(nice)
            st.dataframe(sg.rename(columns={"all_stage1": "all performing in Stage 1", "reported_staging": "as staged",
                                            "all_stage2": "all performing in Stage 2"}).style.format("{:,.1f}"),
                         width="stretch")
            st.caption("Total ACL, C$m: what the 12-month vs lifetime choice is worth.")

        st.subheader("Stage migration over the projection")
        c1, c2 = st.columns([1, 3])
        seg = c1.selectbox("Segment", ["All"] + segs, key="i9_seg", format_func=lambda x: "All segments" if x == "All" else nice(x))
        sname = c1.selectbox("Scenario", [s for s in raw["scenarios"] if s in set(res["paths"].scenario)],
                             index=list(raw["scenarios"]).index("severe") if "severe" in raw["scenarios"] else 0, key="i9_sc")
        P = res["paths"][res["paths"].scenario == sname]
        if seg != "All":
            P = P[P.segment == seg]
        G = P.groupby("date")[["gca_s1", "gca_s2", "gca_s3"]].sum().reset_index().melt("date", var_name="stage")
        G["stage"] = G.stage.map({"gca_s1": "Stage 1", "gca_s2": "Stage 2", "gca_s3": "Stage 3"})
        G["value"] /= 1e6
        c2.altair_chart(alt.Chart(G).mark_area().encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")),
            y=alt.Y("value:Q", stack="normalize", title="share of gross loans", axis=alt.Axis(format="%")),
            color=alt.Color("stage:N", scale=alt.Scale(domain=["Stage 1", "Stage 2", "Stage 3"],
                                                        range=["#9ec5ee", "#e9a23b", "#c0392b"]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["stage", alt.Tooltip("date:T"), alt.Tooltip("value:Q", format=",.0f", title="C$m")]).properties(height=260),
            width="stretch")

        st.subheader(f"Allowance roll-forward, jump-off to quarter {H} (IFRS 7.35H)")
        rf = d["rollforward"]
        gc = d["gca"]
        pick = (lambda df: df[df.scenario == sname].groupby("line", sort=False)[["Stage 1", "Stage 2", "Stage 3", "Total"]].sum()) \
            if seg == "All" else (lambda df: df[(df.scenario == sname) & (df.segment == seg)].set_index("line")[
                ["Stage 1", "Stage 2", "Stage 3", "Total"]])
        st.dataframe((pick(rf) / 1e6).style.format("{:,.1f}"), width="stretch")
        st.caption("Opening = probability-weighted ACL at jump-off. Transfers are measured at the opening coverage of the "
                   "stage loans leave; remeasurement after transfer (e.g. 12-month to lifetime ECL) is in net "
                   "remeasurement. Write-offs use the allowance held; any shortfall goes through provisions.")
        with st.expander("Gross carrying amount roll-forward (IFRS 7.35I)"):
            st.dataframe((pick(gc) / 1e6).style.format("{:,.0f}"), width="stretch")

        with st.expander("How the model meets IFRS 9"):
            st.markdown("""
| IFRS 9 requirement | Implementation |
|---|---|
| Three stages; 12-month vs lifetime ECL (5.5.3, 5.5.5) | Gross loans tracked by stage each quarter; Stage 1 measured over 12 months, Stage 2 and 3 over the lifetime |
| Significant increase in credit risk, relative to origination (5.5.9, B5.5.7) | Borrower PD drift since origination is lognormal, calibrated to the reported Stage 2 share; a loan moves to Stage 2 when its PD exceeds the trigger multiple of its origination PD |
| 30 days past due backstop (5.5.11) | Optional floor on the Stage 2 share per segment |
| Credit-impaired / default, 90 days past due (Appendix A, B5.5.37) | Stage 3 = defaulted; lifetime ECL with PD = 1 |
| Cures and probation | Stage 2 → 1 at a quarterly cure rate; Stage 3 cures return to Stage 2 |
| Write-offs (5.4.4) | Stage 3 resolved after the workout period: the loss written off, the rest recovered |
| Unbiased, probability-weighted, multiple scenarios (5.5.17(a), B5.5.42) | Four scenarios with configurable weights; sensitivity to 100% weights disclosed |
| Reasonable and supportable forward-looking information (5.5.17(c), B5.5.13) | Scenario PDs over a configurable period, then linear reversion to the long-run average |
| Time value of money; effective interest rate (5.5.17(b), B5.5.44) | Expected shortfalls discounted at each segment's EIR; Stage 3 at the midpoint of the workout |
| Maximum contractual period; revolving exception (5.5.19-20, B5.5.40) | Remaining contractual term (to renewal for mortgages); behavioural life for cards with the undrawn limit through a CCF |
| Collateral (B5.5.55) | Mortgage LGD from indexed LTV at liquidation, net of selling costs and distressed-sale discount |
| Post-model adjustments | Overlays per segment, in % or C$, with rationale, reported separately |
| Disclosures (IFRS 7.35H-35M) | Allowance and gross-loan roll-forwards by stage, coverage ratios, scenario and staging sensitivity |
""")
