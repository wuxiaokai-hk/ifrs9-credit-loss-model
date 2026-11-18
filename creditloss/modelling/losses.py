"""Expected-loss assembly, IFRS 9 staging and provisions.

Per segment and projection quarter t:

    defaults_t   = PDq_t x EAD_{t-1}                         (BoC TDSA eq. 21-22; Fed expected-loss framework)
    credit loss_t = defaults_t x (1 - cure_t) x LGD_t
    PCL_t        = credit loss_t + change in performing allowance (Stage 1 12m ECL + Stage 2 lifetime ECL)

Balance-sheet convention (config `ead.balance_sheet`):
    static   balances held constant, defaults replaced by like-for-like lending (EBA static balance sheet;
             Fed 2025 test assumed an unchanged balance sheet)
    runoff   scheduled amortization + prepayment, no new lending
    growth   use quarterly growth rates from a balance-forecast file
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import ifrs9 as I9
from . import lgd as lgdm


def annual_to_quarterly(pd_annual: pd.Series) -> pd.Series:
    return 1 - (1 - pd_annual.clip(0, 0.999)) ** 0.25


def extend_pd(pd_path: pd.Series, pd_long_run: float, need: int, revert_q: int = 8) -> pd.Series:
    """Extend a projected annual PD path past the scenario by linear reversion to the long-run mean."""
    if len(pd_path) >= need:
        return pd_path.iloc[:need].reset_index(drop=True)
    last = pd_path.iloc[-1]
    ext = [last + (pd_long_run - last) * min((j + 1) / revert_q, 1.0) for j in range(need - len(pd_path))]
    return pd.concat([pd_path.reset_index(drop=True), pd.Series(ext)], ignore_index=True)


def extend_series(s: pd.Series, need: int) -> pd.Series:
    s = s.reset_index(drop=True)
    if len(s) >= need:
        return s.iloc[:need]
    return pd.concat([s, pd.Series([s.iloc[-1]] * (need - len(s)))], ignore_index=True)


def stage2_share(pd_ratio: np.ndarray, s2_0: float, sicr_multiple: float, sigma: float = 0.9) -> np.ndarray:
    """Share of the performing book past the SICR trigger (PD > multiple x origination PD).

    Borrower PD drift since origination is lognormal; mu is calibrated so the share
    at jump-off equals the reported Stage 2 share. A scenario that raises PDs by a
    factor r shifts every borrower's ratio by r.
    """
    s2_0 = float(np.clip(s2_0, 1e-4, 0.999))
    mu = np.log(sicr_multiple) - sigma * norm.ppf(1 - s2_0)
    return 1 - norm.cdf((np.log(sicr_multiple) - np.log(np.clip(pd_ratio, 1e-6, None)) - mu) / sigma)


def project_segment(seg: str, seg_cfg: dict, book: pd.DataFrame, pd_jump: float, pd_scen: pd.Series,
                    hpi_jump: float, hpi_scen: pd.Series, cfg: dict, pd_long_run: float,
                    growth: pd.Series | None = None) -> pd.DataFrame:
    """Quarterly loss and provision path for one segment under one scenario.

    Position 0 is the jump-off quarter; positions 1..H are scenario quarters.
    pd_jump / pd_scen  annual default rate (fraction) at jump-off and over the scenario
    hpi_jump / hpi_scen  house price index level at jump-off and over the scenario
    """
    H = cfg["horizon_quarters"]
    life_y = seg_cfg.get("lifetime_years") if seg_cfg.get("ead_model") == "revolving" else seg_cfg.get("contractual_term_years")
    life_q = int(4 * float(life_y if life_y is not None else cfg["ifrs9"]["lifetime_years"]))
    need = H + life_q + 8
    bs = cfg.get("ead", {}).get("balance_sheet", "static")
    dates = [pd_scen.index[0] - pd.offsets.QuarterEnd(1)] + list(pd_scen.index[:H])

    ifrs = cfg["ifrs9"]
    rs = ifrs.get("reasonable_supportable_quarters")      # forward-looking period, then reversion (B5.5.13)
    scen_pd = pd_scen.reset_index(drop=True)
    if rs:
        scen_pd = scen_pd.iloc[:max(int(rs), H)]
    pdA = extend_pd(pd.concat([pd.Series([pd_jump]), scen_pd], ignore_index=True), pd_long_run, need,
                    int(ifrs.get("reversion_quarters", 8)))
    pdQ = annual_to_quarterly(pdA)
    hpi_rel = extend_series(pd.concat([pd.Series([1.0]), (hpi_scen / hpi_jump).reset_index(drop=True)],
                                      ignore_index=True), need)

    drawn0 = book["balance"].sum()
    undrawn0 = book["undrawn"].sum()
    ccf = seg_cfg.get("ccf", 0.0)
    amort = 1 / (4 * float(book["remaining_amort_years"].mean()))
    prepay = float(book["prepay_rate"].mean()) / 4

    # --- balances (EAD) ----------------------------------------------------
    bal = np.empty(need)
    b = drawn0
    for j in range(need):
        if j > 0:
            if bs == "runoff" and seg_cfg.get("ead_model") == "amortizing":
                b = b * (1 - amort - prepay)
            elif bs == "growth" and growth is not None and j - 1 < len(growth):
                b = b * (1 + growth.iloc[j - 1])
        bal[j] = b
    bal = pd.Series(bal)
    # Scheduled (annuity) amortization of *existing* loans lowers their LTV even on a static
    # book; prepayments remove loans but do not change the LTV of the loans that stay.
    rq = seg_cfg.get("contract_rate", 0.045) / 4
    nq = 4 * float(book["remaining_amort_years"].mean())
    bal_rel_existing = pd.Series([((1 + rq) ** nq - (1 + rq) ** k) / ((1 + rq) ** nq - 1) for k in range(need)])
    ead = bal + ccf * undrawn0 * (bal / drawn0)

    # --- LGD ---------------------------------------------------------------
    kind = seg_cfg["lgd_model"]
    if kind == "structural_ltv":
        lgd, by_bucket = lgdm.structural_ltv(book, hpi_rel, bal_rel_existing, seg_cfg)
    elif kind == "insured":
        lgd, by_bucket = lgdm.insured(book, hpi_rel, bal_rel_existing, seg_cfg)
    elif kind == "frye_jacobs":
        lgd = lgdm.frye_jacobs(pdA, pd_long_run, seg_cfg["fixed_lgd"], seg_cfg.get("asset_corr", 0.15))
    else:
        lgd = lgdm.fixed(pdA.index, seg_cfg["fixed_lgd"])
    lgd = lgd.reset_index(drop=True)

    # cures fall when defaults rise (stressed borrowers cure less)
    cure0 = seg_cfg.get("cure_rate", 0.0)
    cure = (cure0 * (pd_long_run / pdA).clip(upper=1.5) ** seg_cfg.get("cure_elasticity", 0.3)).clip(0, 0.9)
    loss_given_default = (1 - cure) * lgd                                  # loss per unit defaulted
    el_rate_q = pdQ * loss_given_default                                   # expected loss per unit EAD

    # --- IFRS 9: three-stage engine ----------------------------------------
    s2_0 = float(np.average(book["stage2_share"].fillna(book["stage2_share"].mean()).fillna(0.05), weights=book["balance"]))
    s3_0 = float(np.average(book["stage3_share"].fillna(0.0), weights=book["balance"]))
    ratio = (pdA / max(pd_jump, 1e-9)).to_numpy()
    s2_target = stage2_share(ratio, s2_0, ifrs["sicr_multiple"])
    st_ = I9.run_stages(need=need, H=H, pdQ=pdQ, lgd_star=loss_given_default, cure=cure, s2_target=s2_target,
                        gca0=drawn0, ead_factor=float(ead.iloc[0] / max(drawn0, 1e-12)), s2_0=s2_0, s3_0=s3_0,
                        seg_cfg=seg_cfg, ifrs=ifrs, bal_rel_existing=bal_rel_existing,
                        revolving=seg_cfg.get("ead_model") == "revolving", repay_q=amort + prepay,
                        balance_sheet=bs, bal_path=bal)

    rows = []
    for j in range(H + 1):  # j=0 is jump-off
        dflt = 0.0 if j == 0 else pdQ.iloc[j] * ead.iloc[j - 1]
        rows.append({"date": dates[j], "q": j, "pd_annual": pdA.iloc[j], "lgd": lgd.iloc[j], "cure": cure.iloc[j],
                     "ead": ead.iloc[j], "defaults": dflt, "credit_loss": dflt * loss_given_default.iloc[j],
                     "allowance_fed4q": ead.iloc[j] * (1 - s3_0) * float(el_rate_q.iloc[j + 1:j + 5].sum())})
    out = pd.DataFrame(rows).merge(st_.drop(columns=["stage2_share"]).assign(stage2_share=st_.stage2_share), on="q")
    out["segment"] = seg
    return out


def summarize(paths: pd.DataFrame, horizons=(9, 13)) -> pd.DataFrame:
    """Cumulative loss rates over the horizon, % of average EAD (Fed reporting convention)."""
    rows = []
    for (scen, seg), g in paths.groupby(["scenario", "segment"]):
        g = g[g["q"] > 0]
        for h in horizons:
            gh = g[g["q"] <= h]
            rows.append({"scenario": scen, "segment": seg, "horizon_q": h,
                         "cum_loss": gh["credit_loss"].sum(), "cum_pcl": gh["pcl"].sum(),
                         "avg_ead": gh["ead"].mean(),
                         "loss_rate_pct": 100 * gh["credit_loss"].sum() / gh["ead"].mean(),
                         "pcl_rate_pct": 100 * gh["pcl"].sum() / gh["ead"].mean(),
                         "peak_pd_pct": 100 * gh["pd_annual"].max(), "peak_lgd_pct": 100 * gh["lgd"].max(),
                         "peak_stage2_pct": 100 * gh["stage2_share"].max(),
                         "cum_net_writeoffs": gh["net_writeoffs"].sum() if "net_writeoffs" in gh else np.nan,
                         "closing_acl": gh["allowance_total"].iloc[-1] if "allowance_total" in gh else np.nan})
    return pd.DataFrame(rows)
