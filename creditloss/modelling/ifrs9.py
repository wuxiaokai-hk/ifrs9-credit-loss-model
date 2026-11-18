"""IFRS 9 expected credit loss: three-stage balance engine, ECL measurement and disclosure flows.

Per segment and scenario, quarter by quarter, the gross carrying amount (GCA) moves between stages:

    Stage 1 -> Stage 2   significant increase in credit risk (SICR): relative PD test against origination
                         (IFRS 9 5.5.3, B5.5.7) calibrated to the reported Stage 2 share, with the 30-days-past-due
                         backstop (5.5.11) as a floor on the Stage 2 share
    Stage 2 -> Stage 1   cure after a probation period (OSFI IFRS 9 guidance; B5.5.27)
    Stage 1/2 -> Stage 3 default = credit-impaired (Appendix A; 90 days past due presumption, B5.5.37)
    Stage 3 -> Stage 2   cure (no loss); Stage 3 -> out: resolution after the workout period, with write-off
                         of the loss (5.4.4) and recovery of the rest
    repayments           scheduled amortization + prepayment (derecognition)
    new originations     enter Stage 1 at their 12-month ECL (static balance sheet: replace what left the book)

ECL measured at each quarter-end (5.5.17-5.5.20, B5.5.44):
    Stage 1   12-month ECL  = sum_{q=1..4}  S(q-1) PD1_q LGD*_q EADf_q DF_q
    Stage 2   lifetime ECL  = sum_{q=1..L}  S(q-1) PD2_q LGD*_q EADf_q DF_q
    Stage 3   lifetime ECL with PD = 1: GCA x LGD* x DF(workout / 2)
    S = survival, LGD* = (1 - cure) x LGD, EADf = remaining contractual balance per unit today (amortizing) or
    drawn + CCF x undrawn (revolving), DF = (1 + EIR/4)^-q discounting at the effective interest rate (B5.5.44).
    L = remaining contractual term (5.5.19), or behavioural life for revolving facilities (5.5.20, B5.5.40).
    Forward-looking PDs follow the scenario over the reasonable and supportable period and revert to the
    long-run average after it (B5.5.13-B5.5.14 style).

PCL_t = change in ACL_t + net write-offs_t. Management overlays (post-model adjustments) are added on top
of the modelled ECL and reported separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def survival_ecl(pd_q: np.ndarray, lgd_star: np.ndarray, ead_f: np.ndarray, df: np.ndarray) -> float:
    """sum_q S(q-1) * PD_q * LGD*_q * EADf_q * DF_q  (marginal-default form)."""
    pd_q = np.clip(pd_q, 0, 1)
    surv = np.cumprod(np.r_[1.0, 1 - pd_q[:-1]])
    return float(np.sum(surv * pd_q * lgd_star * ead_f * df))


def run_stages(*, need: int, H: int, pdQ: pd.Series, lgd_star: pd.Series, cure: pd.Series, s2_target: np.ndarray,
               gca0: float, ead_factor: float, s2_0: float, s3_0: float, seg_cfg: dict, ifrs: dict,
               bal_rel_existing: pd.Series, revolving: bool, repay_q: float, balance_sheet: str,
               bal_path: pd.Series) -> pd.DataFrame:
    k2 = float(seg_cfg.get("stage2_pd_multiple", ifrs.get("stage2_pd_multiple", 3.0)))
    probation_cure = float(seg_cfg.get("stage2_cure_q", ifrs.get("stage2_cure_q", 0.10)))
    W = max(1, int(seg_cfg.get("workout_quarters", ifrs.get("workout_quarters", 4))))
    eir = float(seg_cfg.get("eir", seg_cfg.get("contract_rate", ifrs.get("default_eir", 0.05))))
    disc = bool(ifrs.get("discount", True))
    life_years = seg_cfg.get("lifetime_years") if revolving else seg_cfg.get("contractual_term_years")
    L = int(4 * float(life_years if life_years is not None else ifrs.get("lifetime_years", 5)))
    floor30 = float(seg_cfg.get("dpd30_share", 0.0))           # 30 dpd backstop: these loans are Stage 2 regardless
    ov_pct = float(seg_cfg.get("overlay_pct", 0.0))
    ov_amt = float(seg_cfg.get("overlay_amount", 0.0))

    qs = np.arange(1, need + 1)
    DF = (1 + eir / 4) ** (-qs) if disc else np.ones(need)
    if revolving:
        amort_f = np.ones(need)
    else:  # remaining contractual balance of today's loans, per unit outstanding today
        b = bal_rel_existing.to_numpy()
        amort_f = np.r_[b[1:], b[-1]] / np.maximum(b[0], 1e-12)
    pq, ls = pdQ.to_numpy(), lgd_star.to_numpy()

    def unit_ecl(j: int, horizon: int, mult: float) -> float:
        sl = slice(j + 1, min(j + 1 + horizon, need))
        n = sl.stop - sl.start
        return ead_factor * survival_ecl(pq[sl] * mult, ls[sl], amort_f[:n], DF[:n])

    def scales(s2: float) -> float:   # Stage 1 PD so that the book-average PD is unchanged by the stage mix
        return 1.0 / ((1 - s2) + k2 * s2)

    G1 = gca0 * (1 - s3_0) * (1 - s2_0)
    G2 = gca0 * (1 - s3_0) * s2_0
    G3 = gca0 * s3_0
    rows = []
    prev = None
    for j in range(H + 1):
        flows = dict(orig=0.0, repay1=0.0, repay2=0.0, s1_s2=0.0, s2_s1=0.0, s1_s3=0.0, s2_s3=0.0, s3_s2=0.0,
                     resolved=0.0, writeoff=0.0, recovery=0.0)
        if j > 0:
            s2_prev = G2 / max(G1 + G2, 1e-12)
            c1 = scales(s2_prev)
            pd1, pd2 = pq[j] * c1, pq[j] * k2 * c1
            flows["s1_s3"], flows["s2_s3"] = G1 * pd1, G2 * pd2
            exits = G3 / W                                      # Stage 3 resolved each quarter
            cur = float(cure.iloc[j])
            flows["s3_s2"] = exits * cur
            flows["resolved"] = exits * (1 - cur)
            lgd_j = float(lgd_star.iloc[j] / max(1 - cur, 1e-9))
            flows["writeoff"] = flows["resolved"] * lgd_j
            flows["recovery"] = flows["resolved"] * (1 - lgd_j)
            r = 0.0 if revolving else repay_q
            flows["repay1"], flows["repay2"] = (G1 - flows["s1_s3"]) * r, (G2 - flows["s2_s3"]) * r
            base_cure = (G2 - flows["s2_s3"] - flows["repay2"]) * probation_cure
            G1n = G1 - flows["s1_s3"] - flows["repay1"]
            G2n = G2 - flows["s2_s3"] - flows["repay2"] + flows["s3_s2"]
            G3n = G3 + flows["s1_s3"] + flows["s2_s3"] - exits
            if balance_sheet == "static":
                flows["orig"] = flows["repay1"] + flows["repay2"] + flows["resolved"]
            elif balance_sheet == "growth":
                flows["orig"] = max(bal_path.iloc[j] - (G1n + G2n + G3n), 0.0)
            G1n += flows["orig"]
            # SICR migration: hit the Stage 2 share implied by the relative PD test (and the 30 dpd floor)
            target = max(float(s2_target[j]), floor30)
            need_s2 = target * (G1n + G2n) - G2n
            mig = need_s2 + base_cure
            if mig >= 0:
                flows["s1_s2"], flows["s2_s1"] = mig, base_cure
            else:
                flows["s1_s2"], flows["s2_s1"] = 0.0, base_cure - mig
            flows["s1_s2"] = min(flows["s1_s2"], G1n)
            flows["s2_s1"] = min(flows["s2_s1"], G2n)
            G1 = G1n - flows["s1_s2"] + flows["s2_s1"]
            G2 = G2n + flows["s1_s2"] - flows["s2_s1"]
            G3 = G3n
        s2 = G2 / max(G1 + G2, 1e-12)
        c1 = scales(s2)
        u1, u2 = unit_ecl(j, 4, c1), unit_ecl(j, L, k2 * c1)
        u3 = float(lgd_star.iloc[j]) * (DF[max(W // 2 - 1, 0)] if disc else 1.0)
        e1, e2, e3 = G1 * u1, G2 * u2, G3 * u3
        model = e1 + e2 + e3
        overlay = ov_pct * model + ov_amt
        row = dict(q=j, gca_s1=G1, gca_s2=G2, gca_s3=G3, stage2_share=s2, ecl_stage1=e1, ecl_stage2=e2,
                   ecl_stage3=e3, overlay=overlay, allowance=e1 + e2, allowance_total=model + overlay,
                   cov_s1=u1, cov_s2=u2, cov_s3=u3, net_writeoffs=flows["writeoff"], **{f"f_{k}": v for k, v in flows.items()})
        row["pcl"] = 0.0 if prev is None else row["allowance_total"] - prev["allowance_total"] + flows["writeoff"]
        # allowance roll-forward (IFRS 7.35H): transfers at the origin stage's opening coverage
        if prev is not None:
            o1, o2, o3 = prev["cov_s1"], prev["cov_s2"], prev["cov_s3"]
            row.update({
                "rf_s1_new": flows["orig"] * u1,
                "rf_s1_derecognised": -flows["repay1"] * o1,
                "rf_s2_derecognised": -flows["repay2"] * o2,
                "rf_s1_to_s2": flows["s1_s2"] * o1, "rf_s2_to_s1": flows["s2_s1"] * o2,
                "rf_s1_to_s3": flows["s1_s3"] * o1, "rf_s2_to_s3": flows["s2_s3"] * o2,
                "rf_s3_to_s2": flows["s3_s2"] * o3,
                "rf_s3_writeoff": -min(flows["resolved"] * o3, flows["writeoff"]) if o3 > 0 else 0.0,
                "rf_s3_resolved_release": -(flows["resolved"] * o3 - min(flows["resolved"] * o3, flows["writeoff"])),
            })
        prev = row
        rows.append(row)
    return pd.DataFrame(rows)


STAGE_COLS = {1: "ecl_stage1", 2: "ecl_stage2", 3: "ecl_stage3"}


def allowance_rollforward(path: pd.DataFrame, q_from: int = 0, q_to: int | None = None,
                          opening: pd.Series | None = None) -> pd.DataFrame:
    """IFRS 7.35H reconciliation of the loss allowance by stage between two quarter-ends (one segment,
    one scenario). Remeasurement is the residual after the identified movements. `opening` (ecl_stage1..3)
    replaces the scenario's own jump-off allowance, e.g. with the probability-weighted one; the difference
    is shown as the day-one scenario remeasurement."""
    q_to = int(path.q.max()) if q_to is None else q_to
    g = path[(path.q > q_from) & (path.q <= q_to)]
    o = path[path.q == q_from].iloc[0]
    c = path[path.q == q_to].iloc[0]
    s = g.sum(numeric_only=True)
    own = [o.ecl_stage1, o.ecl_stage2, o.ecl_stage3]
    op = own if opening is None else [opening.ecl_stage1, opening.ecl_stage2, opening.ecl_stage3]
    lines = {
        "Opening allowance": op,
        "Day-one move to the scenario's forward-looking ECL": [a - b for a, b in zip(own, op)],
        "Transfer to Stage 1": [s.rf_s2_to_s1, -s.rf_s2_to_s1, 0.0],
        "Transfer to Stage 2": [-s.rf_s1_to_s2, s.rf_s1_to_s2 + s.rf_s3_to_s2, -s.rf_s3_to_s2],
        "Transfer to Stage 3": [-s.rf_s1_to_s3, -s.rf_s2_to_s3, s.rf_s1_to_s3 + s.rf_s2_to_s3],
        "New originations": [s.rf_s1_new, 0.0, 0.0],
        "Derecognitions and repayments": [s.rf_s1_derecognised, s.rf_s2_derecognised, s.rf_s3_resolved_release],
        "Write-offs": [0.0, 0.0, s.rf_s3_writeoff],
    }
    df = pd.DataFrame(lines, index=["Stage 1", "Stage 2", "Stage 3"]).T
    closing = pd.Series([c.ecl_stage1, c.ecl_stage2, c.ecl_stage3], index=df.columns)
    df.loc["Net remeasurement (PD, LGD, forward-looking information)"] = closing - df.sum()
    df.loc["Closing allowance"] = closing
    df["Total"] = df.sum(axis=1)
    if opening is None:
        df = df.drop(index="Day-one move to the scenario's forward-looking ECL")
    return df


def gca_rollforward(path: pd.DataFrame, q_from: int = 0, q_to: int | None = None) -> pd.DataFrame:
    """Gross carrying amount by stage between two quarter-ends (IFRS 7.35I)."""
    q_to = int(path.q.max()) if q_to is None else q_to
    g = path[(path.q > q_from) & (path.q <= q_to)]
    o = path[path.q == q_from].iloc[0]
    c = path[path.q == q_to].iloc[0]
    s = g.sum(numeric_only=True)
    df = pd.DataFrame({
        "Opening gross carrying amount": [o.gca_s1, o.gca_s2, o.gca_s3],
        "Transfer to Stage 1": [s.f_s2_s1, -s.f_s2_s1, 0.0],
        "Transfer to Stage 2": [-s.f_s1_s2, s.f_s1_s2 + s.f_s3_s2, -s.f_s3_s2],
        "Transfer to Stage 3": [-s.f_s1_s3, -s.f_s2_s3, s.f_s1_s3 + s.f_s2_s3],
        "New originations": [s.f_orig, 0.0, 0.0],
        "Repayments": [-s.f_repay1, -s.f_repay2, 0.0],
        "Recoveries on resolution": [0.0, 0.0, -s.f_recovery],
        "Write-offs": [0.0, 0.0, -s.f_writeoff],
    }, index=["Stage 1", "Stage 2", "Stage 3"]).T
    closing = pd.Series([c.gca_s1, c.gca_s2, c.gca_s3], index=df.columns)
    df.loc["Other (rounding)"] = closing - df.sum()
    df.loc["Closing gross carrying amount"] = closing
    df["Total"] = df.sum(axis=1)
    return df
