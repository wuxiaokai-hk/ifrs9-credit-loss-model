"""Full-chain backtest against EQB's reported provisions for credit losses (PCL).

For each fiscal year, start the model at the quarter before the year begins and run the whole chain over the
year with the macro data that actually happened: the PD equations (dynamic, from the proxy rate observed at the
start), the bank calibration (PD_bank = proxy x today's scale), the house-price path for LGD (re-based to the
start of the year), and the IFRS 9 stage engine. The book is today's segment mix, excluding credit cards (bought in
July 2026, so absent from the history), scaled to the year's reported gross loans. The model's credit losses for the
year are compared with reported PCL on a like-for-like basis: the engine's PCL is write-offs plus the change in
the stage 1-3 allowance over the year, as in the income statement.

Two model columns separate the two sources of error:
    model (macro)  the PD equations forecast the proxy from actual macro data: the whole chain
    model (proxy)  the actual proxy path is used: only the bank mapping, LGD and engine are tested

Caveats. The PD equations are estimated on the full sample, so this is an in-sample test of the chain, not a
forecast test (the out-of-sample tests are in the panel module). PCL includes allowance builds and releases as
well as write-offs, and EQB's book changed over the period (more commercial and decumulation lending), which
today's segment mix does not capture. FY2023 is the ten-month year after EQB moved its year-end to October.
"""
from __future__ import annotations

import pandas as pd

from creditloss.data import features as F
from creditloss.modelling import losses as L
from creditloss.modelling.pd_satellite import dynamic_path
from creditloss.scenarios.run import calibration

# Reported figures, C$m (EQB annual reports and MD&A). Gross loans at year-end; FY2023 interpolated.
PCL = pd.DataFrame([
    ("FY2020", "2019-12-31", 1.0, 42.28, 27779),
    ("FY2021", "2020-12-31", 1.0, -7.67, 32217),
    ("FY2022", "2021-12-31", 1.0, 37.26, 45344),
    ("FY2023", "2022-12-31", 10 / 12, 46.63, (45344 + 46003) / 2),
    ("FY2024", "2023-09-30", 1.0, 107.01, 46003),
    ("FY2025", "2024-09-30", 1.0, 137.43, 45359),
], columns=["year", "jumpoff", "year_fraction", "reported_PCL", "gross_loans"])

SEGMENTS = ["mortgage_uninsured", "mortgage_insured", "commercial"]


def _spec(ctx, seg):
    sel, name = ctx.res["sels"][seg], ctx.res["champions"][seg]
    return max(sel["bma_specs"], key=lambda t: t[1])[0] if name == "BMA" and sel.get("bma_specs") else sel["specs"][name]


def _year_loss(ctx, seg, t0, q_idx, proxy, scale_book):
    sc = ctx.cfg["segments"][seg]
    pd_now, pd_lr, scale = calibration(sc, ctx.port[seg][ctx.dcol])
    bk = ctx.snap[ctx.snap.segment == seg].copy()
    bk["balance"] *= scale_book
    bk["undrawn"] *= scale_book
    pd0 = float(proxy.iloc[0]) * scale
    hpi = ctx.macro[ctx.cidx]
    cfg = {**ctx.cfg, "horizon_quarters": len(q_idx)}
    path = L.project_segment(seg, sc, bk, pd0, (proxy.iloc[1:] * scale).clip(upper=0.99), float(hpi.loc[t0]),
                             hpi.loc[q_idx], cfg, pd_lr)
    g = path[(path.q > 0) & (path.q <= len(q_idx))]
    return float(g["pcl"].sum()), float(g["credit_loss"].sum())


def backtest(ctx) -> pd.DataFrame:
    book_now = ctx.snap[ctx.snap.segment.isin(SEGMENTS)].balance.sum()
    rows = []
    for r in PCL.itertuples():
        t0 = pd.Timestamp(r.jumpoff)
        q_idx = ctx.macro.index[ctx.macro.index > t0][:4]
        scale_book = r.gross_loans * 1e6 / book_now
        out = {"year": r.year, "reported_PCL": r.reported_PCL, "gross_loans": r.gross_loans}
        m_macro = m_proxy = cl_macro = 0.0
        for seg in SEGMENTS:
            sc = ctx.cfg["segments"][seg]
            sel = ctx.res["sels"][seg]
            y = sel["y"]
            sp = _spec(ctx, seg)
            X, _ = F.build_features(ctx.macro, ctx.lit, ctx.cfg, sc.get("sign_priors", "household"), sc.get("candidates"))
            z = dynamic_path(sp.params, sp.include_ar, X, float(y.loc[t0]), q_idx)
            fc = pd.Series(F.inverse_target(pd.concat([pd.Series([y.loc[t0]], index=[t0]), pd.Series(z, index=q_idx)]).to_numpy(), ctx.how) / 100,
                           index=[t0, *q_idx])
            act = pd.Series(F.inverse_target(y.loc[[t0, *q_idx]].to_numpy(), ctx.how) / 100, index=[t0, *q_idx])
            pcl, cl = _year_loss(ctx, seg, t0, q_idx, fc, scale_book)
            m_macro += pcl * r.year_fraction / 1e6
            cl_macro += cl * r.year_fraction / 1e6
            m_proxy += _year_loss(ctx, seg, t0, q_idx, act.ffill(), scale_book)[0] * r.year_fraction / 1e6
        out.update({"model_macro": m_macro, "model_proxy": m_proxy, "model_macro_credit_loss": cl_macro})
        rows.append(out)
    df = pd.DataFrame(rows)
    df["reported_PCL_bp"] = 1e4 * df.reported_PCL / df.gross_loans
    df["model_macro_bp"] = 1e4 * df.model_macro / df.gross_loans
    df["model_proxy_bp"] = 1e4 * df.model_proxy / df.gross_loans
    return df


def fit_stats(df: pd.DataFrame) -> dict:
    e = df.model_macro - df.reported_PCL
    return {"total_reported": df.reported_PCL.sum(), "total_model_macro": df.model_macro.sum(),
            "total_model_proxy": df.model_proxy.sum(), "mean_error": e.mean(), "rmse": (e ** 2).mean() ** 0.5,
            "corr_macro": df.model_macro.corr(df.reported_PCL), "corr_proxy": df.model_proxy.corr(df.reported_PCL)}
