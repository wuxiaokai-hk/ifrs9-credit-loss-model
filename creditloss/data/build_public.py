"""Build the public-data inputs from the raw public files in data/eqb_public/raw/.

Sources (all downloaded 2026-11-04, see data/eqb_public/SOURCES.md):
  CBA    Residential mortgages in arrears 90+ days, Ontario, monthly 1995-2026 (big-bank sample)
  FRED   Canada: unemployment rate (NSA, quarterly), BIS residential property prices,
         3-month interbank rate, 10-year GoC yield, real GDP growth (q/q)
  FRED   US: credit-card and business-loan charge-off rates (all commercial banks), unemployment (NSA)
  EQB    Q3 2026 report to shareholders (July 31, 2026) and Q2 2026 MD&A: loan book, ACL, impaired loans

Writes data/eqb_public/{macro_history, macro_scenarios, portfolio_history, portfolio_snapshot}.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from creditloss import PROJECT_ROOT as ROOT

RAW = ROOT / "data" / "eqb_public" / "raw"
OUT = ROOT / "data" / "eqb_public"
JUMP = pd.Timestamp("2026-06-30")          # jump-off quarter end (2026Q2)
COVID = ("2020-06-30", "2021-09-30")        # quarters whose macro readings are interpolated (see below)
H = 24                                       # scenario quarters


def q_end(s) -> pd.DatetimeIndex:
    return pd.to_datetime(s).dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()


def fred(name: str) -> pd.Series:
    d = pd.read_csv(RAW / f"fred_{name}.csv")
    d["date"] = q_end(d["date"])
    return d.set_index("date")["value"].astype(float)


def seasonal_adjust(s: pd.Series) -> pd.Series:
    """Remove stable quarterly seasonal factors (deviation from a centred 2x4 moving average,
    averaged by quarter, 2020-21 excluded). Adequate for an unemployment rate."""
    ma = s.rolling(4, center=True).mean().rolling(2).mean().shift(-1)
    dev = (s - ma).to_frame("d")
    dev["q"] = dev.index.quarter
    dev = dev[(dev.index < "2020-01-01") | (dev.index > "2021-12-31")]
    f = dev.groupby("q")["d"].mean()
    f -= f.mean()
    return s - s.index.quarter.map(f).to_numpy()


def covid_bridge(s: pd.Series) -> pd.Series:
    """Linear interpolation through 2020Q2-2021Q3. Income support and payment deferrals cut the
    link between the pandemic spike in unemployment and credit losses (arrears and charge-offs
    fell); feeding the raw spike to the satellites would bias the unemployment elasticity
    towards zero. Same treatment as the COVID adjustments in most bank IFRS 9 models."""
    s = s.copy()
    m = (s.index >= COVID[0]) & (s.index <= COVID[1])
    s[m] = np.nan
    return s.interpolate(limit_area="inside")


def build_macro() -> pd.DataFrame:
    ur_ca = covid_bridge(seasonal_adjust(fred("LRHUTTTTCAQ156N")))
    ur_us = covid_bridge(seasonal_adjust(fred("LRUN74TTUSQ156N")))
    hpi = fred("QCAN628BIS")
    r3m, r10 = fred("IR3TIB01CAQ156N"), fred("IRLTLT01CAQ156N")
    g = covid_bridge(fred("NAEXKP01CAQ657S"))
    gdp = 100 * np.exp(np.log1p(g / 100).cumsum())
    idx = pd.date_range("1990-03-31", JUMP, freq="QE")
    m = pd.DataFrame(index=idx)
    m["LBR_CA"] = ur_ca
    m["LBR_US"] = ur_us
    m["HPI_CA"] = hpi
    m["R3M"] = r3m
    m["R10Y"] = r10
    m["GDPR_CA"] = gdp
    # Gaps at the very end (BIS HPI ends 2026Q1, US unemployment ends 2025Q3): carry the last
    # reading forward to the jump-off. Flagged in the report.
    for c in ["HPI_CA", "LBR_US"]:
        last = m[c].last_valid_index()
        m.loc[m.index > last, c] = m.at[last, c]
    m.index.name = "date"
    return m


def build_portfolio() -> pd.DataFrame:
    cba = pd.read_csv(RAW / "cba_arrears_ontario.csv")
    cba["date"] = q_end(cba["month"] + "-01")
    cba = cba.groupby("date").last()                       # 2026-05 stands in for 2026Q2
    arrears = 100 * cba["arrears_count"] / cba["total_mortgages"]
    cards, biz = fred("CORCCACBS"), fred("CORBLACBS")
    rows = []
    proxies = {"mortgage_uninsured": arrears, "mortgage_insured": arrears,
               "credit_cards": cards, "commercial": biz}
    for seg, s in proxies.items():
        s = s[s.index <= JUMP]
        for d, v in s.items():
            rows.append({"date": d.date(), "segment": seg, "balance": np.nan, "undrawn": np.nan,
                         "default_rate": v, "nco_rate": np.nan, "lgd_realized": np.nan,
                         "insured_share": 1.0 if seg == "mortgage_insured" else 0.0})
    return pd.DataFrame(rows)


def build_snapshot() -> pd.DataFrame:
    """EQB Inc., July 31, 2026 (C$000s from the Q3 2026 report), mapped to model segments.

    Disclosed: balances by product, average uninsured single-family LTV (69%), Stage 1+2 ACL by
    loan type, Stage 2 share of gross loans (Apr 30, 2026: personal 20.4%, commercial 11.7%),
    gross impaired loans (personal 462.7m, commercial 570.1m at Apr 30, 2026).
    Assumed (not disclosed): the LTV distribution around the 69% mean, decumulation LTV,
    remaining amortization, prepayment, card limits. Marked 'assumed' in the report.
    """
    k = 1_000
    rows = [
        # segment, bucket, balance, current_ltv, undrawn, amort_yrs, prepay, stage2, stage3
        ("mortgage_uninsured", "SF 0-50", 0.10 * 21_217_462 * k, 0.45, 0, 23, 0.12, 0.204, 0.019),
        ("mortgage_uninsured", "SF 50-65", 0.20 * 21_217_462 * k, 0.60, 0, 23, 0.12, 0.204, 0.019),
        ("mortgage_uninsured", "SF 65-75", 0.35 * 21_217_462 * k, 0.71, 0, 23, 0.12, 0.204, 0.019),
        ("mortgage_uninsured", "SF 75-80", 0.35 * 21_217_462 * k, 0.78, 0, 23, 0.12, 0.204, 0.019),
        ("mortgage_uninsured", "Decumulation", 3_328_640 * k, 0.35, 0, 23, 0.12, 0.204, 0.019),
        ("mortgage_insured", "SF insured", 5_870_845 * k, 0.80, 0, 23, 0.10, 0.15, 0.016),
        ("mortgage_insured", "Multi-unit insured", 3_116_885 * k, 0.75, 0, 30, 0.05, 0.15, 0.016),
        ("credit_cards", "PC Financial cards", 4_523_886 * k, np.nan, 2.0 * 4_523_886 * k, 3, 0.0, 0.12, 0.0),
        ("commercial", "Commercial loans", 10_018_402 * k, 0.65, 0, 6, 0.15, 0.117, 0.053),
        ("commercial", "Equipment financing", 1_118_028 * k, 0.65, 0, 3, 0.10, 0.117, 0.053),
    ]
    cols = ["segment", "ltv_bucket", "balance", "current_ltv", "undrawn", "remaining_amort_years",
            "prepay_rate", "stage2_share", "stage3_share"]
    d = pd.DataFrame(rows, columns=cols)
    d["first_lien_ltv"] = 0.0
    return d


def build_scenarios(m: pd.DataFrame) -> pd.DataFrame:
    """Four paths from the 2026Q2 jump-off. Author-built (commercial vendor scenarios are licensed):
      baseline  unemployment eases slowly, house prices flat then +2%/yr, rates unchanged
      upside    unemployment -0.8pp, house prices +6% over two years
      downside  unemployment +2pp (peak q6), house prices -12% (trough q8), policy cut 100bp
      severe    unemployment +4.5pp (peak q8, ~11%), house prices -30% (trough q9), policy rate to 0.5%,
                10-year yield -100bp, GDP -5% peak to trough. Size matches the BoE 2025 test (-28% house
                prices) and Ontario's early-1990s downturn (unemployment ~11%, Toronto prices -27%).
    """
    last = m.iloc[-1]
    dates = pd.date_range(JUMP + pd.offsets.QuarterEnd(1), periods=H, freq="QE")
    h = np.arange(1, H + 1)

    def hump(peak, at, width):
        return peak * np.exp(-0.5 * ((h - at) / width) ** 2)

    def ramp(total, until):
        return total * np.minimum(h / until, 1.0)

    D = {
        "baseline": dict(ur=-ramp(0.5, 12), hpi=ramp(0.0, 4) + np.maximum(h - 4, 0) * 0.5, r3=0 * h, r10=0 * h, g=0.35 + 0 * h),
        "upside": dict(ur=-ramp(0.8, 8), hpi=ramp(6.0, 8) + np.maximum(h - 8, 0) * 0.5, r3=ramp(0.25, 8), r10=ramp(0.25, 8), g=0.55 + 0 * h),
        "downside": dict(ur=hump(2.0, 6, 4), hpi=-hump(12.0, 8, 5), r3=-ramp(1.0, 4), r10=-hump(0.5, 4, 3), g=0.35 - hump(1.2, 3, 1.5)),
        "severe": dict(ur=hump(4.5, 8, 5), hpi=-hump(30.0, 9, 6), r3=-np.minimum(ramp(1.75, 3), last.R3M - 0.5), r10=-hump(1.0, 4, 3),
                       g=0.35 - hump(2.4, 3, 1.4)),
    }
    out = []
    for name, d in D.items():
        s = pd.DataFrame(index=dates)
        s["LBR_CA"] = last.LBR_CA + d["ur"]
        s["LBR_US"] = last.LBR_US + d["ur"]      # elasticity transfer: US-estimated satellites see Canadian UR moves
        s["HPI_CA"] = last.HPI_CA * (1 + d["hpi"] / 100)
        s["R3M"] = np.maximum(last.R3M + d["r3"], 0.25)
        s["R10Y"] = last.R10Y + d["r10"]
        s["GDPR_CA"] = last.GDPR_CA * np.exp(np.log1p(d["g"] / 100).cumsum())
        s.insert(0, "scenario", name)
        out.append(s)
    sc = pd.concat(out)
    sc.index.name = "date"
    return sc


def main() -> None:
    m = build_macro()
    m.to_csv(OUT / "macro_history.csv")
    build_scenarios(m).to_csv(OUT / "macro_scenarios.csv")
    build_portfolio().to_csv(OUT / "portfolio_history.csv", index=False)
    build_snapshot().to_csv(OUT / "portfolio_snapshot.csv", index=False)
    print(m.tail(6).round(2).to_string())
    print(f"Real-data inputs written to {OUT}")

