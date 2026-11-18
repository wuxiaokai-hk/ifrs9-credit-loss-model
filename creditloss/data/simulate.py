"""Simulate stand-in data with the same shape as the real inputs.

Nothing here is real. The macro history is a synthetic Ontario-like economy with
recessions placed where Canada had them (1990-92, 2001, 2008-09, 2020) and a rate
shock in 2022-23. Portfolio default and loss rates are generated from a known
data-generating process (DGP) so the pipeline can be checked for recovering it.

Replace the four CSVs in data/ with Moody's history/scenarios and the credit
union's own portfolio extracts; keep the column names in config/sample.yaml.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

Q = pd.period_range("1990Q1", "2026Q3", freq="Q")
N = len(Q)

# The true DGP (kept here so tests can check the search recovers it).
TRUE_DGP = {
    # Long-run form, quarterly:
    #   y_t = rho*y_{t-1} + (1-rho)*[a + b_ur*(UR(t-2)-7) + b_hpi*HPIyoy(t-4)
    #                               + b_gap*(5y mortgage rate - neutral)(t-4) + b_cpi*(core CPI(t)-2)] + e_t
    # y = logit(annualized default rate). The b's are long-run effects on the logit.
    "mortgage_uninsured": dict(a=-6.00, rho=0.55, b_ur=0.180, b_hpi=-0.025, b_gap=0.100, b_cpi=-0.060, sd=0.07),
    "mortgage_insured":   dict(a=-5.60, rho=0.55, b_ur=0.185, b_hpi=-0.020, b_gap=0.120, b_cpi=-0.060, sd=0.07),
    "heloc":              dict(a=-5.80, rho=0.50, b_ur=0.170, b_hpi=-0.030, b_gap=0.110, b_cpi=-0.040, sd=0.08),
    "consumer":           dict(a=-3.90, rho=0.45, b_ur=0.160, b_hpi=-0.008, b_gap=0.050, b_cpi=-0.015, sd=0.06),
    # business: CRE price growth and BBB spread instead of HPI and mortgage gap
    "commercial_re":      dict(a=-4.60, rho=0.50, b_ur=0.110, b_cre=-0.028, b_spr=0.300, b_cpi=0.0, sd=0.09),
}


def _shock_path(centre: str, depth: float, width: float) -> np.ndarray:
    """Smooth recession bump centred on a quarter."""
    c = Q.get_loc(pd.Period(centre, "Q"))
    t = np.arange(N)
    return depth * np.exp(-0.5 * ((t - c) / width) ** 2)


def simulate_macro(rng: np.random.Generator) -> pd.DataFrame:
    t = np.arange(N)
    cycle = (_shock_path("1991Q4", 1.0, 4) + _shock_path("2002Q1", 0.30, 3)
             + _shock_path("2009Q2", 0.75, 3) + _shock_path("2020Q3", 0.9, 1.5))

    # Labour market (Ontario)
    ur = 7.0 - 0.006 * t + 5.0 * cycle + np.cumsum(rng.normal(0, 0.08, N)) * 0.4
    ur += _shock_path("1993Q2", 1.2, 4)  # slow 1990s recovery
    ur += 1.6 * _shock_path("2025Q4", 1.0, 4)  # tariff slowdown
    ur = np.clip(ur, 4.8, 14.0)
    sep = 1.9 + 0.22 * (ur - ur.mean()) + rng.normal(0, 0.06, N)
    find = 24 - 1.4 * (ur - ur.mean()) + rng.normal(0, 0.6, N)

    # Rates
    policy = np.interp(t, [0, 8, 20, 40, 60, 72, 76, 80, 100, 120, 128, 131, 133, 136, 140, 146],
                       [12, 7.5, 5, 5, 4.25, 4.0, 1.5, 0.5, 1.0, 1.75, 0.25, 0.25, 4.5, 5.0, 3.0, 2.25])
    policy = np.clip(policy + rng.normal(0, 0.1, N), 0.25, None)
    gov5 = policy + 0.6 + 0.4 * np.sin(t / 9) + rng.normal(0, 0.12, N)
    gov10 = gov5 + 0.45 + rng.normal(0, 0.08, N)
    mort5 = gov5 + 1.55 + 0.6 * cycle + rng.normal(0, 0.1, N)
    neutral = np.interp(t, [0, 40, 80, 120, 146], [6.0, 4.5, 3.5, 2.75, 2.75])

    # Prices and activity
    core = np.clip(np.interp(t, [0, 12, 40, 120, 128, 132, 136, 146], [4.5, 2.0, 1.8, 1.8, 2.0, 5.2, 3.3, 2.6])
                   + rng.normal(0, 0.15, N), 0.5, None)
    gdp_g = 2.2 - 6.0 * np.gradient(cycle) * 4 - 1.5 * cycle + rng.normal(0, 0.8, N)
    gdp_g[Q.get_loc(pd.Period("2020Q2", "Q"))] = -38.0
    gdp_g[Q.get_loc(pd.Period("2020Q3", "Q"))] = 35.0
    gdp = 100 * np.exp(np.cumsum(gdp_g / 400))

    hpi_g = (np.interp(t, [0, 4, 12, 24, 48, 70, 76, 84, 120, 128, 129, 132, 136, 140, 146],
                       [8, -2, -9, 1, 7, 9, -4, 6, 7, 16, 18, -14, -5, -3, -4])
             - 18 * cycle + rng.normal(0, 1.2, N))
    hpi_g[Q.get_loc(pd.Period("2020Q3", "Q")):Q.get_loc(pd.Period("2021Q4", "Q"))] += 22  # pandemic boom
    hpi = 100 * np.exp(np.cumsum(hpi_g / 400))
    cre_g = 0.9 * hpi_g - 10 * cycle + rng.normal(0, 2.0, N)
    cre_g[-16:] -= 6  # office weakness
    cre = 100 * np.exp(np.cumsum(cre_g / 400))
    spread = 1.3 + 2.4 * cycle + 0.3 * np.maximum(policy - neutral, 0) + rng.normal(0, 0.12, N)
    tsx = 100 * np.exp(np.cumsum((7 - 60 * np.gradient(cycle) + rng.normal(0, 14, N)) / 400))
    ydi_g = 1.8 - 2.0 * cycle + rng.normal(0, 1.0, N)
    ydi = 100 * np.exp(np.cumsum(ydi_g / 400))
    dsr = 14 + 0.45 * (mort5 - mort5.mean()) + 0.02 * t + rng.normal(0, 0.2, N)
    starts = 75 + 0.6 * hpi_g - 20 * cycle + rng.normal(0, 4, N)
    insolv_c = 30 + 5 * cycle * 4 + 0.5 * (ur - 7) * 4 + rng.normal(0, 1.2, N)
    insolv_b = 25 + 18 * cycle + rng.normal(0, 1.5, N)
    gap = -3.5 * cycle + 0.4 * np.sin(t / 7) + rng.normal(0, 0.3, N)
    covid = np.zeros(N)
    covid[Q.get_loc(pd.Period("2020Q2", "Q")):Q.get_loc(pd.Period("2021Q2", "Q")) + 1] = 1.0

    macro = pd.DataFrame({
        "date": Q.to_timestamp(how="end").normalize(),
        "LBR_ON": ur, "LBR_CA": ur - 0.3 + rng.normal(0, 0.15, N),
        "JSR_ON": sep, "JFR_ON": find,
        "ET_ON": 7000 * np.exp(np.cumsum((1.4 - 3.6 * np.gradient(ur) + rng.normal(0, 0.8, N)) / 400)),
        "GDPR_ON": gdp, "YPDR_ON": ydi,
        "HPI_ON": hpi, "HPI_TOR": hpi * np.exp(np.cumsum(rng.normal(0.0005, 0.01, N))),
        "PCRE_ON": cre, "HSTARTS_ON": starts,
        "RPOL": policy, "R3M": policy - 0.1 + rng.normal(0, 0.05, N),
        "R5Y": gov5, "R10Y": gov10, "RMORT5": mort5, "RNEUTRAL": neutral,
        "CPIX_CA": core, "CPI_ON": core + rng.normal(0, 0.5, N),
        "GAP_CA": gap, "SPRBBB": spread, "TSX": tsx, "DSR_CA": dsr,
        "INSOLV_CONS_ON": insolv_c, "INSOLV_BUS_ON": insolv_b,
        "D_COVID": covid,
    })
    # Filler variables, the way a Moody's file has hundreds of cycle-correlated series.
    fillers = {
        "RETAIL_ON": 1.0, "MFGSHIP_ON": 1.8, "EXPORTS_CA": 2.2, "POIL": 4.0, "FXUSCAD": 0.4,
        "LBR_US": 0.9, "GDPR_US": 1.0, "HOURS_ON": 0.5, "WAGE_ON": 0.3, "POP_ON": 0.05,
        "CONFCONS_CA": 1.5, "CONFBUS_CA": 1.7, "PERMITS_ON": 2.5, "SALES_RES_ON": 2.0,
        "RENT_ON": 0.2, "LISTINGS_ON": 1.2, "AUTO_SALES_ON": 2.0, "IND_PROD_CA": 1.3,
        "CREDIT_HH_CA": 0.4, "CREDIT_BUS_CA": 0.9, "VACANCY_OFF_TOR": 0.6, "PPI_CA": 1.1,
        "WTI_SPREAD": 3.0, "IMPORTS_CA": 2.0,
    }
    for name, beta in fillers.items():
        z = beta * (-cycle * 10 + rng.normal(0, 3, N)) + np.cumsum(rng.normal(0, 0.5, N))
        macro[name] = 100 + z
    return macro


def _dgp_driver_frame(macro: pd.DataFrame) -> pd.DataFrame:
    hpi_yoy = 100 * macro["HPI_ON"].pct_change(4)
    cre_yoy = 100 * macro["PCRE_ON"].pct_change(4)
    return pd.DataFrame({
        "ur_l2": macro["LBR_ON"].shift(2),
        "hpi_l4": hpi_yoy.shift(4),
        "gap_l4": (macro["RMORT5"] - macro["RNEUTRAL"]).shift(4),
        "cpi": macro["CPIX_CA"],
        "cre_l2": cre_yoy.shift(2),
        "spr_l1": macro["SPRBBB"].shift(1),
        "covid": macro["D_COVID"],
    }).bfill()


def simulate_portfolio(macro: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    x = _dgp_driver_frame(macro)
    rows = []
    bal0 = {"mortgage_uninsured": 6.5e9, "mortgage_insured": 2.5e9, "heloc": 1.4e9,
            "consumer": 0.6e9, "commercial_re": 3.0e9}
    for seg, p in TRUE_DGP.items():
        y = np.empty(N)
        y[0] = p["a"]
        for t in range(1, N):
            if seg == "commercial_re":
                drive = p["b_ur"] * (x.ur_l2[t] - 7) + p["b_cre"] * x.cre_l2[t] + p["b_spr"] * (x.spr_l1[t] - 1.3)
            else:
                drive = (p["b_ur"] * (x.ur_l2[t] - 7) + p["b_hpi"] * x.hpi_l4[t]
                         + p["b_gap"] * x.gap_l4[t] + p["b_cpi"] * (x.cpi[t] - 2))
            drive -= 0.9 * x.covid[t]  # payment deferrals suppressed 2020-21 defaults
            y[t] = p["rho"] * y[t - 1] + (1 - p["rho"]) * (p["a"] + drive) + rng.normal(0, p["sd"])
        pd_annual = expit(y)

        growth = 0.012 + rng.normal(0, 0.004, N)
        bal = bal0[seg] * np.exp(np.cumsum(growth) - growth.sum())
        hpi_rel = macro["HPI_ON"].to_numpy()
        # Realized loss rate on defaulted balances: structural, with noise
        hpi_chg = pd.Series(hpi_rel).pct_change(8).fillna(0).to_numpy()
        if seg == "mortgage_uninsured":
            lgd = np.clip(0.025 + 0.45 * np.maximum(-hpi_chg, 0) + rng.normal(0, 0.01, N), 0.0, 0.6)
        elif seg == "mortgage_insured":
            lgd = np.clip(0.01 + rng.normal(0, 0.004, N), 0, 0.05)
        elif seg == "heloc":
            lgd = np.clip(0.06 + 1.2 * np.maximum(-hpi_chg, 0) + rng.normal(0, 0.02, N), 0.0, 0.9)
        elif seg == "consumer":
            lgd = np.clip(0.75 + rng.normal(0, 0.04, N), 0.5, 0.95)
        else:
            lgd = np.clip(0.25 + 1.5 * (pd_annual - pd_annual.mean()) + rng.normal(0, 0.03, N), 0.1, 0.7)
        cure = {"mortgage_uninsured": 0.45, "mortgage_insured": 0.45, "heloc": 0.30,
                "consumer": 0.10, "commercial_re": 0.20}[seg]
        nco = pd_annual * (1 - cure) * lgd
        arrears = pd.Series(pd_annual).ewm(alpha=0.35).mean().to_numpy() * 1.1
        for t in range(N):
            rows.append({
                "date": macro["date"].iloc[t], "segment": seg, "balance": bal[t],
                "undrawn": bal[t] * (0.55 if seg == "heloc" else 0.30 if seg == "consumer" else 0.0),
                "default_rate": 100 * pd_annual[t], "arrears_rate": 100 * arrears[t],
                "nco_rate": 100 * nco[t], "lgd_realized": lgd[t],
                "insured_share": 1.0 if seg == "mortgage_insured" else 0.0,
            })
    return pd.DataFrame(rows)


def simulate_snapshot(rng: np.random.Generator) -> pd.DataFrame:
    """Jump-off book by LTV bucket (current LTV, indexed)."""
    spec = {
        "mortgage_uninsured": [("0-50", 0.30, 0.38), ("50-65", 0.28, 0.58), ("65-75", 0.24, 0.70), ("75-80", 0.18, 0.78)],
        "mortgage_insured":   [("50-65", 0.10, 0.60), ("65-80", 0.30, 0.73), ("80-90", 0.35, 0.86), ("90-95", 0.25, 0.93)],
        "heloc":              [("0-50", 0.40, 0.42), ("50-65", 0.35, 0.58), ("65-80", 0.25, 0.73)],
        "consumer":           [("n/a", 1.0, np.nan)],
        "commercial_re":      [("0-50", 0.25, 0.45), ("50-65", 0.45, 0.60), ("65-75", 0.30, 0.70)],
    }
    total = {"mortgage_uninsured": 7.9e9, "mortgage_insured": 3.0e9, "heloc": 1.7e9, "consumer": 0.75e9, "commercial_re": 3.6e9}
    rows = []
    for seg, buckets in spec.items():
        for name, share, ltv in buckets:
            rows.append({
                "segment": seg, "ltv_bucket": name, "balance": total[seg] * share,
                "current_ltv": ltv,
                # HELOC: first-lien LTV ahead of the line (combined LTV = first + heloc)
                "first_lien_ltv": (ltv - 0.12) if seg == "heloc" else 0.0,
                "undrawn": total[seg] * share * (0.55 if seg == "heloc" else 0.30 if seg == "consumer" else 0.0),
                "remaining_amort_years": {"consumer": 4, "commercial_re": 18}.get(seg, 21),
                "prepay_rate": {"consumer": 0.15, "commercial_re": 0.06}.get(seg, 0.08),
                "stage2_share": {"consumer": 0.09, "commercial_re": 0.14}.get(seg, 0.07) * (1 + 0.5 * (0.5 if pd.isna(ltv) else ltv)),
                "stage3_share": {"consumer": 0.012, "commercial_re": 0.018}.get(seg, 0.004),
            })
    return pd.DataFrame(rows)


def simulate_scenarios(macro: pd.DataFrame, rng: np.random.Generator, n: int = 24) -> pd.DataFrame:
    """Four Moody's-style paths from the jump-off. Shapes, not forecasts."""
    last = macro.iloc[-1]
    dates = pd.period_range(pd.Period(macro["date"].iloc[-1], "Q") + 1, periods=n, freq="Q").to_timestamp(how="end").normalize()
    h = np.arange(1, n + 1)

    def hump(peak, at, width):
        return peak * np.exp(-0.5 * ((h - at) / width) ** 2)

    def ramp(total, until):  # cumulative move reached by `until`, then held
        return total * np.minimum(h / until, 1.0)

    designs = {
        # name: (UR add, HPI path % vs jump-off, policy add, mortgage spread add, core CPI add, CRE %, BBB spread add, GDP growth add)
        "baseline": dict(ur=hump(0.3, 3, 4) - ramp(0.6, 16), hpi=ramp(2.0, 8) + ramp(6, 24), pol=-ramp(0.25, 4),
                         msp=0 * h, cpi=-ramp(0.4, 8), cre=ramp(1, 12), spr=0 * h, gdp=0.2 + 0 * h),
        "upside": dict(ur=-ramp(1.3, 12), hpi=ramp(9, 12) + ramp(6, 24), pol=ramp(0.25, 8),
                       msp=0 * h, cpi=-ramp(0.3, 8), cre=ramp(6, 12), spr=-ramp(0.2, 6), gdp=1.0 + 0 * h),
        "downside": dict(ur=hump(2.2, 7, 5), hpi=-hump(12, 8, 6), pol=-ramp(1.25, 4),
                         msp=hump(0.5, 4, 3), cpi=-ramp(0.6, 6), cre=-hump(15, 8, 6), spr=hump(1.0, 4, 3), gdp=-hump(3.5, 3, 2)),
        "severe": dict(ur=hump(4.3, 8, 6), hpi=-hump(28, 9, 7), pol=-ramp(1.75, 3),
                       msp=hump(1.4, 4, 3), cpi=hump(1.2, 3, 2) - ramp(1.0, 10), cre=-hump(32, 9, 7), spr=hump(2.6, 4, 3), gdp=-hump(7.0, 3, 2)),
    }
    out = []
    for name, d in designs.items():
        df = pd.DataFrame({"date": dates})
        df["scenario"] = name
        df["LBR_ON"] = last.LBR_ON + d["ur"]
        df["LBR_CA"] = last.LBR_CA + d["ur"]
        df["JSR_ON"] = last.JSR_ON + 0.22 * d["ur"]
        df["JFR_ON"] = last.JFR_ON - 1.4 * d["ur"]
        df["ET_ON"] = last.ET_ON * np.exp(np.log1p(0.014 / 4) * h - 0.009 * d["ur"])
        df["HPI_ON"] = last.HPI_ON * (1 + d["hpi"] / 100)
        df["HPI_TOR"] = last.HPI_TOR * (1 + 1.15 * d["hpi"] / 100)
        df["PCRE_ON"] = last.PCRE_ON * (1 + d["cre"] / 100)
        df["RPOL"] = np.clip(last.RPOL + d["pol"], 0.25, None)
        df["R3M"] = df["RPOL"] - 0.1
        df["R5Y"] = df["RPOL"] + 0.6 + 0.25 * (d["ur"] < 0)
        df["R10Y"] = df["R5Y"] + 0.45
        df["RMORT5"] = df["R5Y"] + 1.55 + d["msp"]
        df["RNEUTRAL"] = last.RNEUTRAL
        df["CPIX_CA"] = last.CPIX_CA + d["cpi"]
        df["CPI_ON"] = df["CPIX_CA"]
        df["GAP_CA"] = last.GAP_CA - 0.8 * d["ur"]
        df["SPRBBB"] = last.SPRBBB + d["spr"]
        df["GDPR_ON"] = last.GDPR_ON * np.exp(np.cumsum((1.6 + d["gdp"] - 0.5 * np.gradient(d["ur"]) * 4) / 400))
        df["YPDR_ON"] = last.YPDR_ON * np.exp(np.cumsum((1.5 - 0.6 * d["ur"]) / 400))
        df["TSX"] = last.TSX * (1 + 0.6 * d["hpi"] / 100)
        df["DSR_CA"] = last.DSR_CA + 0.45 * (df["RMORT5"] - last.RMORT5)
        df["HSTARTS_ON"] = last.HSTARTS_ON + 0.5 * d["hpi"]
        df["INSOLV_CONS_ON"] = last.INSOLV_CONS_ON + 2.5 * d["ur"]
        df["INSOLV_BUS_ON"] = last.INSOLV_BUS_ON + 4.0 * np.maximum(d["ur"], 0)
        df["D_COVID"] = 0.0
        for c in macro.columns:
            if c not in df.columns:
                df[c] = last[c] - 3 * d["ur"]
        out.append(df[["scenario"] + list(macro.columns)])
    return pd.concat(out, ignore_index=True)


def main(out_dir: str = "data/sample", seed: int = 20261026) -> None:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    macro = simulate_macro(rng)
    macro.to_csv(out / "macro_history.csv", index=False)
    simulate_portfolio(macro, rng).to_csv(out / "portfolio_history.csv", index=False)
    simulate_snapshot(rng).to_csv(out / "portfolio_snapshot.csv", index=False)
    simulate_scenarios(macro, rng).to_csv(out / "macro_scenarios.csv", index=False)
    print(f"Simulated data written to {out.resolve()}")

