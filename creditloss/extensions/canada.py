"""Canadian panel data and the mortgage renewal payment shock.

Sources (data/canada/raw/, see data/canada/SOURCES.md):
    cba_arrears_regions.csv            CBA, residential mortgages 90+ days in arrears, quarter-end months, by region
                                       (Atlantic, Quebec, Alberta, British Columbia); Ontario from data/eqb_public/raw
    statcan_unemployment_provinces_q   StatCan 14-10-0287, unemployment rate, 15+, seasonally adjusted, quarterly mean

Payment shock (in the spirit of Bank of Canada SAP 2026-12). Canadian mortgages renew every five years, so the
payment change at renewal is the ratio of the payment at today's 5-year rate to the payment at the rate five years
earlier, on the same amortization:

    pay(r)  = (r/12) / (1 - (1 + r/12)^-300)                          monthly payment per dollar, 25-year amortization
    PS_t    = 100 * mean_{k=0..3} [ pay(m_{t-k}) / pay(m_{t-k-20}) - 1 ]     % change, renewals spread over the year
    m_t     = 5-year mortgage rate, proxied by the 10-year Government of Canada yield + 1.5 pp

A rate rise lifts PS with a lag of up to five years and a rate cut lowers it, so the driver behaves sensibly in
every scenario (unlike the level of a rate, which would make arrears fall the moment rates are cut).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from creditloss import PROJECT_ROOT

RAW = PROJECT_ROOT / "data" / "canada" / "raw"
REGIONS = ["AT", "QC", "ON", "AB", "BC"]
MORTGAGE_SPREAD = 1.5
COVID = ("2020-06-30", "2021-09-30")


def _qend(s) -> pd.DatetimeIndex:
    return pd.PeriodIndex(s, freq="Q").to_timestamp(how="end").normalize()


def payment(rate_pct: np.ndarray | pd.Series, years: int = 25) -> np.ndarray:
    r = np.asarray(rate_pct, float) / 1200
    n = 12 * years
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(r > 0, r / (1 - (1 + r) ** -n), 1 / n)
    return np.where(np.isnan(r), np.nan, out)


def payment_shock(r10y: pd.Series) -> pd.Series:
    """Renewal payment shock, % (see module docstring). r10y: quarterly 10-year yield, % (history + scenario)."""
    m = r10y + MORTGAGE_SPREAD
    ratio = payment(m.to_numpy()) / payment(m.shift(20).to_numpy()) - 1
    return 100 * pd.Series(ratio, index=r10y.index).rolling(4).mean()


def unemployment() -> pd.DataFrame:
    u = pd.read_csv(RAW / "statcan_unemployment_provinces_q.csv")
    u.index = _qend(u.pop("date").str.replace("Q", "-Q"))
    u["AT"] = u[["NL", "PE", "NS", "NB"]].mean(axis=1)
    out = u[REGIONS + ["CA"]].copy()
    m = (out.index >= COVID[0]) & (out.index <= COVID[1])        # income support broke the UR-arrears link
    out[m] = np.nan
    return out.interpolate(limit_area="inside")


def arrears() -> pd.DataFrame:
    """Arrears rate, %, quarter-end, one column per region."""
    r = pd.read_csv(RAW / "cba_arrears_regions.csv")
    on = pd.read_csv(PROJECT_ROOT / "data" / "eqb_public" / "raw" / "cba_arrears_ontario.csv")
    on["region"] = "ON"
    on = on.rename(columns={"total_mortgages": "total_mortgages", "arrears_count": "arrears_count"})
    r = pd.concat([r, on[["region", "month", "total_mortgages", "arrears_count"]]])
    r = r[r.month.str[5:7].isin(["03", "06", "09", "12"]) | (r.month == r.month.max())].copy()
    r["date"] = pd.to_datetime(r.month + "-01").dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()
    r["rate"] = 100 * r.arrears_count / r.total_mortgages
    return r.pivot_table(index="date", columns="region", values="rate")[REGIONS]


def panel(macro: pd.DataFrame) -> pd.DataFrame:
    """Long panel: region, date, y (logit arrears), y_lag, dUR (4q change, regional), HPI yoy (national), PS (lag 4)."""
    a, u = arrears(), unemployment()
    ps = payment_shock(macro["R10Y"])
    hpi = 100 * macro["HPI_CA"].pct_change(4)
    rows = []
    for g in REGIONS:
        d = pd.DataFrame({"rate": a[g]}).join(u[g].rename("ur"), how="left")
        d["y"] = np.log(d.rate / (100 - d.rate))
        d["y_lag"] = d.y.shift(1)
        d["dUR"] = d.ur - d.ur.shift(4)
        d["HPI"] = hpi.reindex(d.index)
        d["PS"] = ps.shift(4).reindex(d.index)
        d["region"] = g
        rows.append(d)
    return pd.concat(rows).reset_index(names="date").dropna()


def write_sources(path: Path | None = None) -> None:
    p = path or (PROJECT_ROOT / "data" / "canada" / "SOURCES.md")
    p.write_text("""# Canadian panel data (retrieved 2026-11-26)

| File (`raw/`) | Series | Source |
|---|---|---|
| `cba_arrears_regions.csv` | Residential mortgages 90+ days in arrears, quarter-end months 1995-03 to 2026-05: Atlantic (AT), Quebec (QC), Alberta (AB), British Columbia (BC) | [Canadian Bankers Association, May 2026 table](https://votreargent.cba.ca/Assets/CanadianBankersAssociation/Documents/Articles/Statistics/stat-mortgages-arrears-may-2026-en.pdf) |
| `statcan_unemployment_provinces_q.csv` | Unemployment rate, 15+, seasonally adjusted, quarterly mean of monthly, 1990Q1–2026Q3 | Statistics Canada, table 14-10-0287-01 (Web Data Service) |

Ontario arrears come from `data/eqb_public/raw/cba_arrears_ontario.csv` (same CBA table). Manitoba and Saskatchewan
are left out: the transcription of those two tables could not be verified. The five regions cover over 90% of bank
mortgages. Atlantic unemployment is the simple mean of the four Atlantic provinces. House prices are national (BIS),
as no free provincial index covers 1995–2026.
""", encoding="utf-8")
