"""Top-down net charge-off (loss-rate) model: the cross-check on the PD x LGD build.

Follows three published top-down loss equations:
  * Fed net charge-off approach for portfolios without loan-level data;
  * Hirtle, Kovner, Vickery & Bhanot (2016) CLASS: loss rate on its own lag and macro changes;
  * BoC TDSA aggregate PCL (TR-128, eq. 25): PCL ratio on its lag and the *one-quarter-ahead*
    change in unemployment, GDP growth, TSX and house prices (forward-looking provisioning).
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

NCO_DRIVERS = {
    "household": ["LBR_ON|d4", "HPI_ON|yoy", "GDPR_ON|yoy"],
    "business": ["LBR_ON|d4", "PCRE_ON|yoy", "SPRBBB|level"],
}


def _drivers(macro: pd.DataFrame, book: str, lead: int) -> pd.DataFrame:
    out = {}
    for d in NCO_DRIVERS[book]:
        mn, tr = d.split("|")
        s = macro[mn]
        s = s - s.shift(4) if tr == "d4" else 100 * (s / s.shift(4) - 1) if tr == "yoy" else s
        out[f"{d}|F{lead}"] = s.shift(-lead)
    return pd.DataFrame(out)


def fit(nco_rate: pd.Series, macro: pd.DataFrame, book: str, lead: int = 1, include_ar: bool = True):
    X = _drivers(macro, book, lead).reindex(nco_rate.index)
    if include_ar:
        X["AR1"] = nco_rate.shift(1)
    D = sm.add_constant(X)
    ok = D.notna().all(axis=1) & nco_rate.notna()
    res = sm.OLS(nco_rate[ok], D[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return res


def project(res, nco_hist: pd.Series, macro_full: pd.DataFrame, book: str, index: pd.Index, lead: int = 1) -> pd.Series:
    X = _drivers(macro_full, book, lead).ffill()  # last scenario quarter reuses its own value
    prev = nco_hist.dropna().iloc[-1]
    out = []
    for t in index:
        v = res.params["const"] + sum(res.params[c] * X.at[t, c] for c in X.columns)
        if "AR1" in res.params:
            v += res.params["AR1"] * prev
        v = max(v, 0.0)
        out.append(v)
        prev = v
    return pd.Series(out, index=index)
