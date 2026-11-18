"""Loss given default under a scenario.

structural_ltv  Collateral shortfall from the indexed LTV at liquidation, the
                approach behind the Fed first-lien model (updated LTV from house
                prices), the BoC TDSA mortgage LGD adjustment
                LGD = costs/value + max(1 - (1+g)/LTV, 0)  (TR-128, footnote 7)
                and the EBA perfect-foresight rule (collateral valued on the
                scenario path at the time it is sold). Implemented as
                LGD = workout cost + max(1 - (1+g)(1-d-c)/LTV, 0): selling costs c
                and the distressed-sale discount d are netted from proceeds, so a
                low-LTV loan loses only its workout cost.
insured         Lender keeps only what the insurer does not reimburse (CMHC /
                private mortgage insurance), as in OSFI capital treatment.
frye_jacobs     LGD rises with the default rate (Frye & Jacobs 2012); used where
                collateral data are thin (business / CRE books).
fixed           Constant (unsecured consumer).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def ltv_risk_weights(ltv: np.ndarray, balance: np.ndarray, slope: float = 3.0) -> np.ndarray:
    """Relative PD by LTV bucket, normalised so the balance-weighted mean is 1.

    Defaults concentrate in high-LTV loans; TDSA (eq. 33) allocates defaults to
    the riskiest buckets in the same spirit.
    """
    ltv = np.nan_to_num(ltv, nan=float(np.nanmean(ltv)) if np.isfinite(ltv).any() else 0.5)
    m = np.exp(slope * (ltv - np.average(ltv, weights=balance)))
    return m / np.average(m, weights=balance)


def structural_ltv(book: pd.DataFrame, hpi_rel: pd.Series, bal_rel: pd.Series, seg_cfg: dict,
                   liquidation_lag: int = 4) -> tuple[pd.Series, pd.DataFrame]:
    """Portfolio LGD path, default-weighted across LTV buckets.

    hpi_rel  house price index / jump-off value, indexed by projection quarter (extended past horizon)
    bal_rel  balance / jump-off balance (amortization) on the same index
    """
    c, d = seg_cfg.get("sale_cost", 0.08), seg_cfg.get("forced_sale_discount", 0.10)
    second = seg_cfg.get("lien_position") == "second"
    w = ltv_risk_weights(book["current_ltv"].to_numpy(), book["balance"].to_numpy())
    dw = book["balance"].to_numpy() * w  # default weights
    idx = bal_rel.index
    h_sale = hpi_rel.shift(-liquidation_lag).reindex(idx).ffill()  # value at sale, perfect foresight
    rows = {}
    for _, b in book.iterrows():
        if second:
            first0 = b["first_lien_ltv"]
            line0 = b["current_ltv"] - first0
            first_t = first0 * bal_rel                      # first lien amortizes
            line_t = line0                                  # line stays drawn (revolving)
            proceeds = h_sale * (1 - d - c)                 # per unit of original value
            left = (proceeds - first_t).clip(lower=0)
            lgd = (1 - (left / line_t).clip(upper=1)).clip(0, 1)
        else:
            ltv_t = b["current_ltv"] * bal_rel / h_sale      # loan / indexed value at sale
            # sale proceeds net of the distressed-sale discount and selling costs
            lgd = (1 - (1 - d - c) / ltv_t).clip(lower=0)
        rows[b["ltv_bucket"]] = (lgd + seg_cfg.get("workout_cost", 0.02)).clip(0, 1)
    by_bucket = pd.DataFrame(rows)
    port = (by_bucket * dw).sum(axis=1) / dw.sum()
    return port, by_bucket


def insured(book, hpi_rel, bal_rel, seg_cfg, liquidation_lag=4):
    raw, by_bucket = structural_ltv(book, hpi_rel, bal_rel, seg_cfg, liquidation_lag)
    cov = seg_cfg.get("insurer_coverage", 1.0)
    port = seg_cfg.get("residual_cost", 0.01) + (1 - cov) * raw
    return port, by_bucket * (1 - cov)


def frye_jacobs(pd_annual: pd.Series, pd_long_run: float, elgd: float, rho: float) -> pd.Series:
    """Frye & Jacobs (2012): LGD_t = Phi(Phi^-1(DR_t) - k) / DR_t, k = (Phi^-1(PD) - Phi^-1(PD * ELGD)) / sqrt(1 - rho).

    ELGD is the default-weighted expected LGD over the Vasicek distribution of default rates, so LGD at the long-run
    PD sits slightly below ELGD when rho > 0 (and equals it as rho -> 0). LGD rises with the default rate.
    """
    k = (norm.ppf(pd_long_run) - norm.ppf(pd_long_run * elgd)) / np.sqrt(1 - rho)
    p = pd_annual.clip(1e-6, 0.999)
    return pd.Series(norm.cdf(norm.ppf(p) - k) / p, index=pd_annual.index).clip(0, 1)


def fixed(index, value: float) -> pd.Series:
    return pd.Series(value, index=index)
