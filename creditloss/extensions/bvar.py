"""Model-based scenarios from a Bayesian VAR.

Variables (quarterly, Canada): change in unemployment, house-price growth (log %), 3-month rate, 10-year yield,
real GDP growth (log %). Two lags and a constant; Minnesota prior (own first lag ~ N(0 or 1, l1^2), other lags
shrunk harder, l1 = 0.2, cross-variable factor 0.5, lag decay 1/l), estimated equation by equation (Theil mixed
estimation) on 1993Q2-2026Q2 with the COVID quarters bridged as in the main data.

Simulation: several thousand paths (default 3,000 in the report) of 24 quarters from the jump-off, drawing the coefficients from their posterior and the
shocks from the residual covariance. Each path gets a severity score (standardized peak rise in unemployment plus
standardized peak fall in house prices over 12 quarters). Scenarios are the median paths within severity bands, and
the band widths are their probabilities:

    upside   0-10%    baseline 10-50%    downside 50-90%    severe 90-100%    tail (1 in 40) 97.5-100%

So the weights and the shape of each scenario come from the data rather than from judgement. Rates fall in the
severe paths because they did in past downturns; the policy response is part of the history the VAR learns from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VARS = ["dUR", "gHPI", "R3M", "R10Y", "gGDP"]
BANDS = {"upside": (0.0, 0.10), "baseline": (0.10, 0.50), "downside": (0.50, 0.90), "severe": (0.90, 1.0),
         "tail": (0.975, 1.0)}


def _data(macro: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"dUR": macro.LBR_CA.diff(), "gHPI": 100 * np.log(macro.HPI_CA).diff(), "R3M": macro.R3M,
                      "R10Y": macro.R10Y, "gGDP": 100 * np.log(macro.GDPR_CA).diff()})
    return d.dropna()


def fit(macro: pd.DataFrame, p: int = 2, l1: float = 0.2, l2: float = 0.5):
    d = _data(macro)
    Y = d.iloc[p:].to_numpy()
    X = np.column_stack([np.ones(len(Y))] + [d.shift(L).iloc[p:].to_numpy() for L in range(1, p + 1)])
    k, n = X.shape[1], len(VARS)
    sig = np.array([np.std(np.diff(d[v].to_numpy())) if v in ("R3M", "R10Y") else d[v].std() for v in VARS])
    B, covs = np.zeros((k, n)), []
    for j, v in enumerate(VARS):
        prior_mean = np.zeros(k)
        if v in ("R3M", "R10Y"):
            prior_mean[1 + j] = 1.0                      # levels: random-walk prior
        sd = np.full(k, 1e3)                              # loose on the constant
        for L in range(1, p + 1):
            for i in range(n):
                s = l1 / L * (1 if i == j else l2 * sig[j] / sig[i])
                sd[1 + (L - 1) * n + i] = s
        P = np.diag(1 / sd ** 2)
        s2 = sig[j] ** 2
        A = X.T @ X / s2 + P
        Ainv = np.linalg.inv(A)
        B[:, j] = Ainv @ (X.T @ Y[:, j] / s2 + P @ prior_mean)
        covs.append(Ainv)
    E = Y - X @ B
    Sigma = np.cov(E.T)
    return {"B": B, "covs": covs, "Sigma": Sigma, "p": p, "last": d.iloc[-p:].to_numpy(), "data": d}


def simulate(model, T: int = 24, n: int = 5000, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B, p, nv = model["B"], model["p"], len(VARS)
    L = np.linalg.cholesky(model["Sigma"])
    out = np.empty((n, T, nv))
    for i in range(n):
        Bi = B + np.column_stack([rng.multivariate_normal(np.zeros(B.shape[0]), c) for c in model["covs"]])
        hist = list(model["last"])
        for t in range(T):
            x = np.concatenate([[1.0]] + [hist[-L_] for L_ in range(1, p + 1)])
            y = x @ Bi + L @ rng.standard_normal(nv)
            y[2] = max(y[2], 0.25)                        # policy-rate floor
            y[3] = max(y[3], 0.5)
            hist.append(y)
            out[i, t] = y
    return out


def scenarios(ctx, sims: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Scenario DataFrames in the model's format, plus a table of severities and probabilities."""
    base = ctx.scen["baseline"]
    T = min(sims.shape[1], len(base))
    last = ctx.macro.iloc[-1]
    ur = last.LBR_CA + np.cumsum(sims[:, :T, 0], axis=1)
    hpi = last.HPI_CA * np.exp(np.cumsum(sims[:, :T, 1], axis=1) / 100)
    gdp = last.GDPR_CA * np.exp(np.cumsum(sims[:, :T, 4], axis=1) / 100)
    H = 12
    ur_rise = ur[:, :H].max(axis=1) - last.LBR_CA
    hpi_fall = 100 * (1 - hpi[:, :H].min(axis=1) / last.HPI_CA)
    sev = (ur_rise - ur_rise.mean()) / ur_rise.std() + (hpi_fall - hpi_fall.mean()) / hpi_fall.std()
    rank = sev.argsort().argsort() / (len(sev) - 1)
    out, rows = {}, []
    for name, (lo, hi) in BANDS.items():
        m = (rank >= lo) & (rank <= hi)
        s = base.iloc[:T].copy()
        s["LBR_CA"] = np.median(ur[m], axis=0)
        if "LBR_US" in s:
            s["LBR_US"] = last.LBR_US + (s["LBR_CA"] - last.LBR_CA)
        s["HPI_CA"] = np.median(hpi[m], axis=0)
        s["R3M"] = np.median(sims[m, :T, 2], axis=0)
        s["R10Y"] = np.median(sims[m, :T, 3], axis=0)
        s["GDPR_CA"] = np.median(gdp[m], axis=0)
        out[name] = s
        rows.append({"scenario": name, "probability": hi - lo, "paths": int(m.sum()),
                     "peak_UR": float(s.LBR_CA.iloc[:H].max()), "UR_rise_pp": float(s.LBR_CA.iloc[:H].max() - last.LBR_CA),
                     "HPI_trough_pct": float(100 * (s.HPI_CA.iloc[:H].min() / last.HPI_CA - 1)),
                     "R3M_end_y3": float(s.R3M.iloc[H - 1])})
    return out, pd.DataFrame(rows)


def losses(ctx, scen_set: dict) -> pd.DataFrame:
    rows = []
    for name, s in scen_set.items():
        for seg in ctx.segments:
            rows.append({"scenario": name, "segment": seg,
                         "loss_C$m": ctx.cum_loss(ctx.losses(seg, ctx.proxy_path(seg, s), s)) / 1e6})
    return pd.DataFrame(rows)
