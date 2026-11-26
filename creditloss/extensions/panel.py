"""Provincial panel PD satellite with the renewal payment shock.

    logit(arrears_r,t) = a_r + rho * logit(arrears_r,t-1) + b1 * dUR_r,t + b2 * HPI_yoy_t + b3_ON * PS_t-4 * 1[r=ON]
                         + b3_other * PS_t-4 * 1[r!=ON] + e_r,t

The payment-shock slope is estimated separately for Ontario: pooled across provinces it is insignificant and
wrong-signed, because arrears fell in Alberta, Quebec and the Atlantic while rates rose in 2022-26; in Ontario,
where the 2023-26 rise in arrears was concentrated, it is positive and significant. This split was chosen after
looking at the data, so the out-of-sample test below is the guard against over-fitting.

Five regions (Atlantic, Quebec, Ontario, Alberta, BC), 1999Q4-2026Q2. Region intercepts a_r, common slopes.
Standard errors are clustered by quarter (shocks hit all provinces at once). The panel adds Alberta's 2015-16 oil
shock and the regional cycles to Ontario's single downturn, so the slopes rest on far more variation.

Compared for Ontario with Ontario-only regressions and an AR(1), out of sample: at every fourth quarter from 2007Q4,
re-estimate on data to that date and simulate 1-12 quarters ahead with the macro that happened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from creditloss.extensions import canada as C
from creditloss.validation.oos import diebold_mariano

MODELS = {
    "AR(1) benchmark": dict(panel=False, x=[]),
    "Ontario: UR + HPI": dict(panel=False, x=["dUR", "HPI"]),
    "Ontario: UR + HPI + payment shock": dict(panel=False, x=["dUR", "HPI", "PS"]),
    "Panel: UR + HPI": dict(panel=True, x=["dUR", "HPI"]),
    "Panel: UR + HPI + Ontario payment shock": dict(panel=True, x=["dUR", "HPI", "PS"], ps_region="ON"),
}
SIGNS = {"dUR": 1, "HPI": -1, "PS": 1}


def _design(d: pd.DataFrame, spec: dict) -> pd.DataFrame:
    X = d[["y_lag"] + spec["x"]].copy()
    if spec.get("ps_region"):          # payment-shock slope for one region, a separate one for the rest
        own = (d.region == spec["ps_region"]).astype(float)
        X["PS"], X["PS_other"] = d.PS * own, d.PS * (1 - own)
    if spec["panel"]:
        for g in C.REGIONS[1:]:
            X[f"FE_{g}"] = (d.region == g).astype(float)
    return sm.add_constant(X, has_constant="add")


def fit(p: pd.DataFrame, name: str, end=None):
    spec = MODELS[name]
    d = p if spec["panel"] else p[p.region == "ON"]
    if end is not None:
        d = d[d.date <= end]
    X = _design(d, spec)
    if spec["panel"]:
        return sm.OLS(d.y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.date)[0]})
    return sm.OLS(d.y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})


def simulate_on(res, name: str, p_on: pd.DataFrame, y0: float) -> np.ndarray:
    """Dynamic path of Ontario logit arrears over the rows of p_on (actual drivers), from y0."""
    spec, b = MODELS[name], res.params
    base = b["const"] + (b.get("FE_ON", 0.0) if spec["panel"] else 0.0)
    y, out = y0, []
    for _, r in p_on.iterrows():
        y = base + b["y_lag"] * y + sum(b[k] * r[k] for k in spec["x"])
        out.append(y)
    return np.array(out)


def _rate(y):
    return 100 / (1 + np.exp(-np.asarray(y)))


def coefficients(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODELS:
        r = fit(p, name)
        for k in ["y_lag"] + MODELS[name]["x"]:
            rows.append({"model": name, "term": k, "coef": r.params[k], "t": r.tvalues[k],
                         "sign_ok": (k == "y_lag") or np.sign(r.params[k]) == SIGNS[k], "nobs": int(r.nobs), "r2": r.rsquared})
    return pd.DataFrame(rows)


def backtest(p: pd.DataFrame, first="2007-12-31", step=4, H=12) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    on = p[p.region == "ON"].set_index("date")
    dates = on.index
    origins = [d for d in dates[dates >= pd.Timestamp(first)][::step] if d < dates[-1]]
    fc = []
    for o in origins:
        fut = on[on.index > o].iloc[:H]
        y0 = on.loc[o, "y"]
        for name in MODELS:
            r = fit(p, name, end=o)
            sim = simulate_on(r, name, fut, y0)
            for h, (d, v) in enumerate(zip(fut.index, sim), 1):
                fc.append({"segment": "mortgage (Ontario arrears)", "origin": o, "h": h, "model": name,
                           "forecast": _rate(v), "actual": fut.loc[d, "rate"]})
    fc = pd.DataFrame(fc)
    fc["err"] = fc.forecast - fc.actual
    m = (fc.assign(bucket=np.where(fc.h <= 4, "1-4", np.where(fc.h <= 8, "5-8", "9-12")))
         .groupby(["model", "bucket"]).err.apply(lambda e: float(np.sqrt(np.mean(e ** 2)))).unstack())
    m["1-12"] = fc.groupby("model").err.apply(lambda e: float(np.sqrt(np.mean(e ** 2))))
    m = m.loc[list(MODELS)].reset_index()
    dm = diebold_mariano(fc, [4, 8])
    return m, dm, fc


def holdout(p: pd.DataFrame, end="2022-12-31") -> pd.DataFrame:
    """Estimate to 2022Q4, then simulate Ontario through 2026Q2 with actual drivers: the renewal-wave test."""
    on = p[p.region == "ON"].set_index("date")
    fut = on[on.index > end]
    y0 = on.loc[pd.Timestamp(end), "y"]
    out = pd.DataFrame({"actual": fut.rate})
    for name in MODELS:
        out[name] = _rate(simulate_on(fit(p, name, end=end), name, fut, y0))
    return out


def scenario_drivers(ctx, scen: pd.DataFrame) -> pd.DataFrame:
    """Ontario drivers over a scenario: Ontario UR moves with the scenario's national UR; PS from the 10-year yield."""
    from creditloss.data.load import splice
    full = splice(ctx.macro, scen)
    ur_on = C.unemployment()["ON"].reindex(full.index)
    last = ur_on.last_valid_index()
    gap = ur_on[last] - full.loc[last, "LBR_CA"]
    ur_on[ur_on.index > last] = full.loc[ur_on.index > last, "LBR_CA"] + gap
    d = pd.DataFrame(index=full.index)
    d["dUR"] = ur_on - ur_on.shift(4)
    d["HPI"] = 100 * full["HPI_CA"].pct_change(4)
    d["PS"] = C.payment_shock(full["R10Y"]).shift(4)
    return d.loc[scen.index]


def scenario_losses(ctx, p: pd.DataFrame, name="Panel: UR + HPI + Ontario payment shock") -> pd.DataFrame:
    """Mortgage losses when the Ontario proxy follows the panel model instead of the champion."""
    r = fit(p, name)
    y0 = float(p[p.region == "ON"].sort_values("date").y.iloc[-1])
    rows = []
    for sname, sdf in ctx.scen.items():
        drv = scenario_drivers(ctx, sdf)
        proxy = pd.Series(_rate(simulate_on(r, name, drv, y0)) / 100, index=sdf.index)
        for seg in [s for s in ctx.segments if s.startswith("mortgage")]:
            champ = ctx.cum_loss(ctx.losses(seg, ctx.proxy_path(seg, sdf), sdf))
            new = ctx.cum_loss(ctx.losses(seg, proxy, sdf))
            rows.append({"segment": seg, "scenario": sname, "champion_C$m": champ / 1e6, "panel_payment_shock_C$m": new / 1e6,
                         "peak_arrears_pct": float(100 * proxy.max()), "peak_PS_pct": float(drv.PS.max())})
    return pd.DataFrame(rows)
