"""True out-of-sample test of the PD satellite models.

At every origin T the whole selection (screen -> cluster -> search -> BMA) is re-run
on data up to T only, every literature spec is re-estimated on data up to T, and the
models then forecast the default rate for T+1..T+h *conditional on the macro path that
actually happened* -- the way a satellite model is used in a stress test, where the
macro path is given. This is the backtest the Fed asks of supervisory models
("performance under stressful conditions") and the one ECB WP 1845 uses to show that
hand-picked equations understate stress responses.

Outputs
  errors     one row per origin x model x horizon (forecast, actual)
  metrics    RMSE / MAE / bias (pp of annual PD) by model and horizon
  dm         Diebold-Mariano test of each model against the AR(1) benchmark
  stress     for origins just before each stress window: predicted vs actual peak
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..data import features as F
from ..modelling import pd_satellite as PD


def _forecast(sel, name, X, y_train, idx, how):
    if name == "BMA":
        z = PD.project_bma(sel["bma_specs"], X, y_train, idx)
    else:
        z = PD.project(sel["specs"][name], X, y_train, idx)
    return F.inverse_target(z.to_numpy(), how)


def run_segment(seg: str, cfg: dict, macro: pd.DataFrame, rate: pd.Series, lit: pd.DataFrame, book: str,
                candidates: list[str] | None = None,
                log=print) -> pd.DataFrame:
    how = cfg["pd_model"]["transform"]
    o = cfg["oos"]
    H = max(o["horizons"])
    y_all = F.target_transform(rate, how)
    X_all, meta = F.build_features(macro, lit, cfg, book, candidates)  # lags only use the past: safe to build once
    dates = y_all.dropna().index
    start = pd.Timestamp(o["first_origin"])
    origins = [d for d in dates[::1] if d >= start][::o["step_quarters"]]
    origins = [T for T in origins if dates.get_loc(T) + 1 < len(dates)]
    rows = []
    for T in origins:
        y_tr = y_all.loc[:T]
        X_tr = X_all.loc[:T]
        try:
            sel = PD.run_selection(y_tr, X_tr, meta, cfg, book)
        except Exception as e:  # noqa: BLE001 - keep the backtest going, record the failure
            log(f"    {seg} {T.date()}: selection failed ({e})")
            continue
        pos = dates.get_loc(T)
        idx = dates[pos + 1: pos + 1 + H]
        actual = F.inverse_target(y_all.loc[idx].to_numpy(), how)
        names = list(sel["specs"]) + (["BMA"] if sel.get("bma_specs") else [])
        for nm in names:
            fc = _forecast(sel, nm, X_all, y_tr, idx, how)
            for h, (d, f, a) in enumerate(zip(idx, fc, actual), start=1):
                rows.append({"segment": seg, "origin": T, "date": d, "h": h, "model": nm,
                             "forecast": f, "actual": a,
                             "drivers": sel["specs"][nm].label() if nm in sel["specs"] else "BMA"})
    return pd.DataFrame(rows)


def metrics(err: pd.DataFrame, horizons) -> pd.DataFrame:
    e = err.assign(err=err.forecast - err.actual)
    out = []
    for (seg, m, h), g in e[e.h.isin(horizons)].groupby(["segment", "model", "h"]):
        out.append({"segment": seg, "model": m, "h": h, "n": len(g),
                    "rmse_pp": np.sqrt(np.mean(g.err ** 2)), "mae_pp": np.mean(np.abs(g.err)),
                    "bias_pp": g.err.mean()})
    # path-average error across 1..max(h): one number per model
    for (seg, m), g in e.groupby(["segment", "model"]):
        out.append({"segment": seg, "model": m, "h": f"1-{e.h.max()}", "n": len(g),
                    "rmse_pp": np.sqrt(np.mean(g.err ** 2)), "mae_pp": np.mean(np.abs(g.err)),
                    "bias_pp": g.err.mean()})
    return pd.DataFrame(out)


def diebold_mariano(err: pd.DataFrame, horizons, bench="AR(1) benchmark") -> pd.DataFrame:
    """DM test on squared errors with Newey-West variance (h-1 lags) and the HLN small-sample correction."""
    out = []
    e = err.assign(se=(err.forecast - err.actual) ** 2)
    for (seg, h), g in e[e.h.isin(horizons)].groupby(["segment", "h"]):
        b = g[g.model == bench].set_index("origin")["se"]
        for m in g.model.unique():
            if m == bench:
                continue
            a = g[g.model == m].set_index("origin")["se"]
            d = (a - b).dropna()
            n = len(d)
            if n < 6:
                continue
            dbar = d.mean()
            gamma = [np.sum((d - dbar).iloc[k:].to_numpy() * (d - dbar).iloc[:n - k].to_numpy()) / n for k in range(h)]
            var = (gamma[0] + 2 * sum(gamma[1:])) / n
            if var <= 0:
                continue
            dm = dbar / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
            out.append({"segment": seg, "h": h, "model": m, "dm_stat": dm,
                        "p_value": 2 * (1 - stats.t.cdf(abs(dm), df=n - 1)),
                        "better_than_ar": dbar < 0})
    return pd.DataFrame(out)


def stress_capture(err: pd.DataFrame, windows) -> pd.DataFrame:
    """For each stress window, the last origin before it: forecast vs actual peak and cumulative defaults."""
    out = []
    for w0, w1 in windows:
        w0, w1 = pd.Timestamp(w0), pd.Timestamp(w1)
        cand = err[err.origin < w0]
        if cand.empty:
            continue
        T = cand.origin.max()
        g = err[(err.origin == T) & (err.date >= w0) & (err.date <= w1)]
        for (seg, m), gg in g.groupby(["segment", "model"]):
            out.append({"segment": seg, "window": f"{w0.date()} to {w1.date()}", "origin": T.date(), "model": m,
                        "actual_peak_pp": gg.actual.max(), "forecast_peak_pp": gg.forecast.max(),
                        "peak_capture_pct": 100 * gg.forecast.max() / gg.actual.max(),
                        "cum_capture_pct": 100 * gg.forecast.sum() / gg.actual.sum()})
    return pd.DataFrame(out)


def choose_champion(metrics: pd.DataFrame, stress: pd.DataFrame, window_prefix: str = "2008",
                    min_capture: float = 80.0, max_capture: float = 200.0, exclude: set | None = None,
                    bench: str = "AR(1) benchmark") -> pd.DataFrame:
    """Champion per segment.

    1. Admissible: not the AR(1) benchmark (no macro sensitivity, useless for stress) and, when
       `exclude` is given, no driver with the wrong sign in the full-sample fit.
    2. Among admissible models whose out-of-sample capture of the reference stress window
       (GFC by default) lies in [min_capture, max_capture] %, the lowest OOS RMSE wins.
       The floor is the supervisory conservatism screen (Fed: performance under stress;
       ECB WP 1845); the ceiling stops a model that wildly over-predicts from winning on
       conservatism alone.
    3. If none qualifies, the admissible model whose capture is closest to 100% (in logs).
    """
    exclude = exclude or set()
    m = metrics[metrics.h.astype(str).str.startswith("1-") & (metrics.model != bench)]
    s = stress[stress.window.str.startswith(window_prefix)].set_index(["segment", "model"])["cum_capture_pct"]
    out = []
    for seg, g in m.groupby("segment"):
        g = g[[(seg, x) not in exclude for x in g.model]]
        g = g.assign(capture=[s.get((seg, x), float("nan")) for x in g.model])
        ok = g[(g.capture >= min_capture) & (g.capture <= max_capture)]
        if not ok.empty:
            pick, met = ok.sort_values("rmse_pp").iloc[0], True
        else:
            pick = g.assign(d=(np.log(g.capture / 100)).abs()).sort_values("d").iloc[0]
            met = False
        out.append({"segment": seg, "champion": pick.model, "oos_rmse_pp": pick.rmse_pp,
                    "gfc_capture_pct": pick.capture, "rule_met": met})
    return pd.DataFrame(out)
