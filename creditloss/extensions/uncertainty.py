"""Loss distributions and a reverse stress test.

Loss distribution. For each segment's champion PD equation, logit(PD_t) = c + rho * logit(PD_t-1) + x_t'b + e_t,
draw the macro coefficients from their estimated sampling distribution, b* ~ N(b, diag(se^2)) with se = |b / t|, and
the quarterly shocks e_t ~ N(0, sigma^2) from the residual standard deviation. The constant and rho are kept at
their estimates: their sampling errors are strongly correlated, and drawing them independently (only their t-stats
are stored, not their covariance) would make the paths drift off without any macro cause. Each draw gives a PD path, and the loss
engine turns it into losses. The shocks are common to all segments in a draw (a downturn in credit performance
hits every book), which is the conservative choice. Result: percentiles of the 13-quarter loss per scenario.

Reverse stress test. Start from the baseline and add an unemployment hump (peak at quarter 8) and a house-price fall
(trough at quarter 9), shaped like the severe scenario, over a grid of sizes, with rates unchanged. The grid of
13-quarter losses shows which combinations reach a given loss, e.g. the severe-scenario loss or C$2bn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from creditloss.data import features as F
from creditloss.data.load import splice


def _spec(ctx, seg):
    sel, name = ctx.res["sels"][seg], ctx.res["champions"][seg]
    if name == "BMA" and sel.get("bma_specs"):
        return max(sel["bma_specs"], key=lambda t: t[1])[0]
    return sel["specs"][name]


def _x(ctx, seg, scen):
    sc = ctx.cfg["segments"][seg]
    full = splice(ctx.macro, scen)
    X, _ = F.build_features(full, ctx.lit, ctx.cfg, sc.get("sign_priors", "household"), sc.get("candidates"))
    return X.loc[scen.index]


def simulate(ctx, n: int = 400, scenarios=("baseline", "severe"), seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(draw-level total losses, quarterly proxy-path quantiles) for the named scenarios."""
    rng = np.random.default_rng(seed)
    T = len(next(iter(ctx.scen.values())))
    shocks = rng.standard_normal((n, T))                        # common credit shock across segments
    zb = {seg: rng.standard_normal((n, 6)) for seg in ctx.segments}
    draws, bands = [], []
    for sname in scenarios:
        scen = ctx.scen[sname]
        total = np.zeros(n)
        for seg in ctx.segments:
            sp = _spec(ctx, seg)
            X = _x(ctx, seg, scen)
            keys = [k for k in sp.params if k not in ("const", "AR1")]
            b = np.array([sp.params[k] for k in keys])
            se = np.array([abs(sp.params[k] / sp.tstats[k]) if sp.tstats.get(k) else 0.0 for k in keys])
            c, rho = sp.params.get("const", 0.0), sp.params.get("AR1", 0.0) if sp.include_ar else 0.0
            y0 = float(ctx.res["sels"][seg]["y"].dropna().iloc[-1])
            Xv = X[keys].to_numpy() if keys else np.zeros((T, 0))
            paths = np.empty((n, T))
            for i in range(n):
                z = zb[seg][i]
                bi = b + se * z[:len(b)]
                prev = y0
                for t in range(T):
                    prev = c + rho * prev + (Xv[t] @ bi if len(bi) else 0.0) + sp.sigma * shocks[i, t]
                    paths[i, t] = prev
            rates = F.inverse_target(paths, ctx.how) / 100
            for i in range(n):
                total[i] += ctx.cum_loss(ctx.losses(seg, pd.Series(rates[i], index=scen.index), scen))
            q = np.quantile(100 * rates, [0.05, 0.5, 0.95], axis=0)
            bands.append(pd.DataFrame({"scenario": sname, "segment": seg, "date": scen.index,
                                       "p5": q[0], "p50": q[1], "p95": q[2]}))
        draws.append(pd.DataFrame({"scenario": sname, "draw": np.arange(n), "loss_C$m": total / 1e6}))
    return pd.concat(draws, ignore_index=True), pd.concat(bands, ignore_index=True)


def distribution_table(draws: pd.DataFrame, point: dict) -> pd.DataFrame:
    rows = []
    for sname, g in draws.groupby("scenario", sort=False):
        q = g["loss_C$m"].quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        rows.append({"scenario": sname, "point estimate": point.get(sname), "p5": q[0.05], "p25": q[0.25],
                     "median": q[0.5], "p75": q[0.75], "p95": q[0.95], "p99": q[0.99]})
    return pd.DataFrame(rows)


def _hump(T, peak, at, width):
    h = np.arange(1, T + 1)
    return peak * np.exp(-0.5 * ((h - at) / width) ** 2)


def reverse_stress(ctx, ur_peaks=range(0, 9), hpi_falls=range(0, 51, 10)) -> pd.DataFrame:
    base = ctx.scen["baseline"]
    T = len(base)
    rows = []
    for u in ur_peaks:
        for h in hpi_falls:
            s = base.copy()
            for col in ("LBR_CA", "LBR_US"):
                if col in s:
                    s[col] = s[col] + _hump(T, u, 8, 5)
            s["HPI_CA"] = s["HPI_CA"] * (1 - _hump(T, h, 9, 6) / 100)
            tot = sum(ctx.cum_loss(ctx.losses(seg, ctx.proxy_path(seg, s), s)) for seg in ctx.segments)
            rows.append({"unemployment_rise_pp": u, "house_price_fall_pct": h, "loss_C$m": tot / 1e6})
    return pd.DataFrame(rows)
