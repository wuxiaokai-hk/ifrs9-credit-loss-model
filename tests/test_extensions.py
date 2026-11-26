"""Checks on the model extensions (EQB public-data configuration)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditloss import PROJECT_ROOT
from creditloss.extensions import bvar, canada, history, panel, uncertainty


@pytest.fixture(scope="module")
def ctx():
    from creditloss.extensions.context import Context
    return Context(str(PROJECT_ROOT / "config" / "eqb_public.yaml"))


def test_payment_shock_basics():
    assert canada.payment(np.array([0.0, 5.0, np.nan]))[2] != canada.payment(np.array([0.0, 5.0, np.nan]))[2]  # NaN
    r = pd.Series(4.0, index=pd.date_range("2000-03-31", periods=40, freq="QE"))
    assert canada.payment_shock(r).dropna().abs().max() < 1e-9          # flat rates: no shock
    up = pd.Series(np.linspace(2, 6, 40), index=r.index)
    assert canada.payment_shock(up).dropna().iloc[-1] > 0               # rising rates: payments rise at renewal


def test_panel_data_and_ontario_payment_shock(ctx):
    p = canada.panel(ctx.macro)
    assert set(p.region) == set(canada.REGIONS)
    assert p.rate.between(0, 2).all()
    r = panel.fit(p, "Panel: UR + HPI + Ontario payment shock")
    assert r.params["dUR"] > 0 and r.params["HPI"] < 0 and r.params["PS"] > 0


def test_loss_distribution_centred_on_point(ctx):
    s = ctx.res["summary"]
    point = (s[s.horizon_q == ctx.H].groupby("scenario").cum_loss.sum() / 1e6).to_dict()
    draws, _ = uncertainty.simulate(ctx, n=40)
    t = uncertainty.distribution_table(draws, point).set_index("scenario")
    assert t.loc["baseline", "median"] < t.loc["severe", "median"]
    for k in ("baseline", "severe"):
        assert t.loc[k, "p5"] < point[k] < t.loc[k, "p95"]


def test_reverse_stress_monotone(ctx):
    g = uncertainty.reverse_stress(ctx, ur_peaks=[0, 4, 8], hpi_falls=[0, 40])
    g = g.pivot(index="unemployment_rise_pp", columns="house_price_fall_pct", values="loss_C$m")
    assert (g.diff().dropna() > 0).all().all()
    assert (g[40] > g[0]).all()


def test_bvar_scenarios_ordered(ctx):
    sims = bvar.simulate(bvar.fit(ctx.macro), n=400)
    sc, tab = bvar.scenarios(ctx, sims)
    tab = tab.set_index("scenario")
    assert tab.loc[["upside", "baseline", "downside", "severe"], "probability"].sum() == pytest.approx(1.0)
    assert tab.loc["upside", "UR_rise_pp"] < tab.loc["baseline", "UR_rise_pp"] < tab.loc["severe", "UR_rise_pp"]
    assert set(sc) == set(bvar.BANDS)


def test_pcl_backtest_runs(ctx):
    d = history.backtest(ctx)
    assert list(d.year) == list(history.PCL.year)
    assert (d.model_macro > 0).all()
    assert history.fit_stats(d)["corr_macro"] > 0
