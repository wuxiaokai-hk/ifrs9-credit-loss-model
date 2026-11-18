"""LGD and validation-rule tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditloss.modelling import lgd
from creditloss.validation.oos import choose_champion, diebold_mariano


def _book(ltv):
    return pd.DataFrame({"ltv_bucket": ["a"], "balance": [1.0], "current_ltv": [ltv]})


def test_low_ltv_loses_only_workout_cost():
    idx = pd.RangeIndex(12)
    port, _ = lgd.structural_ltv(_book(0.3), pd.Series(1.0, idx), pd.Series(1.0, idx), {"workout_cost": 0.02})
    assert np.allclose(port, 0.02)


def test_lgd_rises_when_house_prices_fall():
    idx = pd.RangeIndex(12)
    flat, _ = lgd.structural_ltv(_book(0.8), pd.Series(1.0, idx), pd.Series(1.0, idx), {})
    crash, _ = lgd.structural_ltv(_book(0.8), pd.Series(np.linspace(1, 0.7, 12), idx), pd.Series(1.0, idx), {})
    assert crash.iloc[0] > flat.iloc[0]


def test_frye_jacobs_anchored_and_increasing():
    p = pd.Series([0.01, 0.02, 0.05])
    out = lgd.frye_jacobs(p, pd_long_run=0.02, elgd=0.4, rho=0.15)
    assert out.is_monotonic_increasing
    assert 0.3 < out.iloc[1] < 0.4                                     # below ELGD at the long-run PD when rho > 0
    assert lgd.frye_jacobs(p, 0.02, 0.4, rho=1e-9).iloc[1] == pytest.approx(0.4, abs=1e-6)


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    rows = []
    for o in pd.date_range("2000-03-31", periods=60, freq="QE"):
        a = rng.normal()
        rows += [{"segment": "s", "origin": o, "h": 4, "model": "AR(1) benchmark", "forecast": a + rng.normal(0, 2), "actual": a},
                 {"segment": "s", "origin": o, "h": 4, "model": "good", "forecast": a + rng.normal(0, 0.5), "actual": a}]
    dm = diebold_mariano(pd.DataFrame(rows), [4])
    r = dm[dm.model == "good"].iloc[0]
    assert r.better_than_ar and r.p_value < 0.01


def test_champion_rule_band_and_exclusions():
    m = pd.DataFrame({"segment": "s", "model": ["AR(1) benchmark", "a", "b", "c"], "h": "1-12",
                      "rmse_pp": [0.1, 0.2, 0.3, 0.15]})
    st = pd.DataFrame({"segment": "s", "window": "2008-01-01 to 2010-12-31", "model": ["a", "b", "c"],
                       "cum_capture_pct": [60, 120, 150]})
    ch = choose_champion(m, st, exclude={("s", "c")})
    assert ch.champion.iloc[0] == "b" and ch.rule_met.iloc[0]          # a fails the band, c has a wrong sign
