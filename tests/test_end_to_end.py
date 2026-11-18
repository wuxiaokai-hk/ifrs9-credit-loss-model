"""End-to-end checks on both shipped configurations."""
from __future__ import annotations

import numpy as np
import pytest

from creditloss.modelling import ifrs9 as I9


@pytest.mark.parametrize("which", ["eqb", "sample"])
def test_scenario_ordering(which, request):
    cfg, r = request.getfixturevalue(which)
    s = r["summary"]
    tot = s[s.horizon_q == cfg["horizon_quarters"]].groupby("scenario").cum_loss.sum()
    assert tot["upside"] <= tot["baseline"] < tot["downside"] < tot["severe"]


@pytest.mark.parametrize("which", ["eqb", "sample"])
def test_probability_weighted_allowance_within_scenarios(which, request):
    _, r = request.getfixturevalue(which)
    sens = r["ifrs9"]["sensitivity"]
    scen = sens[[c for c in ["upside", "baseline", "downside", "severe"] if c in sens]]
    pw = sens["probability_weighted"]
    assert ((pw >= scen.min(axis=1) - 1e-6) & (pw <= scen.max(axis=1) + 1e-6)).all()


def test_card_allowance_calibrated_to_reported(eqb):
    cfg, r = eqb
    target = cfg["segments"]["credit_cards"]["target_acl"]
    assert r["pw_ecl"].loc["credit_cards", "prob_weighted"] == pytest.approx(target, rel=0.01)


def test_rollforward_opens_at_weighted_allowance_and_closes(eqb):
    cfg, r = eqb
    d = r["ifrs9"]
    for (seg, _sc), g in r["paths"].groupby(["segment", "scenario"]):
        a = I9.allowance_rollforward(g.sort_values("q"), 0, cfg["horizon_quarters"], opening=d["pw_by_stage"].loc[seg])
        assert a.loc["Opening allowance", "Total"] == pytest.approx(
            d["pw_by_stage"].loc[seg, ["ecl_stage1", "ecl_stage2", "ecl_stage3"]].sum())
        assert np.isclose(a.drop(index="Closing allowance").Total.sum(), a.loc["Closing allowance", "Total"])


def test_champions_have_correct_signs(eqb):
    _, r = eqb
    for seg, nm in r["champions"].items():
        spec = r["sels"][seg]["specs"].get(nm)
        if spec is not None:
            assert spec.sign_ok, f"{seg}: champion {nm} has a wrong-signed driver"
