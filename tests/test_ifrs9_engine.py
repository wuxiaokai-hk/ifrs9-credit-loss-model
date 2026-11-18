"""Unit tests of the IFRS 9 stage engine and its disclosure tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditloss.modelling import ifrs9 as I9


def _engine(pd_level=0.01, need=60, H=13, discount=True, balance_sheet="static", dpd30=0.0, s2_0=0.15, s3_0=0.02,
            revolving=False, shock=1.0):
    idx = range(need)
    pdA = np.full(need, pd_level)
    pdA[3:10] *= shock
    pdQ = pd.Series(1 - (1 - pdA) ** 0.25, index=idx)
    lgd_star = pd.Series(0.25, index=idx)
    cure = pd.Series(0.3, index=idx)
    ratio = pdA / pd_level if pd_level > 0 else np.ones(need)
    from creditloss.modelling.losses import stage2_share
    s2t = stage2_share(ratio, s2_0, 3.0)
    return I9.run_stages(need=need, H=H, pdQ=pdQ, lgd_star=lgd_star, cure=cure, s2_target=s2t, gca0=1e9,
                         ead_factor=1.0, s2_0=s2_0, s3_0=s3_0,
                         seg_cfg={"contract_rate": 0.06, "dpd30_share": dpd30, "contractual_term_years": 3},
                         ifrs={"discount": discount, "lifetime_years": 5}, bal_rel_existing=pd.Series(np.linspace(1, .5, need)),
                         revolving=revolving, repay_q=0.03, balance_sheet=balance_sheet,
                         bal_path=pd.Series(1e9, index=idx))


def test_survival_ecl_matches_hand_calculation():
    pd_q = np.array([0.1, 0.2])
    got = I9.survival_ecl(pd_q, np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.array([1.0, 0.9]))
    assert got == pytest.approx(0.1 * 0.5 + 0.9 * 0.2 * 0.5 * 0.9)


def test_static_balance_sheet_conserves_gross_loans():
    p = _engine(shock=4.0)
    total = p.gca_s1 + p.gca_s2 + p.gca_s3
    assert np.allclose(total, total.iloc[0], rtol=1e-9)


def test_pcl_is_change_in_allowance_plus_writeoffs():
    p = _engine(shock=3.0)
    expect = p.allowance_total.diff() + p.net_writeoffs
    assert np.allclose(p.pcl.iloc[1:], expect.iloc[1:])


def test_rollforwards_close():
    p = _engine(shock=3.0)
    a = I9.allowance_rollforward(p)
    assert np.allclose(a.loc["Closing allowance", ["Stage 1", "Stage 2", "Stage 3"]],
                       p.iloc[-1][["ecl_stage1", "ecl_stage2", "ecl_stage3"]].to_numpy())
    assert a.drop(index="Closing allowance").sum().round(6).equals(a.loc["Closing allowance"].round(6))
    g = I9.gca_rollforward(p)
    assert abs(g.loc["Other (rounding)", "Total"]) < 1e-3          # every movement of gross loans identified
    for line in ["Transfer to Stage 1", "Transfer to Stage 2", "Transfer to Stage 3"]:
        assert g.loc[line, "Total"] == pytest.approx(0, abs=1e-3)   # transfers move loans, they do not create them


def test_stress_raises_stage2_and_allowance():
    base, stress = _engine(shock=1.0), _engine(shock=4.0)
    assert stress.stage2_share.max() > base.stage2_share.max() + 0.05
    assert stress.allowance_total.max() > base.allowance_total.max()


def test_dpd30_backstop_floors_stage2():
    p = _engine(dpd30=0.30)
    assert (p.stage2_share.iloc[1:] >= 0.30 - 1e-9).all()


def test_discounting_lowers_ecl():
    assert _engine(discount=True).allowance.iloc[0] < _engine(discount=False).allowance.iloc[0]


def test_lifetime_exceeds_twelve_month_coverage():
    p = _engine()
    assert (p.cov_s2 > p.cov_s1).all()


def test_no_default_risk_no_ecl():
    p = _engine(pd_level=0.0, s3_0=0.0)
    assert p[["ecl_stage1", "ecl_stage2", "ecl_stage3"]].abs().to_numpy().max() == pytest.approx(0)
