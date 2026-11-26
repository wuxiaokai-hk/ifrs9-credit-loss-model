"""Shared plumbing: load a configuration's inputs and run results, and turn any proxy-rate path into losses
with the same engine and calibration as the main projection."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from creditloss.data import features as F
from creditloss.data.load import load_macro_history, load_portfolio_history, load_scenarios, load_snapshot, splice
from creditloss.modelling import losses as L
from creditloss.scenarios.run import _pd_path, calibration
from creditloss.utils.config import load_config


@dataclass
class Context:
    config: str
    cfg: dict = field(init=False)
    res: dict = field(init=False)

    def __post_init__(self):
        self.cfg = load_config(self.config)
        out = Path(self.cfg["paths"]["outputs"])
        f = out / "run.pkl"
        try:
            self.res = pickle.loads(f.read_bytes())
        except Exception:                       # missing, or written by another pandas version: rebuild
            from creditloss.pipelines.projection import run_projection
            self.res = run_projection(self.config)
        champ = self.res["champions"]
        self.cfg["pd_model"]["champion_by_segment"] = champ
        for seg, pd_ in self.res["jumpoff_pd"].items():     # use the calibrated jump-off PDs (cards solved to ACL)
            if pd_ is not None:
                self.cfg["segments"][seg]["jumpoff_pd"] = pd_
                self.cfg["segments"][seg]["target_acl"] = None
        self.macro = load_macro_history(self.cfg)
        self.scen = load_scenarios(self.cfg)
        self.port = load_portfolio_history(self.cfg)
        self.snap = load_snapshot(self.cfg)
        self.lit = F.load_literature(self.cfg)
        self.how = self.cfg["pd_model"]["transform"]
        self.cidx = self.cfg.get("collateral_index", "HPI_ON")
        self.hpi_jump = float(self.macro[self.cidx].dropna().iloc[-1])
        self.H = self.cfg["horizon_quarters"]
        self.dcol = self.cfg["columns"]["default_rate"]

    @property
    def segments(self) -> list[str]:
        return list(self.cfg["segments"])

    def proxy_path(self, seg: str, scen: pd.DataFrame, spec_name: str | None = None) -> pd.Series:
        """Champion (or named model) path of the proxy rate, fraction, over the scenario quarters."""
        sc = self.cfg["segments"][seg]
        sel = self.res["sels"][seg]
        full = splice(self.macro, scen)
        X_full, _ = F.build_features(full, self.lit, self.cfg, sc.get("sign_priors", "household"), sc.get("candidates"))
        return _pd_path(sel, spec_name or self.res["champions"][seg], X_full, scen.index, self.how)

    def losses(self, seg: str, proxy: pd.Series, scen: pd.DataFrame) -> pd.DataFrame:
        """Quarterly loss path for one segment given a proxy-rate path (fraction) and the scenario's house prices."""
        sc = self.cfg["segments"][seg]
        pd_jump, pd_lr, scale = calibration(sc, self.port[seg][self.dcol])
        bk = self.snap[self.snap.segment == seg]
        full = splice(self.macro, scen)
        hpi = full.loc[scen.index, self.cidx]
        return L.project_segment(seg, sc, bk, pd_jump, (proxy * scale).clip(upper=0.99), self.hpi_jump, hpi, self.cfg, pd_lr)

    def cum_loss(self, path: pd.DataFrame) -> float:
        g = path[(path.q > 0) & (path.q <= self.H)]
        return float(g.credit_loss.sum())
