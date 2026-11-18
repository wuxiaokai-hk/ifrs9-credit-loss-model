"""Load the YAML configuration, fill defaults, resolve paths, check required names."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

DEFAULTS = {
    "pd_model": {"transform": "logit", "include_ar": True, "max_drivers": 3, "lags": [0, 1, 2, 3, 4],
                 "variable_transforms": ["level", "d4", "yoy"], "screen_top_n": 40, "cluster_corr": 0.85,
                 "require_sign_priors": True, "max_vif": 5, "min_obs": 60, "force_categories": [],
                 "bma": {"enabled": True, "top_k": 50}, "stress_weight": 0.0},
    "nco_model": {"enabled": True, "include_ar": True, "forward_lead": 1},
    "ifrs9": {"sicr_multiple": 3.0, "lifetime_years": 5, "no_cure_stage3": True,
              "allowance_cover_quarters": 4, "scenario_weights": {"baseline": 1.0},
              "reasonable_supportable_quarters": 12, "reversion_quarters": 8, "stage2_pd_multiple": 3.0,
              "stage2_cure_q": 0.10, "workout_quarters": 4, "discount": True},
    "horizon_quarters": 13,
    "oos": {"first_origin": "2006-12-31", "step_quarters": 4, "horizons": [4, 8, 12],
            "stress_windows": [], "selection_tolerance": 0.05},
    "random_seed": 0,
}

REQUIRED_PATHS = ["macro_history", "macro_scenarios", "portfolio_history", "portfolio_snapshot", "literature_table"]


def _merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with open(path) as fh:
        cfg = _merge(DEFAULTS, yaml.safe_load(fh))
    root = path.parent.parent  # config/ lives one level below the project root
    cfg["root"] = str(root)
    for key, rel in cfg["paths"].items():
        p = Path(rel)
        cfg["paths"][key] = str(p if p.is_absolute() else root / p)
    missing = [k for k in REQUIRED_PATHS if k not in cfg["paths"]]
    if missing:
        raise KeyError(f"config paths missing: {missing}")
    for name in ("outputs", "cache"):
        Path(cfg["paths"].setdefault(name, str(root / name))).mkdir(parents=True, exist_ok=True)
    w = cfg["ifrs9"]["scenario_weights"]
    if abs(sum(w.values()) - 1) > 1e-6:
        raise ValueError(f"ifrs9.scenario_weights must sum to 1, got {sum(w.values()):.3f}")
    return cfg
