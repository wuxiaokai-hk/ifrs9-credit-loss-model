"""Out-of-sample backtest of the PD satellite models, with the whole selection re-run at every origin.

    creditloss backtest --config config/sample.yaml
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pandas as pd

from creditloss.data import features as F
from creditloss.data.load import load_macro_history, load_portfolio_history
from creditloss.utils.config import load_config
from creditloss.validation import oos as O


def run_backtest(config: str) -> dict:
    """Expanding-window backtest for every segment; results are cached per segment and settings."""
    cfg = load_config(config)
    macro = load_macro_history(cfg)
    port = load_portfolio_history(cfg)
    lit = F.load_literature(cfg)
    cache = Path(cfg["paths"]["cache"])
    errs = []
    for seg, sc in cfg["segments"].items():
        rate = port[seg][cfg["columns"]["default_rate"]]
        key = hashlib.md5(pickle.dumps((seg, rate.round(10).to_dict(), cfg["pd_model"], cfg["oos"], sc.get("candidates"),
                                        macro.round(8).to_dict()))).hexdigest()[:16]
        f = cache / f"oos_{seg}_{key}.pkl"
        if f.exists():
            e = pickle.loads(f.read_bytes())
        else:
            print(f"OOS backtest: {seg} ...", flush=True)
            e = O.run_segment(seg, cfg, macro, rate, lit, sc.get("sign_priors", "household"), sc.get("candidates"))
            f.write_bytes(pickle.dumps(e))
        errs.append(e)
    err = pd.concat(errs, ignore_index=True)
    res = {"errors": err,
           "metrics": O.metrics(err, cfg["oos"]["horizons"]),
           "dm": O.diebold_mariano(err, cfg["oos"]["horizons"]),
           "stress": O.stress_capture(err, cfg["oos"]["stress_windows"])}
    m = res["metrics"]
    print("\nOut-of-sample RMSE (pp of annual PD), average over 1-12 quarters ahead")
    print(m[m.h.astype(str).str.startswith("1-")].pivot(index="model", columns="segment", values="rmse_pp")
          .round(3).to_string())
    return res

