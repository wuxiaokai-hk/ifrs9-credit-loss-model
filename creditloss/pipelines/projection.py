"""Full projection: PD model selection, optional out-of-sample backtest and champion choice, scenario
losses, IFRS 9 allowance and provisions, charts, Excel workbook and run artefacts.

    creditloss run --config config/sample.yaml [--backtest]
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from creditloss.reporting import report as R
from creditloss.scenarios.run import run
from creditloss.utils.config import load_config


def run_projection(config: str, with_oos: bool = False) -> dict:
    cfg = load_config(config)
    if not Path(cfg["paths"]["macro_history"]).exists():
        from creditloss.data import simulate
        print("No data found; simulating stand-in data.")
        simulate.main(str(Path(cfg["paths"]["macro_history"]).parent), cfg["random_seed"])
    out = Path(cfg["paths"]["outputs"])
    oos = None
    saved = out / "champions.yaml"
    if not with_oos and saved.exists() and not cfg["pd_model"].get("champion_by_segment"):
        # Reuse the champions chosen by the last out-of-sample backtest, so a plain run reproduces it.
        import yaml
        cfg["pd_model"]["champion_by_segment"] = yaml.safe_load(open(saved))
        print("Using the champions from the last backtest:", saved)
    if with_oos:
        # Validate first, then let the out-of-sample record choose each segment's champion.
        from creditloss.validation.backtest import run_backtest as oos_main
        from creditloss.validation.oos import choose_champion
        oos = oos_main(config)
        from creditloss.data import features as F
        from creditloss.data.load import load_macro_history, load_portfolio_history
        from creditloss.scenarios.run import select_models
        full_sels = select_models(cfg, load_macro_history(cfg), load_portfolio_history(cfg), F.load_literature(cfg),
                                  log=lambda *a: None)
        wrong_sign = {(seg, nm) for seg, sl in full_sels.items() for nm, sp in sl["specs"].items() if not sp.sign_ok}
        champ = choose_champion(oos["metrics"], oos["stress"], min_capture=cfg["oos"].get("champion_min_capture", 80),
                                max_capture=cfg["oos"].get("champion_max_capture", 200), exclude=wrong_sign)
        oos["wrong_sign"] = pd.DataFrame(sorted(wrong_sign), columns=["segment", "model"])
        oos["champions"] = champ
        cfg["pd_model"]["champion_by_segment"] = dict(zip(champ.segment, champ.champion))
        print("\nChampion PD model by segment (correct signs; lowest OOS RMSE with GFC capture in "
              f"[{cfg['oos'].get('champion_min_capture', 80)}, {cfg['oos'].get('champion_max_capture', 200)}]%):")
        print(champ.round(3).to_string(index=False))
    res = run(cfg)
    R._style()
    figs = [R.chart_pd_paths(res, cfg, out), R.chart_loss_rates(res, cfg, out), R.chart_lgd(res, cfg, out),
            R.chart_attribution(res, out), R.chart_model_range(res, out), R.chart_provisions(res, out),
            R.chart_stages(res, out)]
    if oos:
        figs.append(R.chart_oos(oos["metrics"], oos["stress"], out, res["champions"]))
    prior = None
    if not oos:   # keep the last out-of-sample record in the workbook
        files = {"metrics": "oos_metrics.csv", "stress": "oos_stress_capture.csv", "dm": "oos_dm.csv",
                 "champions": "oos_champions.csv"}
        prior = {k: pd.read_csv(out / f) for k, f in files.items() if (out / f).exists()} or None
    xl = R.write_excel(res, out, oos or prior)
    if oos:                                    # read by the app: last validated champions and the OOS record
        import yaml
        oos["metrics"].to_csv(out / "oos_metrics.csv", index=False)
        oos["stress"].to_csv(out / "oos_stress_capture.csv", index=False)
        oos["champions"].to_csv(out / "oos_champions.csv", index=False)
        if "dm" in oos:
            oos["dm"].to_csv(out / "oos_dm.csv", index=False)
        yaml.safe_dump(dict(res["champions"]), open(out / "champions.yaml", "w"))
    (out / "run.pkl").write_bytes(pickle.dumps({k: v for k, v in res.items()}))
    s = res["summary"]
    print("\nCumulative loss rate, % of average EAD, over", cfg["horizon_quarters"], "quarters")
    print(s[s.horizon_q == cfg["horizon_quarters"]].pivot(index="segment", columns="scenario",
                                                          values="loss_rate_pct").round(3).to_string())
    print(f"\nWrote {xl} and {len(figs)} charts to {out}")
    return res

