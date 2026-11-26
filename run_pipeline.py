"""Run the whole credit-loss pipeline with plain Python, no `creditloss` command needed.

    python run_pipeline.py                                   # EQB public-data book, validated models (~15 s)
    python run_pipeline.py --config config/sample.yaml       # simulated credit-union book
    python run_pipeline.py --backtest                        # re-run the out-of-sample backtest first (minutes)
    python run_pipeline.py --build-data --backtest --report  # everything, from the raw public files
    python run_pipeline.py --extensions                      # plus panel, payment shock, loss distribution,
                                                             # BVAR scenarios and the PCL backtest (~1 min)

Or edit the SETTINGS block below and just run `python run_pipeline.py` (handy in Spyder, VS Code or a notebook:
`%run run_pipeline.py`). Command-line flags override the settings.

Steps, in order:
    1. build-data   rebuild data/eqb_public/*.csv from the raw public files (only for the EQB config)
    2. backtest     out-of-sample test of every PD model; the champions it picks are saved and reused
    3. projection   scenario losses, IFRS 9 allowance and provisions, Excel workbook, charts
    4. report       HTML report
    +  extensions   outputs/eqb_public/extensions/ (HTML page, CSV tables)
Results go to the `outputs` folder named in the config (outputs/eqb_public/ or outputs/sample/).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # works without `pip install -e .`, as long as the dependencies are installed

# ---------------------------------------------------------------- SETTINGS
CONFIG = "config/eqb_public.yaml"      # or "config/sample.yaml", or your own config
BUILD_DATA = False                     # rebuild the public inputs from data/eqb_public/raw first
BACKTEST = False                       # re-validate the PD models (slow); False reuses the saved champions
REPORT = True                          # write the HTML report
EXTENSIONS = False                     # also run the model extensions (EQB config; about a minute)
# --------------------------------------------------------------------------


def summary_table(res: dict) -> str:
    """Cumulative credit losses and provisions by segment and scenario, C$ millions, at the full horizon."""

    s = res["summary"]
    s = s[s.horizon_q == s.horizon_q.max()]
    order = [c for c in ["upside", "baseline", "downside", "severe"] if c in set(s.scenario)]
    order += [c for c in s.scenario.unique() if c not in order]
    loss = s.pivot(index="segment", columns="scenario", values="cum_loss")[order] / 1e6
    loss.loc["Total credit losses"] = loss.sum()
    pcl = s.groupby("scenario").cum_pcl.sum()[order] / 1e6
    loss.loc["IFRS 9 provisions (PCL)"] = pcl
    out = [f"Cumulative over {int(s.horizon_q.max())} quarters, C$ millions",
           loss.round(1).to_string()]
    if "ifrs9" in res:
        pw = res["ifrs9"]["pw_by_stage"]
        tot = pw[["ecl_stage1", "ecl_stage2", "ecl_stage3"]].sum() / 1e6
        out.append(f"\nAllowance today, probability-weighted: C${pw.allowance_total.sum() / 1e6:,.1f}m "
                   f"(Stage 1 {tot.ecl_stage1:,.1f} · Stage 2 {tot.ecl_stage2:,.1f} · Stage 3 {tot.ecl_stage3:,.1f})")
    out.append("PD model per segment: " + ", ".join(f"{k}: {v}" for k, v in res["champions"].items()))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG, help=f"YAML config (default: {CONFIG})")
    ap.add_argument("--build-data", action="store_true", default=BUILD_DATA, help="rebuild public inputs first")
    ap.add_argument("--backtest", action="store_true", default=BACKTEST, help="re-run the out-of-sample backtest")
    ap.add_argument("--report", action=argparse.BooleanOptionalAction, default=REPORT, help="write the HTML report")
    ap.add_argument("--extensions", action=argparse.BooleanOptionalAction, default=EXTENSIONS,
                    help="also run the extensions: provincial panel, payment shock, loss distribution, BVAR, PCL backtest")
    args = ap.parse_args(argv)

    config = Path(args.config)
    config = config if config.is_absolute() else ROOT / config
    if not config.exists():
        sys.exit(f"Config not found: {config}. Choose one of: "
                 + ", ".join(p.name for p in sorted((ROOT / "config").glob("*.yaml")) if not p.name.startswith(".")))

    t0 = time.time()
    if args.build_data:
        if config.stem != "eqb_public":
            print("--build-data applies to config/eqb_public.yaml only; skipped.")
        else:
            print("[1/4] Rebuilding the public-data inputs …")
            from creditloss.data import build_public
            build_public.main()

    print(f"[{'2-3' if args.backtest else '3'}/4] Projection{' with out-of-sample backtest' if args.backtest else ''}: "
          f"{config.relative_to(ROOT) if config.is_relative_to(ROOT) else config}")
    from creditloss.pipelines.projection import run_projection
    res = run_projection(str(config), args.backtest)

    if args.report:
        print("[4/4] Writing the HTML report …")
        from creditloss.cli import _report
        print("Report:", _report(str(config)))

    if args.extensions:
        if config.stem != "eqb_public":
            print("--extensions uses the EQB public data; skipped for this config.")
        else:
            print("[+] Running the model extensions …")
            from creditloss.extensions.report import run as run_ext
            print("Extensions:", run_ext(str(config)))

    print("\n" + summary_table(res))
    from creditloss.utils.config import load_config
    print(f"\nOutputs in {load_config(config)['paths']['outputs']}  ·  done in {time.time() - t0:.0f} s")
    return res


if __name__ == "__main__":
    main()
