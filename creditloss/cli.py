"""Command-line interface.

    creditloss simulate                    # write the simulated sample book to data/sample/
    creditloss build-data                  # rebuild data/eqb_public/ from the raw public files
    creditloss run --config CONFIG [--backtest]
    creditloss backtest --config CONFIG
    creditloss report --config CONFIG      # HTML report for the last run of CONFIG
    creditloss extensions --config CONFIG  # panel, payment shock, loss distribution, BVAR, PCL backtest
    creditloss app                         # launch the Streamlit app
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from creditloss import PROJECT_ROOT, __version__

DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "sample.yaml")


def _report(config: str) -> Path:
    from creditloss.utils.config import load_config
    cfg = load_config(config)
    out = cfg["paths"]["outputs"]
    if Path(config).stem == "eqb_public":
        from creditloss.reporting import eqb_report as rep
    else:
        from creditloss.reporting import methodology_report as rep
    return rep.main(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="creditloss", description="Scenario credit-loss projection and IFRS 9 ECL.")
    ap.add_argument("--version", action="version", version=f"creditloss {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="write the simulated sample book")
    s.add_argument("--out", default=str(PROJECT_ROOT / "data" / "sample"))
    s.add_argument("--seed", type=int, default=20261026)
    sub.add_parser("build-data", help="rebuild the public-data inputs from data/eqb_public/raw")
    r = sub.add_parser("run", help="project losses, allowance and provisions")
    r.add_argument("--config", default=DEFAULT_CONFIG)
    r.add_argument("--backtest", action="store_true", help="run the out-of-sample backtest and pick champions first")
    r.add_argument("--report", action="store_true", help="also write the HTML report")
    b = sub.add_parser("backtest", help="out-of-sample backtest of the PD models")
    b.add_argument("--config", default=DEFAULT_CONFIG)
    rp = sub.add_parser("report", help="HTML report for the last run")
    rp.add_argument("--config", default=DEFAULT_CONFIG)
    e = sub.add_parser("extensions", help="panel and payment shock, loss distribution, BVAR scenarios, PCL backtest")
    e.add_argument("--config", default=str(PROJECT_ROOT / "config" / "eqb_public.yaml"))
    e.add_argument("--draws", type=int, default=400, help="Monte Carlo draws for the loss distribution")
    a = sub.add_parser("app", help="launch the Streamlit app")
    a.add_argument("--port", default="8501")
    args = ap.parse_args(argv)

    if args.cmd == "simulate":
        from creditloss.data import simulate
        simulate.main(args.out, args.seed)
    elif args.cmd == "build-data":
        from creditloss.data import build_public
        build_public.main()
    elif args.cmd == "run":
        from creditloss.pipelines.projection import run_projection
        run_projection(args.config, args.backtest)
        if args.report:
            print("Report:", _report(args.config))
    elif args.cmd == "backtest":
        from creditloss.validation.backtest import run_backtest
        run_backtest(args.config)
    elif args.cmd == "report":
        print("Report:", _report(args.config))
    elif args.cmd == "extensions":
        from creditloss.extensions.report import run as run_ext
        print("Extensions:", run_ext(args.config, n_draws=args.draws))
    elif args.cmd == "app":
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "streamlit_app.py"),
                                "--server.port", args.port])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
