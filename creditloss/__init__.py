"""creditloss: scenario-conditional credit-loss projection and IFRS 9 expected credit loss.

Modules
    data        loading, feature transforms, public-data builder, simulated sample data
    modelling   PD satellites (selection, BMA), LGD, exposure and losses, IFRS 9 stage engine, NCO check
    scenarios   end-to-end scenario run, calibration, attribution, IFRS 7 disclosure tables
    validation  out-of-sample backtest with re-selection, Diebold-Mariano tests, stress capture, champion rule
    reporting   charts, Excel workbook, HTML reports
    pipelines   the projection pipeline used by the CLI and the app
"""
from pathlib import Path

__version__ = "1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
