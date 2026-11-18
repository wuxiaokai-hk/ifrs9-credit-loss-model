"""Configuration loading and the command-line interface."""
from __future__ import annotations

import pytest
import yaml

from creditloss import PROJECT_ROOT, __version__
from creditloss.cli import main
from creditloss.utils.config import load_config


def test_paths_resolve_to_project_root():
    cfg = load_config(PROJECT_ROOT / "config" / "sample.yaml")
    assert cfg["paths"]["macro_history"].startswith(str(PROJECT_ROOT))


def test_weights_must_sum_to_one(tmp_path):
    raw = yaml.safe_load(open(PROJECT_ROOT / "config" / "sample.yaml"))
    raw["ifrs9"]["scenario_weights"] = {"baseline": 0.5, "severe": 0.3}
    (tmp_path / "config").mkdir()
    f = tmp_path / "config" / "bad.yaml"
    yaml.safe_dump(raw, open(f, "w"))
    with pytest.raises(ValueError, match="sum to 1"):
        load_config(f)


def test_cli_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_simulated_data_reproducible(tmp_path):
    main(["simulate", "--out", str(tmp_path)])
    import pandas as pd
    a = pd.read_csv(tmp_path / "portfolio_history.csv")
    b = pd.read_csv(PROJECT_ROOT / "data" / "sample" / "portfolio_history.csv")
    pd.testing.assert_frame_equal(a, b)
