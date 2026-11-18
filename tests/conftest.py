"""Shared fixtures: an end-to-end run of each shipped configuration into a temporary output folder."""
from __future__ import annotations

import pytest
import yaml

from creditloss import PROJECT_ROOT
from creditloss.scenarios.run import run
from creditloss.utils.config import load_config


def _run(name: str, tmp_path_factory) -> tuple[dict, dict]:
    raw = yaml.safe_load(open(PROJECT_ROOT / "config" / f"{name}.yaml"))
    out = tmp_path_factory.mktemp(name)
    raw["paths"]["outputs"] = str(out)
    raw["paths"]["cache"] = str(PROJECT_ROOT / ".cache" / name)        # shared selection cache keeps tests fast
    champions = PROJECT_ROOT / "outputs" / name / "champions.yaml"
    if champions.exists():
        raw.setdefault("pd_model", {})["champion_by_segment"] = yaml.safe_load(open(champions))
    f = out / f"{name}.yaml"
    for k, v in raw["paths"].items():
        if not str(v).startswith("/"):
            raw["paths"][k] = str(PROJECT_ROOT / v)
    yaml.safe_dump(raw, open(f, "w"))
    cfg = load_config(f)
    return cfg, run(cfg, log=lambda *a: None)


@pytest.fixture(scope="session")
def eqb(tmp_path_factory):
    return _run("eqb_public", tmp_path_factory)


@pytest.fixture(scope="session")
def sample(tmp_path_factory):
    return _run("sample", tmp_path_factory)
