"""Read the four inputs and put them on a common quarterly index."""
from __future__ import annotations

import pandas as pd


def _quarter_end(s: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(s).dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()


def add_constructed(macro: pd.DataFrame) -> pd.DataFrame:
    """Series the literature uses that Moody's does not publish directly."""
    m = macro.copy()
    if {"RMORT5", "RNEUTRAL"} <= set(m.columns):
        m["MORTGAP"] = m["RMORT5"] - m["RNEUTRAL"]
    return m


def load_macro_history(cfg: dict) -> pd.DataFrame:
    c = cfg["columns"]["date"]
    df = pd.read_csv(cfg["paths"]["macro_history"])
    df[c] = _quarter_end(df[c])
    return add_constructed(df.set_index(c).sort_index())


def load_scenarios(cfg: dict) -> dict[str, pd.DataFrame]:
    c = cfg["columns"]["date"]
    df = pd.read_csv(cfg["paths"]["macro_scenarios"])
    df[c] = _quarter_end(df[c])
    out = {}
    for key, name in cfg["scenarios"].items():
        sub = df[df["scenario"] == name].drop(columns="scenario").set_index(c).sort_index()
        if sub.empty:
            raise ValueError(f"scenario '{name}' (config key {key}) not found in {cfg['paths']['macro_scenarios']}")
        out[key] = add_constructed(sub)
    return out


def load_portfolio_history(cfg: dict) -> dict[str, pd.DataFrame]:
    cols = cfg["columns"]
    df = pd.read_csv(cfg["paths"]["portfolio_history"])
    df[cols["date"]] = _quarter_end(df[cols["date"]])
    return {seg: g.drop(columns=cols["segment"]).set_index(cols["date"]).sort_index()
            for seg, g in df.groupby(cols["segment"]) if seg in cfg["segments"]}


def load_snapshot(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(cfg["paths"]["portfolio_snapshot"])


def splice(history: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    """History up to the jump-off, then the scenario path (needed for lags)."""
    cols = [c for c in history.columns if c in scenario.columns]
    return pd.concat([history[cols], scenario.loc[scenario.index > history.index.max(), cols]])
