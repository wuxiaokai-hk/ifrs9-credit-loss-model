"""Candidate driver construction: variable x transform x lag, with sign priors.

A feature name is "MNEMONIC|transform|Llag", e.g. "LBR_ON|level|L2".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit as _logit

TRANSFORMS = {
    "level": lambda s: s,
    "d4": lambda s: s - s.shift(4),
    "yoy": lambda s: 100 * (s / s.shift(4) - 1),
}
# Unmapped (no literature entry) series are usually trending levels: changes only.
UNMAPPED_TRANSFORMS = ["d4", "yoy"]
EXCLUDE = {"D_COVID", "RNEUTRAL"}


def load_literature(cfg: dict) -> pd.DataFrame:
    lit = pd.read_csv(cfg["paths"]["literature_table"]).set_index("mnemonic")
    lit["transforms"] = lit["transforms"].str.split("|")
    return lit


def sign_prior(lit: pd.DataFrame, mnemonic: str, transform: str, book: str) -> int | None:
    """+1 / -1 expected sign on default risk; 0 = no prior expected effect; None = unknown."""
    if mnemonic not in lit.index:
        return None
    raw = str(lit.at[mnemonic, f"sign_{book}"]).strip()
    s = {"+": 1, "-": -1, "0": 0}.get(raw)
    return s


def build_features(macro: pd.DataFrame, lit: pd.DataFrame, cfg: dict, book: str,
                   candidates: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (feature matrix, metadata). Metadata carries category and sign prior.

    `candidates` restricts the series a segment may draw on (e.g. US unemployment only
    for satellites estimated on US charge-off data)."""
    pm = cfg["pd_model"]
    cols, meta = {}, []
    for mn in macro.columns:
        if mn in EXCLUDE or (candidates and mn not in candidates):
            continue
        mapped = mn in lit.index
        trs = [t for t in (lit.at[mn, "transforms"] if mapped else UNMAPPED_TRANSFORMS) if t in pm["variable_transforms"]]
        for tr in trs:
            base = TRANSFORMS[tr](macro[mn].astype(float))
            sp = sign_prior(lit, mn, tr, book)
            if sp == 0:  # literature says no effect on this book: do not offer it
                continue
            for lag in pm["lags"]:
                name = f"{mn}|{tr}|L{lag}"
                cols[name] = base.shift(lag)
                meta.append({"feature": name, "mnemonic": mn, "transform": tr, "lag": lag,
                             "category": lit.at[mn, "category"] if mapped else "Unmapped",
                             "sign": sp, "mapped": mapped,
                             "primary": mapped and book in str(lit.at[mn, "primary"]).split("|")})
    return pd.DataFrame(cols, index=macro.index), pd.DataFrame(meta).set_index("feature")


def target_transform(rate_pct: pd.Series, how: str) -> pd.Series:
    p = (rate_pct / 100).clip(1e-6, 1 - 1e-6)
    if how == "logit":
        return pd.Series(_logit(p), index=rate_pct.index)
    if how == "log":
        return np.log(p)
    return rate_pct


def inverse_target(y: np.ndarray | pd.Series, how: str):
    if how == "logit":
        return 100 / (1 + np.exp(-y))
    if how == "log":
        return 100 * np.exp(y)
    return y
