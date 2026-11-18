"""PD satellite models: screen -> cluster -> constrained exhaustive search -> BMA.

The engine is the balance-forecasting search adapted to a default *rate*:
  * target is logit(annualized default rate), so projected PDs stay in (0, 1);
  * an AR(1) term is allowed, as in the BoC TDSA generic equation (TR-128, Box 1),
    and models are ranked on a *dynamic* simulation, i.e. the model must rebuild
    the whole default-rate path from macro drivers alone, the way it will have
    to under a scenario (Fed: "stable, performs under stress");
  * every driver must carry the sign the literature table expects;
  * Bayesian model averaging over the admissible set (Gross & Poblacion 2015,
    ECB WP 1845) is reported next to the single best model, because hand-picked
    equations tend to understate stress responses.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..data.features import inverse_target


@dataclass
class Spec:
    name: str
    drivers: list[str]
    include_ar: bool
    params: dict = field(default_factory=dict)       # name -> coefficient
    tstats: dict = field(default_factory=dict)        # HAC t-stats
    sigma: float = np.nan
    bic: float = np.nan
    dyn_rmse: float = np.nan                          # pp of annual PD, dynamic simulation
    r2: float = np.nan
    nobs: int = 0
    sample: tuple = ()
    source: str = "data-driven"
    sign_ok: bool = True                              # every driver carries its literature sign

    def label(self) -> str:
        ar = "AR(1) + " if self.include_ar else ""
        return ar + (" + ".join(self.drivers) if self.drivers else "constant")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _design(y: pd.Series, X: pd.DataFrame, drivers: list[str], include_ar: bool) -> tuple[pd.Series, pd.DataFrame]:
    parts = [X[drivers]] if drivers else []
    if include_ar:
        parts.append(y.shift(1).rename("AR1"))
    D = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=y.index)
    D = sm.add_constant(D, has_constant="add")
    ok = D.notna().all(axis=1) & y.notna()
    return y[ok], D[ok]


def _fast_ols(yv: np.ndarray, Xv: np.ndarray):
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ beta
    n, k = Xv.shape
    s2 = resid @ resid / max(n - k, 1)
    try:
        cov = s2 * np.linalg.inv(Xv.T @ Xv)
    except np.linalg.LinAlgError:
        return beta, np.full(k, np.nan), resid
    return beta, beta / np.sqrt(np.clip(np.diag(cov), 1e-18, None)), resid


def dynamic_path(spec_params: dict, include_ar: bool, X: pd.DataFrame, y0: float, index: pd.Index) -> pd.Series:
    """Simulate the transformed target over `index`, feeding back its own forecasts."""
    c = spec_params.get("const", 0.0)
    rho = spec_params.get("AR1", 0.0) if include_ar else 0.0
    betas = {k: v for k, v in spec_params.items() if k not in ("const", "AR1")}
    xb = X.loc[index, list(betas)].to_numpy() @ np.array(list(betas.values())) if betas else np.zeros(len(index))
    out = np.empty(len(index))
    prev = y0
    for i, v in enumerate(c + xb):
        prev = v + rho * prev
        out[i] = prev
    return pd.Series(out, index=index)


def _dyn_rmse(params, include_ar, X, y, idx, how) -> float:
    if len(idx) < 8:
        return np.nan
    sim = dynamic_path(params, include_ar, X, y.loc[idx[0]], idx[1:])
    err = inverse_target(sim.to_numpy(), how) - inverse_target(y.loc[idx[1:]].to_numpy(), how)
    return float(np.sqrt(np.mean(err ** 2)))


def fit_spec(y: pd.Series, X: pd.DataFrame, drivers: list[str], include_ar: bool, how: str,
             name: str = "", source: str = "data-driven", weights: pd.Series | None = None) -> Spec:
    yy, D = _design(y, X, drivers, include_ar)
    if weights is not None:
        res = sm.WLS(yy, D, weights=weights.reindex(yy.index).fillna(1.0)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    else:
        res = sm.OLS(yy, D).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    params = res.params.to_dict()
    spec = Spec(name=name or " + ".join(drivers), drivers=list(drivers), include_ar=include_ar,
                params=params, tstats=res.tvalues.to_dict(), sigma=float(np.sqrt(res.scale)),
                bic=float(res.bic), r2=float(res.rsquared), nobs=int(res.nobs),
                sample=(yy.index.min(), yy.index.max()), source=source)
    spec.dyn_rmse = _dyn_rmse(params, include_ar, X, y, yy.index, how)
    return spec


def _vif_ok(Xv: np.ndarray, max_vif: float) -> bool:
    if Xv.shape[1] < 2:
        return True
    c = np.corrcoef(Xv, rowvar=False)
    try:
        return bool(np.all(np.diag(np.linalg.inv(c)) < max_vif))
    except np.linalg.LinAlgError:
        return False


# ---------------------------------------------------------------------------
# 1. univariate screen (controls for the AR term)
# ---------------------------------------------------------------------------
def screen(y: pd.Series, X: pd.DataFrame, meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pm = cfg["pd_model"]
    rows = []
    ylag = y.shift(1)
    for f in X.columns:
        D = pd.concat([X[f], ylag], axis=1) if pm["include_ar"] else X[[f]]
        ok = D.notna().all(axis=1) & y.notna()
        if ok.sum() < pm["min_obs"]:
            continue
        Xv = np.column_stack([np.ones(ok.sum()), D[ok].to_numpy()])
        beta, t, _ = _fast_ols(y[ok].to_numpy(), Xv)
        sign = meta.at[f, "sign"]
        sign_ok = (sign is None or pd.isna(sign)) or np.sign(beta[1]) == sign
        rows.append({"feature": f, "beta": beta[1], "t": t[1], "abs_t": abs(t[1]), "sign_ok": bool(sign_ok),
                     "mnemonic": meta.at[f, "mnemonic"], "category": meta.at[f, "category"],
                     "has_prior": not (sign is None or pd.isna(sign))})
    s = pd.DataFrame(rows, columns=["feature", "beta", "t", "abs_t", "sign_ok", "mnemonic", "category", "has_prior"])
    s = s.sort_values("abs_t", ascending=False)
    if pm["require_sign_priors"]:
        s = s[s["has_prior"]]
    s = s[s["sign_ok"]]
    # best lag/transform per mnemonic first, then the overall top-N
    best = s.groupby("mnemonic", sort=False).head(3)
    return best.head(pm["screen_top_n"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. redundancy clustering (greedy, |corr| threshold, keep highest |t|)
# ---------------------------------------------------------------------------
def cluster(screened: pd.DataFrame, X: pd.DataFrame, thr: float, meta: pd.DataFrame | None = None,
            primary_ratio: float = 0.8) -> pd.DataFrame:
    """Greedy clustering; the representative is the highest-|t| member, unless a
    'primary' literature driver (e.g. the unemployment rate for household books)
    sits in the cluster with at least `primary_ratio` of the leader's |t|. That keeps
    the model in the variables a business audience and the literature expect when
    the data cannot tell near-collinear series apart."""
    reps, members = [], {}
    C = X[screened["feature"]].corr().abs()
    for f in screened["feature"]:  # already sorted by |t|
        hit = next((r for r in reps if C.at[f, r] >= thr or f.split("|")[0] == r.split("|")[0] and C.at[f, r] >= 0.6), None)
        if hit is None:
            reps.append(f)
            members[f] = [f]
        else:
            members[hit].append(f)
    t = screened.set_index("feature")["abs_t"]
    final = {}
    for r, mem in members.items():
        rep = r
        if meta is not None and not bool(meta.at[r, "primary"]):
            prim = [m for m in mem if bool(meta.at[m, "primary"]) and t[m] >= primary_ratio * t[r]]
            if prim:
                rep = max(prim, key=lambda m: t[m])
        final[rep] = [m for m in mem if m != rep]
    out = screened[screened["feature"].isin(final)].copy()
    out["cluster_members"] = out["feature"].map(lambda r: ", ".join(final[r]))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. constrained exhaustive search
# ---------------------------------------------------------------------------
def search(y: pd.Series, X: pd.DataFrame, meta: pd.DataFrame, reps: list[str], cfg: dict) -> pd.DataFrame:
    pm = cfg["pd_model"]
    how = pm["transform"]
    rows = []
    for k in range(1, pm["max_drivers"] + 1):
        for combo in itertools.combinations(reps, k):
            mns = [c.split("|")[0] for c in combo]
            if len(set(mns)) < k:
                continue
            cats = set(meta.loc[list(combo), "category"])
            if pm["force_categories"] and not set(pm["force_categories"]) <= cats:
                continue
            yy, D = _design(y, X, list(combo), pm["include_ar"])
            if len(yy) < pm["min_obs"]:
                continue
            Xv = D.to_numpy()
            if not _vif_ok(Xv[:, 1:1 + k], pm["max_vif"]):
                continue
            beta, t, resid = _fast_ols(yy.to_numpy(), Xv)
            signs = meta.loc[list(combo), "sign"].to_numpy()
            b = beta[1:1 + k]
            if pm["require_sign_priors"] and np.any(np.sign(b) != signs):
                continue
            if np.any(np.abs(t[1:1 + k]) < 1.65):
                continue
            if pm["include_ar"] and not (0 <= beta[-1] < 0.98):
                continue  # explosive or oscillating persistence is not usable for projection
            n = len(yy)
            bic = n * np.log(resid @ resid / n) + Xv.shape[1] * np.log(n)
            params = dict(zip(D.columns, beta))
            rows.append({"drivers": combo, "k": k, "bic": bic,
                         "dyn_rmse": _dyn_rmse(params, pm["include_ar"], X, y, yy.index, how),
                         "rho": params.get("AR1", 0.0), "categories": ", ".join(sorted(cats))})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return res.sort_values(["dyn_rmse", "bic"]).reset_index(drop=True)


def pick(results: pd.DataFrame, tol: float) -> tuple:
    """Simplest model within `tol` of the best dynamic RMSE (parsimony rule)."""
    best = results["dyn_rmse"].min()
    near = results[results["dyn_rmse"] <= best * (1 + tol)]
    return tuple(near.sort_values(["k", "dyn_rmse"]).iloc[0]["drivers"])


def bma_weights(results: pd.DataFrame, top_k: int) -> pd.DataFrame:
    r = results.nsmallest(top_k, "bic").copy()
    w = np.exp(-0.5 * (r["bic"] - r["bic"].min()))
    r["weight"] = w / w.sum()
    return r


# ---------------------------------------------------------------------------
# literature specifications (fixed, re-estimated on our data)
# ---------------------------------------------------------------------------
LITERATURE_SPECS = {
    "household": {
        # BoC SAP 2026-12 eq. (1): UR(t-6m), HPI y/y (t-12m), mortgage rate - neutral (t-12m),
        # core CPI (t), output gap (t-12m); OLS, no AR term.
        "BoC SAP 2026-12 (arrears regression)": (["LBR_ON|level|L2", "HPI_ON|yoy|L4", "MORTGAP|level|L4",
                                                  "CPIX_CA|level|L0", "GAP_CA|level|L4"], False),
        # BoC TDSA (TR-128) generic form: own lag + UR change + HPI growth
        "BoC TDSA-style ARDL": (["LBR_ON|d4|L0", "HPI_ON|yoy|L0"], True),
        # ECB/EBA satellite core: unemployment, house prices, rates
        "ECB/EBA core satellite": (["LBR_ON|level|L1", "HPI_ON|yoy|L2", "RMORT5|level|L2"], True),
    },
    "business": {
        "BoC CDM-style (UR, spread, GDP)": (["LBR_ON|level|L1", "SPRBBB|level|L1", "GDPR_ON|yoy|L1"], True),
        "Fed CRE-style (prices, UR)": (["PCRE_ON|yoy|L2", "LBR_ON|level|L2"], True),
    },
}


def benchmark_ar(y: pd.Series, X: pd.DataFrame, how: str) -> Spec:
    return fit_spec(y, X, [], True, how, name="AR(1) benchmark", source="benchmark")


def literature_specs(cfg: dict, book: str) -> dict:
    """Config `pd_model.literature_specs.<book>` overrides the built-in table.
    Each entry: name -> {drivers: [...], ar: bool}."""
    conf = cfg["pd_model"].get("literature_specs", {}).get(book)
    if conf is None:
        return LITERATURE_SPECS.get(book, {})
    return {nm: (v["drivers"], bool(v.get("ar", True))) for nm, v in conf.items()}


def run_selection(y: pd.Series, X: pd.DataFrame, meta: pd.DataFrame, cfg: dict, book: str) -> dict:
    pm = cfg["pd_model"]
    how = pm["transform"]
    scr = screen(y, X, meta, cfg)
    reps = cluster(scr, X, pm["cluster_corr"], meta, pm.get("primary_ratio", 0.8))
    results = search(y, X, meta, list(reps["feature"]), cfg)
    out = {"screen": scr, "clusters": reps, "search": results, "specs": {}}
    weights = None
    if pm.get("stress_weight", 0) > 0:
        z = (y - y.mean()) / y.std()
        weights = 1 + pm["stress_weight"] * (z > 1).astype(float)
    if not results.empty:
        best = pick(results, cfg["oos"]["selection_tolerance"])
        out["specs"]["Data-driven search"] = fit_spec(y, X, list(best), pm["include_ar"], how,
                                                      name="Data-driven search", weights=weights)
        if pm["bma"]["enabled"]:
            bw = bma_weights(results, pm["bma"]["top_k"])
            out["bma"] = bw
            out["bma_specs"] = [(fit_spec(y, X, list(r.drivers), pm["include_ar"], how, source="bma"), r.weight)
                                for r in bw.itertuples()]
    for nm, (drv, ar) in literature_specs(cfg, book).items():
        if all(d in X.columns for d in drv):
            out["specs"][nm] = fit_spec(y, X, drv, ar, how, name=nm, source="literature")
    out["specs"]["AR(1) benchmark"] = benchmark_ar(y, X, how)
    for sp in out["specs"].values():
        sp.sign_ok = all(np.sign(sp.params[d]) == meta.at[d, "sign"] for d in sp.drivers
                         if d in meta.index and pd.notna(meta.at[d, "sign"]))
    return out


def project(spec: Spec, X_full: pd.DataFrame, y_hist: pd.Series, index: pd.Index) -> pd.Series:
    """Projected transformed target over `index` (scenario quarters), seeded at jump-off."""
    y0 = y_hist.dropna().iloc[-1]
    return dynamic_path(spec.params, spec.include_ar, X_full, y0, index)


def project_bma(bma_specs, X_full, y_hist, index) -> pd.Series:
    acc = None
    for sp, w in bma_specs:
        p = project(sp, X_full, y_hist, index) * w
        acc = p if acc is None else acc + p
    return acc
