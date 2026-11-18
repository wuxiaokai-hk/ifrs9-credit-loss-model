"""End-to-end projection: select PD models, project PD/LGD/EAD by scenario, provisions, attribution."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import features as F
from ..data.load import load_macro_history, load_portfolio_history, load_scenarios, load_snapshot, splice
from ..modelling import ifrs9 as I9
from ..modelling import losses as L
from ..modelling import nco as NCO
from ..modelling import pd_satellite as PD


def _book(seg_cfg: dict) -> str:
    return seg_cfg.get("sign_priors", "household")


def _cache_key(*parts) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(pickle.dumps(p))
    return h.hexdigest()[:16]


def select_models(cfg, macro, port, lit, log=print) -> dict:
    out = {}
    cache = Path(cfg["paths"]["cache"])
    for seg, sc in cfg["segments"].items():
        book = _book(sc)
        y = F.target_transform(port[seg][cfg["columns"]["default_rate"]], cfg["pd_model"]["transform"])
        X, meta = F.build_features(macro, lit, cfg, book, sc.get("candidates"))
        key = _cache_key(seg, sc.get("candidates"), y.round(10).to_dict(), X.shape, list(X.columns), cfg["pd_model"], cfg["oos"]["selection_tolerance"],
                         X.sum().round(6).to_dict())
        f = cache / f"select_{seg}_{key}.pkl"
        if f.exists():
            sel = pickle.loads(f.read_bytes())
        else:
            log(f"  selecting PD model for {seg} ...")
            sel = PD.run_selection(y, X, meta, cfg, book)
            f.write_bytes(pickle.dumps(sel))
        sel["y"], sel["meta"] = y, meta
        out[seg] = sel
    return out


def _pd_path(sel, spec_name, X_full, idx, how):
    if spec_name == "BMA" and sel.get("bma_specs"):
        z = PD.project_bma(sel["bma_specs"], X_full, sel["y"], idx)
    else:
        z = PD.project(sel["specs"][spec_name], X_full, sel["y"], idx)
    return pd.Series(F.inverse_target(z.to_numpy(), how) / 100, index=idx)


def calibration(sc: dict, rate: pd.Series) -> tuple[float, float, float]:
    """(pd_jump, pd_long_run, scale) as fractions.

    Without `jumpoff_pd` the segment's own history is the PD (simulated data).
    With it, the history is a *proxy* (industry arrears or charge-off rate): the satellite
    projects the proxy and the bank's own jump-off PD is moved in proportion,
    PD_bank,t = PD_bank,0 x proxy_t / proxy_0  (EBA constrained bottom-up logic)."""
    last, mean = float(rate.dropna().iloc[-1]), float(rate.dropna().mean())
    if sc.get("jumpoff_pd") is None:
        return last / 100, mean / 100, 1.0
    scale = float(sc["jumpoff_pd"]) / last
    return float(sc["jumpoff_pd"]) / 100, mean * scale / 100, scale


def calibrate_to_acl(seg, sc, cfg, sel, spec_name, bk, rate, scen, macro, lit, hpi_jump, how, log=print) -> float:
    """Solve the bank jump-off PD (annual %) so that the probability-weighted performing-loan
    allowance at jump-off equals the allowance the bank reports for the segment. Used where the
    bank gives no default or impairment data for the book (here: newly acquired credit cards)."""
    target = float(sc["target_acl"])
    w = cfg["ifrs9"]["scenario_weights"]
    cidx = cfg.get("collateral_index", "HPI_ON")
    raw = {}
    for sname in w:
        full = splice(macro, scen[sname])
        X_full, _ = F.build_features(full, lit, cfg, _book(sc), sc.get("candidates"))
        raw[sname] = (_pd_path(sel, spec_name, X_full, scen[sname].index, how), full.loc[scen[sname].index, cidx])
    last, mean = float(rate.dropna().iloc[-1]), float(rate.dropna().mean())

    def acl(pd0):
        sc_ = dict(sc, jumpoff_pd=pd0)
        scale = pd0 / last
        tot = 0.0
        for sname, wt in w.items():
            pdp, hp = raw[sname]
            p = L.project_segment(seg, sc_, bk, pd0 / 100, (pdp * scale).clip(upper=0.99), hpi_jump, hp, cfg,
                                  mean * scale / 100)
            tot += wt * p.loc[p.q == 0, "allowance"].iloc[0]
        return tot

    lo, hi = 0.05, 30.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if acl(mid) < target else (lo, mid)
    log(f"  {seg}: jump-off PD calibrated to reported ACL {target/1e6:,.1f}m -> {mid:.2f}%")
    return mid


def run(cfg: dict, log=print) -> dict:
    macro = load_macro_history(cfg)
    scen = load_scenarios(cfg)
    port = load_portfolio_history(cfg)
    snap = load_snapshot(cfg)
    lit = F.load_literature(cfg)
    how = cfg["pd_model"]["transform"]
    champion = cfg["pd_model"].get("projection_model", "Data-driven search")
    dcol = cfg["columns"]["default_rate"]

    log("Selecting PD satellite models")
    sels = select_models(cfg, macro, port, lit, log)

    cidx = cfg.get("collateral_index", "HPI_ON")
    hpi_jump = float(macro[cidx].dropna().iloc[-1])
    growth_tbl = None
    ead_cfg = cfg.get("ead", {})
    if ead_cfg.get("balance_sheet") == "growth":
        gf = pd.read_csv(Path(cfg["root"]) / ead_cfg["growth_file"])  # columns: scenario, date, segment, growth
        growth_tbl = gf
    paths, model_range, nco_rows, pd_paths, champions = [], [], [], [], {}
    calibrated = {}
    for seg, sc in cfg["segments"].items():
        sel, book = sels[seg], _book(sc)
        bk = snap[snap["segment"] == seg]
        want = cfg["pd_model"].get("champion_by_segment", {}).get(seg, champion)
        spec_name = want if (want in sel["specs"] or (want == "BMA" and sel.get("bma_specs"))) else "Data-driven search"
        champions[seg] = spec_name
        if sc.get("target_acl") is not None:
            sc = dict(sc, jumpoff_pd=calibrate_to_acl(seg, sc, cfg, sel, spec_name, bk, port[seg][dcol], scen, macro,
                                                      lit, hpi_jump, how, log))
            cfg["segments"][seg] = sc
        calibrated[seg] = sc.get("jumpoff_pd")
        pd_jump, pd_lr, scale = calibration(sc, port[seg][dcol])
        do_nco = cfg["nco_model"].get("enabled", True) and port[seg][cfg["columns"]["nco_rate"]].notna().sum() > 40 \
            and set(sum((d for d in NCO.NCO_DRIVERS.values()), [])) and all(
                d.split("|")[0] in macro.columns for d in NCO.NCO_DRIVERS[book]) if book in NCO.NCO_DRIVERS else False
        nco_res = NCO.fit(port[seg][cfg["columns"]["nco_rate"]], macro, book, cfg["nco_model"]["forward_lead"]) if do_nco else None
        for sname, sdf in scen.items():
            full = splice(macro, sdf)
            X_full, _ = F.build_features(full, lit, cfg, book, sc.get("candidates"))
            idx = sdf.index
            pdp = (_pd_path(sel, spec_name, X_full, idx, how) * scale).clip(upper=0.99)
            hpi_s = full.loc[idx, cidx]
            gr = None
            if growth_tbl is not None:
                g0 = growth_tbl[(growth_tbl.segment == seg) & (growth_tbl.scenario == cfg["scenarios"][sname])]
                gr = g0.sort_values("date")["growth"].reset_index(drop=True)
            p = L.project_segment(seg, sc, bk, pd_jump, pdp, hpi_jump, hpi_s, cfg, pd_lr, growth=gr)
            p["scenario"] = sname
            paths.append(p)
            # PD paths under every candidate model (model-risk range)
            names = list(sel["specs"]) + (["BMA"] if sel.get("bma_specs") else [])
            for nm in names:
                alt = (_pd_path(sel, nm, X_full, idx, how) * scale).clip(upper=0.99)
                pd_paths.append(pd.DataFrame({"segment": seg, "scenario": sname, "model": nm,
                                              "date": idx, "pd_pct": 100 * alt.to_numpy()}))
                if sname in ("baseline", "severe"):
                    pa = L.project_segment(seg, sc, bk, pd_jump, alt, hpi_jump, hpi_s, cfg, pd_lr)
                    g = pa[(pa.q > 0) & (pa.q <= cfg["horizon_quarters"])]
                    model_range.append({"segment": seg, "scenario": sname, "model": nm,
                                        "loss_rate_pct": 100 * g.credit_loss.sum() / g.ead.mean(),
                                        "peak_pd_pct": 100 * g.pd_annual.max()})
            # top-down cross-check
            H = cfg["horizon_quarters"]
            if nco_res is None:
                continue
            nco_p = NCO.project(nco_res, port[seg][cfg["columns"]["nco_rate"]], full, book, idx[:H],
                                cfg["nco_model"]["forward_lead"])
            g = p[(p.q > 0)]
            nco_rows.append({"segment": seg, "scenario": sname,
                             "bottom_up_loss_rate_pct": 100 * g.credit_loss.sum() / g.ead.mean(),
                             "top_down_nco_rate_pct": nco_p.sum() / 4})  # annualized % per quarter -> cumulative
    paths = pd.concat(paths, ignore_index=True)

    # Probability-weighted ECL at jump-off (IFRS 9 5.5.17): Stage 1+2 for comparison with the reported
    # performing allowance, and by stage. Every scenario opens from this probability-weighted allowance,
    # so the first quarter's PCL includes the day-one move to the scenario's own forward-looking ECL.
    w = cfg["ifrs9"]["scenario_weights"]
    j0 = paths[paths.q == 0].pivot_table(index="segment", columns="scenario", values="allowance")
    j0["prob_weighted"] = sum(j0[s] * wt for s, wt in w.items() if s in j0)
    j0["nonlinearity_vs_baseline_pct"] = 100 * (j0["prob_weighted"] / j0["baseline"] - 1)
    stage_cols = ["ecl_stage1", "ecl_stage2", "ecl_stage3", "overlay", "allowance_total"]
    q0 = paths[paths.q == 0]
    pw_stage = sum(q0[q0.scenario == s].set_index("segment")[stage_cols] * wt for s, wt in w.items()
                   if s in set(q0.scenario))
    for (seg, _sname), g in paths.groupby(["segment", "scenario"]):
        i1 = g.index[g.q == 1]
        i0 = g.index[g.q == 0]
        if len(i1):
            day1 = float(paths.loc[i0, "allowance_total"].iloc[0] - pw_stage.loc[seg, "allowance_total"])
            paths.loc[i1, "pcl"] += day1
            paths.loc[i0, "day1_remeasurement"] = day1
    summary = L.summarize(paths, horizons=(9, cfg["horizon_quarters"]))
    disclosures = ifrs9_disclosures(cfg, paths, pw_stage)

    attrib = attribution(cfg, sels, macro, scen, port, snap, lit, hpi_jump, champions)
    nco_df = pd.DataFrame(nco_rows, columns=["segment", "scenario", "bottom_up_loss_rate_pct", "top_down_nco_rate_pct"])
    return {"jumpoff_pd": calibrated, "champions": champions, "sels": sels, "paths": paths, "summary": summary, "model_range": pd.DataFrame(model_range),
            "pd_paths": pd.concat(pd_paths, ignore_index=True), "nco_check": nco_df,
            "pw_ecl": j0, "pw_ecl_by_stage": pw_stage, "ifrs9": disclosures, "attribution": attrib, "macro": macro, "scenarios": scen, "portfolio": port,
            "snapshot": snap}


def ifrs9_disclosures(cfg, paths, pw_stage) -> dict:
    """IFRS 7-style tables from the stage engine.

    rollforward  allowance by stage, jump-off to horizon end, per segment and scenario (IFRS 7.35H)
    gca          gross carrying amount by stage, same span (IFRS 7.35I)
    sensitivity  ECL at jump-off: probability-weighted vs 100% weight on each scenario (forward-looking
                 information sensitivity, as banks disclose), by stage
    staging      ECL at jump-off if every performing loan were in Stage 2 (lifetime), and if none were
    coverage     allowance / gross carrying amount by stage at jump-off (probability-weighted)
    """
    H = cfg["horizon_quarters"]
    w = cfg["ifrs9"]["scenario_weights"]
    rf, gca = [], []
    for (seg, sname), g in paths.groupby(["segment", "scenario"]):
        g = g.sort_values("q")
        a = I9.allowance_rollforward(g, 0, H, opening=pw_stage.loc[seg])
        b = I9.gca_rollforward(g, 0, H)
        for df, out in ((a, rf), (b, gca)):
            d = df.reset_index().rename(columns={"index": "line"})
            d.insert(0, "scenario", sname)
            d.insert(0, "segment", seg)
            out.append(d)
    q0 = paths[paths.q == 0].copy()
    sens = q0.pivot_table(index="segment", columns="scenario", values="allowance_total")
    sens["probability_weighted"] = pw_stage["allowance_total"]
    for s in [c for c in sens.columns if c != "probability_weighted"]:
        sens[f"{s}_vs_pw_pct"] = 100 * (sens[s] / sens["probability_weighted"] - 1)
    q0["all_stage2"] = (q0.gca_s1 + q0.gca_s2) * q0.cov_s2 + q0.ecl_stage3
    q0["all_stage1"] = (q0.gca_s1 + q0.gca_s2) * q0.cov_s1 + q0.ecl_stage3
    stg = pd.DataFrame({c: sum(q0[q0.scenario == s].set_index("segment")[c] * wt for s, wt in w.items())
                        for c in ["all_stage1", "all_stage2", "allowance_total"]})
    stg["reported_staging"] = stg.pop("allowance_total")
    gpw = sum(q0[q0.scenario == s].set_index("segment")[["gca_s1", "gca_s2", "gca_s3"]] * wt for s, wt in w.items())
    cov = pd.DataFrame({f"Stage {k}": 100 * pw_stage[f"ecl_stage{k}"] / gpw[f"gca_s{k}"].replace(0, np.nan)
                        for k in (1, 2, 3)})
    cov["Total"] = 100 * pw_stage[["ecl_stage1", "ecl_stage2", "ecl_stage3"]].sum(axis=1) / gpw.sum(axis=1)
    return {"rollforward": pd.concat(rf, ignore_index=True), "gca": pd.concat(gca, ignore_index=True),
            "sensitivity": sens, "staging": stg, "coverage": cov, "pw_by_stage": pw_stage, "gca_jumpoff": gpw}


def _driver_mnemonics(sel, spec_name) -> list[str]:
    if spec_name == "BMA":
        mns = {d.split("|")[0] for sp, w in sel["bma_specs"] if w > 0.01 for d in sp.drivers}
    else:
        mns = {d.split("|")[0] for d in sel["specs"][spec_name].drivers}
    return sorted(mns)


def attribution(cfg, sels, macro, scen, port, snap, lit, hpi_jump, champions, stress="severe", base="baseline"):
    """Counterfactual substitution: severe minus baseline cumulative loss, split by cause.

    Step 1 (PD vs collateral): Shapley average of the two orderings.
    Step 2 (macro drivers of the PD leg): hold one driver at its baseline path, re-project.
    """
    how = cfg["pd_model"]["transform"]
    H = cfg["horizon_quarters"]
    dcol = cfg["columns"]["default_rate"]
    rows = []
    fs, fb = splice(macro, scen[stress]), splice(macro, scen[base])
    idx = scen[stress].index
    for seg, sc in cfg["segments"].items():
        book = _book(sc)
        sel = sels[seg]
        spec_name = champions[seg]
        bk = snap[snap["segment"] == seg]
        pd_jump, pd_lr, scale = calibration(sc, port[seg][dcol])
        cand = sc.get("candidates")
        cidx = cfg.get("collateral_index", "HPI_ON")
        Xs, _ = F.build_features(fs, lit, cfg, book, cand)
        Xb, _ = F.build_features(fb, lit, cfg, book, cand)
        pds = (_pd_path(sel, spec_name, Xs, idx, how) * scale).clip(upper=0.99)
        pdb = (_pd_path(sel, spec_name, Xb, idx, how) * scale).clip(upper=0.99)
        hs, hb = fs.loc[idx, cidx], fb.loc[idx, cidx]

        def cum(pdp, hp, seg=seg, sc=sc, bk=bk, pd_jump=pd_jump, pd_lr=pd_lr):
            p = L.project_segment(seg, sc, bk, pd_jump, pdp, hpi_jump, hp, cfg, pd_lr)
            g = p[(p.q > 0) & (p.q <= H)]
            return g.credit_loss.sum()

        L_bb, L_ss, L_sb, L_bs = cum(pdb, hb), cum(pds, hs), cum(pds, hb), cum(pdb, hs)
        pd_eff = 0.5 * ((L_sb - L_bb) + (L_ss - L_bs))
        col_eff = 0.5 * ((L_bs - L_bb) + (L_ss - L_sb))
        rows.append({"segment": seg, "component": "Baseline loss", "amount": L_bb})
        rows.append({"segment": seg, "component": "Higher defaults (PD)", "amount": pd_eff})
        rows.append({"segment": seg, "component": "Weaker collateral (LGD)", "amount": col_eff})
        # driver split of the PD leg
        mns = _driver_mnemonics(sel, spec_name)
        single = {}
        for mn in mns:
            mixed = fs.copy()
            mixed.loc[idx, mn] = fb.loc[idx, mn]
            if mn in ("RMORT5", "RNEUTRAL"):
                mixed["MORTGAP"] = mixed["RMORT5"] - mixed["RNEUTRAL"]
            for d_mn, (a, b) in cfg.get("derived_series", {}).items():  # keep constructed spreads consistent
                mixed[d_mn] = mixed[a] - mixed[b]
            Xm, _ = F.build_features(mixed, lit, cfg, book, cand)
            single[mn] = L_ss - cum((_pd_path(sel, spec_name, Xm, idx, how) * scale).clip(upper=0.99), hs)
        tot = sum(single.values())
        for mn, v in single.items():  # rescale one-at-a-time effects to the Shapley PD total
            rows.append({"segment": seg, "component": f"  of which {mn}", "amount": v * pd_eff / tot if tot else 0.0})
        rows.append({"segment": seg, "component": "Severe loss", "amount": L_ss})
    return pd.DataFrame(rows)
