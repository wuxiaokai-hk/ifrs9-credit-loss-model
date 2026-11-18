# Model documentation: IFRS 9 Credit-Loss Scenario Model

*Structured along the model lifecycle in OSFI Guideline E-23 (Model Risk Management) and the Federal Reserve's SR 11-7.
Version 1.0.0, November 2026. Author: Xiaokai Wu. Independent project on public data.*

## 1. Purpose and intended use

| | |
|---|---|
| **Purpose** | Project credit losses, the IFRS 9 allowance for credit losses (ACL) and provisions (PCL) for a loan book under macroeconomic scenarios |
| **Intended uses** | Scenario analysis and stress testing; IFRS 9 forward-looking ECL with probability-weighted scenarios; sensitivity analysis (scenario weights, staging, overlays); challenger to production ECL models |
| **Not intended for** | Regulatory capital; loan-level pricing or approval; reported financial statements without calibration to the institution's own loss data |
| **Users** | Credit-risk and stress-testing analysts, model validators, finance (ECL) teams |
| **Outputs** | Per segment, scenario and quarter: PD, LGD, EAD, credit loss, gross loans and ECL by stage, ACL, PCL, write-offs; IFRS 7 roll-forwards; attribution; model-risk range |

## 2. Scope and portfolio

The shipped public-data application covers EQB Inc.'s book at July 31, 2026 (C$49bn), in four segments:

| Segment | Balance | PD proxy | LGD approach |
|---|---:|---|---|
| Uninsured mortgages (incl. decumulation) | C$24.5bn | Ontario 90+ day arrears (CBA) | Indexed LTV at liquidation |
| Insured mortgages (single-family, multi-unit) | C$9.0bn | Same | Insurer coverage; residual cost |
| Credit cards (PC Financial) | C$4.5bn drawn + C$9.0bn undrawn | US card charge-off rate | Fixed 85%; CCF 25% |
| Commercial and equipment | C$11.1bn | US business-loan charge-off rate | Frye-Jacobs (ELGD 15%) |

## 3. Methodology

**3.1 PD satellites.**
- **Form:** logit(default proxy) = α + ρ·logit(lag) + Σ βₖ·xₖ, with up to three macro drivers.
- **Candidate drivers:** levels, four-quarter changes and year-on-year growth at lags 0–4.
- **Selection:**
  1. a univariate screen controlling for the lag;
  2. correlation clustering (|r| > 0.85);
  3. a constrained exhaustive search that requires literature sign priors and VIF ≤ 5, ranked by the RMSE of a dynamic
     simulation (no lagged actuals).
- **Challengers:**
  - Bayesian model averaging over admissible models;
  - re-estimated literature models: BoC SAP 2026-12 arrears model, BoC TDSA aggregate PCL ARDL, ECB/EBA core satellite,
    Fed CLASS;
  - an AR(1) benchmark.

**3.2 Bank PD.**
- **Fixed jump-off PD:** the bank's PD moves in proportion to the projected proxy, PD_bank,t = PD_bank,0 × proxy_t / proxy_0
  (EBA constrained bottom-up).
- **Solved jump-off PD:** where `target_acl` is set (cards), PD_bank,0 is solved so the model's probability-weighted
  Stage 1+2 allowance equals the reported allowance.

**3.3 LGD.**
- **Uninsured mortgages:** LGD = workout cost + max(1 − (1 − d − c)/LTV_sale, 0). The LTV at sale uses the scenario house-price
  path four quarters after default (perfect foresight, EBA). Defaults are weighted toward high-LTV buckets.
- **Insured mortgages:** the lender keeps only what the insurer does not reimburse.
- **Commercial:** Frye-Jacobs, so LGD rises with the default rate.
- **Cards:** fixed.

**3.4 IFRS 9.** Quarterly three-stage engine (`creditloss/modelling/ifrs9.py`):

| Element | Treatment | Setting (public application) |
|---|---|---|
| SICR | PD > 3× origination PD; lognormal borrower PD drift calibrated to the reported Stage 2 share; 30 dpd floor optional | `sicr_multiple: 3` |
| Stage 2 PD | 3× Stage 1, book average preserved | `stage2_pd_multiple: 3` |
| Cures | Stage 2 → 1 at 10% a quarter; Stage 3 cures to Stage 2 | `stage2_cure_q: 0.10` |
| Stage 3 | PD = 1; resolved after the workout period, written off or cured | 6 quarters (mortgages, commercial), 2 (cards) |
| Lifetime | Remaining contractual term; behavioural for revolving | 2.5–3 years assumed; cards 3 years |
| Discounting | Effective interest rate | Segment contract rate |
| Forward-looking period | Scenario PDs, then linear reversion to the long-run PD | 12 quarters + 8 |
| Scenario weights | Upside 10%, baseline 40%, downside 30%, severe 20% | Config |
| PCL | Change in ACL + net write-offs; every scenario opens from the probability-weighted ACL | — |

**3.5 Exposure.** Static balance sheet: repayments and resolved defaults are replaced by new Stage 1 loans. Run-off and growth
are options. Amortizing exposure follows the annuity schedule; revolving exposure is drawn + CCF × undrawn.

## 4. Data

- **Public sources:** listed in `data/eqb_public/SOURCES.md`. `creditloss build-data` reproduces the inputs byte for byte.
- **Treatment of macro series:**
  - unemployment is seasonally adjusted;
  - unemployment and GDP are bridged through 2020Q2–2021Q3, when support programmes broke their link with losses;
  - series ending before the jump-off are carried forward and flagged.

**Key assumptions (not disclosed by EQB):**

| Assumption | Value |
|---|---|
| LTV distribution around the disclosed 69% average | Buckets at 0.45, 0.60, 0.71, 0.78 (10/20/35/35%) |
| Remaining amortization | 23 years (mortgages) |
| Remaining contractual term | 2.5–3 years |
| Prepayment | 12% a year (uninsured mortgages) |
| Card limits | Twice drawn balances |
| Workout period | 6 quarters (mortgages, commercial) |

## 5. Performance and validation

**5.1 Out-of-sample backtest.**
- **Design:** expanding window, origins every four quarters from 2006Q4, 1–12 quarters ahead. The whole selection is re-run at
  every origin.
- **Diebold-Mariano tests:** against AR(1), with the Harvey-Leybourne-Newbold correction.
- **Stress capture:** cumulative forecast over 2008–10 from the last origin before it.

| Segment | Champion | OOS RMSE, pp | AR(1) RMSE, pp | GFC capture | Rule met |
|---|---|---:|---:|---:|---|
| Mortgages (both) | BoC TDSA-style ARDL | 0.066 | 0.070 | 124% | Yes |
| Credit cards | BMA | 0.98 | 1.91 | 88% | Yes |
| Commercial | BMA | 1.48 | 0.50 | 336% | **No** |

**Champion rule:** correct signs in the full-sample fit; among models with GFC capture of 80–200%, the lowest OOS RMSE.
Otherwise, the model with capture closest to 100%.

**5.2 Simulated book (known data-generating process).**
- **Accuracy and stress capture:** every champion beats AR(1) by 30–60% in RMSE and captures 84–148% of the GFC.
- **Uninsured mortgages:** the data-driven search finds the true unemployment, house-price and mortgage-rate-gap
  channels.
- **Commercial real estate:** the champion uses two of the three true drivers at their true lags.

**5.3 Reconciliation to reported figures (July 31, 2026).**

| | Model | Reported |
|---|---:|---:|
| Credit cards, Stage 1+2 ACL | C$239.8m (calibrated) | C$239.8m |
| Mortgages and commercial, Stage 1+2 ACL | about 43% below reported | C$160.0m |
| Stage 3 ACL (all segments) | C$68m | about C$86m (implied) |

The Stage 1+2 gap comes from discounting, short contractual lifetimes and the absence of an overlay. It is left visible
rather than calibrated away.

**5.4 Automated tests (25, run in CI).**
- **Stage engine:** conservation of gross loans; PCL = ΔACL + write-offs; both roll-forwards close; transfers net to zero;
  stress raises Stage 2 and the ACL; the 30 dpd floor binds; discounting lowers ECL; lifetime coverage exceeds 12-month
  coverage.
- **LGD and selection:** LGD monotone in house prices; Frye-Jacobs anchoring; the Diebold-Mariano test detects a better model;
  the champion rule respects the band and exclusions.
- **End to end:** scenario ordering; the probability-weighted ACL lies within the scenario range; the card calibration holds;
  every champion is correctly signed.
- **Reproducibility:** configuration validation; simulated data reproduces exactly.

## 6. Limitations and compensating controls

| Limitation | Effect | Control |
|---|---|---|
| Proxy loss histories (big-bank Ontario arrears; US charge-offs) | Dynamics may differ from EQB's alternative-lending book | Bank PD level anchored to the bank's jump-off PD or to its reported ACL; model-risk range reported |
| One Canadian mortgage stress episode since 1996 | Low power of OOS tests | Literature challengers; stress-capture screen; BMA |
| Commercial champion fails the stress band | Conservative severe losses (over-predicts the GFC) and weaker calm-period accuracy | Flagged; full range across models shown |
| Assumed LTV distribution, terms, workout | Allowance level sensitivity | All in config, editable in the app; staging and scenario-weight sensitivity disclosed |
| Author-built scenarios | Not comparable with supervisory scenarios | Severe sized against the BoE 2025 test and Ontario's early-1990s downturn |
| Short-rate level not used as a mortgage driver | Misses the 2023–26 renewal payment shock | Handle as an overlay or a separate payment-shock model |

## 7. Monitoring and change control

- **When to rerun:** re-run `creditloss run --config … --backtest` when new quarterly data arrive. The champion changes only
  when the out-of-sample record changes.
- **Change control:** every setting is in version-controlled YAML. Runs write the configuration, champions and
  out-of-sample record next to the results.
- **Thresholds that trigger review:**
  - OOS RMSE above AR(1) for two consecutive updates;
  - GFC capture outside 80–200%;
  - a sign flip in a champion's driver;
  - modelled ACL moving more than 25% from reported without a change in scenarios.

| Version | Date | Change |
|---|---|---|
| 1.0.0 | Nov 2026 | IFRS 9 three-stage engine, IFRS 7 tables, package/CLI/tests/CI, app |
