"""Part 1 of the GenAI app: the CONTEXT.

A language model knows nothing about your model run. Before each question we
give it the facts it needs, as plain text. This is called "grounding" (or, when
the facts are searched for first, "retrieval-augmented generation", RAG).

Here the facts are a few small tables from the results workbook, in C$ millions.
We keep them short because a local model reads about 40 words per second on this
Mac: every extra table makes every answer slower.
"""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "eqb_public" / "credit_loss_results.xlsx"
CONFIG = ROOT / "config" / "eqb_public.yaml"

SEGMENTS = {
    "mortgage_uninsured": "Uninsured mortgages",
    "mortgage_insured": "Insured mortgages",
    "credit_cards": "Credit cards",
    "commercial": "Commercial + equipment",
}


def _table(df: pd.DataFrame, cols: list[str], scale: float = 1e6, digits: int = 1) -> str:
    """Turn rows of numbers into a compact Markdown table the model can read.

    Amount tables get a TOTAL row computed here, in code: small language models
    are bad at arithmetic, so we never ask them to add numbers up themselves.
    """
    df = df.copy()
    df["segment"] = df["segment"].map(SEGMENTS).fillna(df["segment"])
    if scale != 1:  # amounts, not percentages: add a total row
        total = {"segment": "TOTAL, all segments", **{c: df[c].sum() for c in cols}}
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    head = "| Segment | " + " | ".join(cols) + " |"
    rule = "|---" * (len(cols) + 1) + "|"
    rows = [
        "| " + r["segment"] + " | "
        + " | ".join("n/a" if pd.isna(r[c]) else f"{r[c] / scale:,.{digits}f}" for c in cols) + " |"
        for _, r in df.iterrows()
    ]
    return "\n".join([head, rule, *rows])


def build_context() -> str:
    """Read the shipped results and return them as one block of text."""
    weights = yaml.safe_load(CONFIG.read_text())["ifrs9"]["scenario_weights"]
    x = pd.ExcelFile(RESULTS)

    acl = x.parse("ifrs9_pw_ecl_by_stage")
    acl = acl.rename(columns={"ecl_stage1": "Stage 1", "ecl_stage2": "Stage 2",
                              "ecl_stage3": "Stage 3", "allowance_total": "Total"})
    acl_total = acl["Total"].sum() / 1e6

    sens = x.parse("ifrs9_scenario_sensitivity")
    summ = x.parse("summary")
    losses = (summ[summ["horizon_q"] == 13]
              .pivot(index="segment", columns="scenario", values="cum_loss").reset_index())
    cov = x.parse("ifrs9_coverage_pct")

    weight_text = ", ".join(f"{k} {v:.0%}" for k, v in weights.items())
    return f"""\
IFRS 9 credit-loss model for EQB Inc.'s loan book at July 31, 2026, built only from public data.
Amounts in C$ millions unless marked %. Scenario weights: {weight_text}.

ALLOWANCE (ACL) TODAY, probability-weighted, by stage. Total C${acl_total:,.0f}m.
{_table(acl, ["Stage 1", "Stage 2", "Stage 3", "Total"])}

ALLOWANCE IF 100% WEIGHT WERE ON ONE SCENARIO
{_table(sens, ["upside", "baseline", "downside", "severe", "probability_weighted"])}

CUMULATIVE CREDIT LOSSES OVER 13 QUARTERS, by scenario
{_table(losses, ["upside", "baseline", "downside", "severe"])}

COVERAGE RATIO (allowance / gross loans, %)
{_table(cov, ["Stage 1", "Stage 2", "Stage 3", "Total"], scale=1, digits=2)}

NOTES
- Severe scenario: unemployment +4.5 pp to about 11%, house prices -30%, policy rate to 0.5%.
- Credit cards are calibrated so the weighted Stage 1+2 allowance equals EQB's reported C$240m.
- Loss history uses proxies (Ontario mortgage arrears, US charge-off rates); scenarios are author-built.
- Results illustrate the method and are not a forecast of EQB's results.
"""


if __name__ == "__main__":
    # Run `python genai/context.py` to see exactly what the model is given.
    print(build_context())
