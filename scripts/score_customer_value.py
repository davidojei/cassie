"""
scripts/score_customer_value.py

Phase 11: Customer value + revenue at risk.

revenue_at_risk = lifetime_revenue (customer_value) x predicted_churn_probability

Scores every active customer as of MAX_VALID_DATE (2018-08-31 -- the
same constant used in build_features.py) using the chosen baseline
logistic regression model (Phase 9). Reuses build_snapshot() and the
same base-table loaders unchanged, so no new leakage surface is
introduced.

Usage:
    python scripts/score_customer_value.py

ASSUMPTION TO VERIFY: raw.customer_costs and raw.customer_segments
table names/columns -- I inferred these from the naming pattern of
the other synthetic tables (raw.customer_support, raw.marketing_campaigns,
raw.retention_interventions) referenced in build_features.py, but I
haven't seen docs/synthetic_data_generation.md's actual schema for
these two. Adjust the SELECT in load_costs_and_segments() if the real
column/table names differ.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent))
from build_features import (  # noqa: E402
    MAX_VALID_DATE,
    build_snapshot,
    load_base_tables,
    load_synthetic_tables,
)
from train_baseline_model import (  # noqa: E402
    CATEGORICAL_FEATURES,
    INDICATOR_SPECS,
    NUMERIC_FEATURES,
    add_missingness_indicators,
)

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://cassie:cassie@localhost:5432/cassie")

MODEL_PATH = Path(__file__).parent.parent / "artifacts" / "models" / "baseline_logistic_regression.joblib"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "customer_revenue_at_risk.csv"

# Last training snapshot, already sitting in analytics.customer_snapshot_features
# -- used as the comparison baseline for the sanity check below.
TRAINING_SNAPSHOT_FOR_COMPARISON = pd.Timestamp("2018-03-01")

MODEL_CAVEAT = (
    "Estimated based on a model with limited predictive power "
    "(ROC-AUC ~0.55-0.65) - treat as a rough prioritization signal "
    "for where to focus retention effort, not a precise forecast."
)


def load_costs_and_segments(engine):
    """ASSUMPTION: table names raw.customer_costs / raw.customer_segments
    -- verify against docs/synthetic_data_generation.md and fix if wrong."""
    costs = pd.read_sql("SELECT * FROM raw.customer_costs", engine)
    segments = pd.read_sql("SELECT * FROM raw.customer_segments", engine)
    return costs, segments


def sanity_check_against_training_snapshot(engine, scoring_df):
    """
    Compare the scoring snapshot's feature summary stats (nulls, min/max)
    against the last real training snapshot already stored in
    analytics.customer_snapshot_features. This project has hit the same
    fan-out/aggregation bug three times and one leakage bug that first
    showed up as an unexpected stat shift -- so before trusting
    probabilities from a brand-new end-of-data snapshot, diff the two.

    Prints a report. Does not fail the run automatically -- eyeball it.
    """
    print("\n--- Sanity check: scoring snapshot vs. last training snapshot (2018-03-01) ---")
    training_df = pd.read_sql(
        "SELECT * FROM analytics.customer_snapshot_features WHERE snapshot_date = %(d)s",
        engine, params={"d": TRAINING_SNAPSHOT_FOR_COMPARISON},
    )

    if training_df.empty:
        print("  Could not load the training snapshot for comparison -- skipping this check.")
        return None

    feature_cols = [c for c in NUMERIC_FEATURES if c in scoring_df.columns and c in training_df.columns]

    rows = []
    for col in feature_cols:
        rows.append({
            "feature": col,
            "scoring_null_pct": round(scoring_df[col].isna().mean() * 100, 2),
            "training_null_pct": round(training_df[col].isna().mean() * 100, 2),
            "scoring_min": scoring_df[col].min(),
            "training_min": training_df[col].min(),
            "scoring_max": scoring_df[col].max(),
            "training_max": training_df[col].max(),
        })
    report = pd.DataFrame(rows)
    report["null_pct_jump"] = (report["scoring_null_pct"] - report["training_null_pct"]).abs()
    flagged = report[report["null_pct_jump"] > 5.0]

    print(report.to_string(index=False))
    if len(flagged):
        print(f"\n  {len(flagged)} feature(s) show a >5pt null-rate shift vs. training -- inspect before trusting scores:")
        print(flagged[["feature", "scoring_null_pct", "training_null_pct"]].to_string(index=False))
    else:
        print("\n  No feature shows a >5pt null-rate shift vs. training.")
    print("--- end sanity check ---\n")
    return report


def check_recent_customer_skew(scored_df, purchase_events):
    """
    Cold-start hypothesis is flagged UNRESOLVED in this project. Check
    whether predicted_churn_probability looks suspiciously different for
    recently-acquired customers at this specific edge-date snapshot --
    exactly the shape the synthetic-data span_days leakage bug took
    before it was found (a distribution artifact, not an obvious crash).

    days_since_first_purchase isn't a stored feature column, so it's
    derived here directly from purchase_events (same source build_snapshot
    itself uses).
    """
    first_purchase = (
        purchase_events.groupby("customer_unique_id")["order_purchase_timestamp"]
        .min()
        .rename("first_purchase")
    )
    df = scored_df.merge(first_purchase, on="customer_unique_id", how="left")
    df["days_since_first_purchase"] = (MAX_VALID_DATE - df["first_purchase"]).dt.days

    df["recency_bucket"] = pd.qcut(df["days_since_first_purchase"], q=4, duplicates="drop")
    summary = df.groupby("recency_bucket", observed=True)["predicted_churn_probability"].agg(
        ["mean", "median", "std", "count"]
    )
    print("\n--- Predicted churn probability by customer tenure (newest -> oldest quartile) ---")
    print(summary.to_string())
    print(
        "Eyeball this: a probability distribution that's flat or "
        "implausibly extreme in the newest-customer bucket relative to "
        "the others is worth investigating before picking risk_tier cutoffs.\n"
    )


def main():
    engine = create_engine(DATABASE_URL)

    print("Loading base tables...")
    purchase_events, order_summary, item_summary, customers = load_base_tables(engine)
    print("Loading synthetic tables (support, campaigns, interventions)...")
    support, campaigns, interventions = load_synthetic_tables(engine)

    print(f"\nBuilding scoring snapshot at {MAX_VALID_DATE.date()} ...")
    scoring_df = build_snapshot(
        MAX_VALID_DATE, purchase_events, order_summary, item_summary, customers,
        support, campaigns, interventions,
    )
    print(f"  {len(scoring_df)} customers active as of this snapshot")

    # The scoring snapshot's churned / days_to_next_purchase columns are
    # meaningless at the true end of the data -- there is no future
    # purchase data to check against, so every customer trivially shows
    # churned=1. Drop both so they can never be accidentally reported.
    scoring_df = scoring_df.drop(columns=["churned", "days_to_next_purchase"], errors="ignore")

    sanity_check_against_training_snapshot(engine, scoring_df)

    print(f"Loading model from {MODEL_PATH} ...")
    pipeline = joblib.load(MODEL_PATH)

    scoring_df = add_missingness_indicators(scoring_df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + list(INDICATOR_SPECS.keys())
    scoring_df["predicted_churn_probability"] = pipeline.predict_proba(scoring_df[feature_cols])[:, 1]

    check_recent_customer_skew(scoring_df, purchase_events)

    costs, segments = load_costs_and_segments(engine)

    result = scoring_df[["customer_unique_id", "lifetime_revenue", "predicted_churn_probability"]].copy()
    result = result.rename(columns={"lifetime_revenue": "customer_value"})
    result["revenue_at_risk"] = result["customer_value"] * result["predicted_churn_probability"]

    result = result.merge(segments, on="customer_unique_id", how="left")
    result = result.merge(costs, on="customer_unique_id", how="left")
    result["caveat"] = MODEL_CAVEAT

    # --- Smoke test -- run before writing output, fail loudly if wrong ---
    assert result["revenue_at_risk"].isna().sum() == 0, \
        "Null values in revenue_at_risk -- investigate before shipping this output."
    assert result["predicted_churn_probability"].between(0, 1).all(), \
        "predicted_churn_probability out of [0,1] range."
    assert result["customer_unique_id"].is_unique, \
        "Duplicate customer_unique_id rows -- check the joins to costs/segments for fan-out."

    print("\n--- revenue_at_risk distribution ---")
    print(result["revenue_at_risk"].describe())
    print("\n--- predicted_churn_probability distribution ---")
    print(result["predicted_churn_probability"].describe())

    # risk_tier cutoffs deliberately NOT hardcoded -- see the printed
    # distribution above and pick sensible cutoffs once you've looked at
    # real numbers. Placeholder tertiles shown so the column isn't left
    # hanging open; replace before this feeds any report or dashboard.
    result["risk_tier"] = pd.qcut(
        result["predicted_churn_probability"], q=[0, 0.5, 0.8, 1.0], labels=["low", "medium", "high"],
    )
    print("\n--- risk_tier counts (PLACEHOLDER cutoffs -- revisit) ---")
    print(result["risk_tier"].value_counts())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(result)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
