"""
Phase 10 — SHAP explainability for the baseline logistic regression
model (the one actually chosen — see docs/model_comparison_conclusion.md).

The whole point of this file is the translation layer: SHAP itself
produces technical feature names and signed numeric contributions.
Nobody on a retention team should ever see "days_since_last_purchase:
165, SHAP=+0.42" -- they should see "Hasn't purchased in 165 days
(typical gap for returning customers is ~40 days)". FEATURE_LABELS and
explain_customer() below exist entirely to make that translation.

Usage:
    python scripts/explain_model.py
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent))
from train_baseline_model import (  # noqa: E402
    DATABASE_URL, NUMERIC_FEATURES, CATEGORICAL_FEATURES, INDICATOR_SPECS,
    LABEL_COLUMN, load_data,
)

# ============================================================
# Plain-language translation layer
# ============================================================

# Features that are real, safe signal (kept in the model and in global
# importance) but suppressed from the CUSTOMER-FACING explanation
# specifically: they tell the retention team something it already knew
# rather than something new and actionable. has_been_targeted_for_retention
# is the clear case -- "the top reason they're at risk is that we already
# flagged them as at risk" isn't useful framing for a human reader, even
# though the model is using the signal legitimately.
EXCLUDE_FROM_CUSTOMER_EXPLANATION = {"has_been_targeted_for_retention"}

FEATURE_LABELS = {
    "days_since_last_purchase": "Days since their last purchase",
    "orders_last_30_days": "Orders placed in the last 30 days",
    "orders_last_90_days": "Orders placed in the last 90 days",
    "orders_last_180_days": "Orders placed in the last 180 days",
    "lifetime_orders": "Total number of purchases ever made",
    "lifetime_revenue": "Total amount spent, lifetime",
    "average_order_value": "Average amount spent per order",
    "max_order_value": "Largest single order value",
    "purchase_frequency": "How often they typically purchase",
    "purchase_gap_mean": "Average days between their purchases",
    "purchase_gap_std": "How consistent their purchase timing is",
    "average_delivery_delay": "Average delivery delay vs. the promised date",
    "late_delivery_rate": "Share of their orders that arrived late",
    "average_delivery_days": "Average time from order to delivery",
    "average_review_score": "Average rating they've given (1-5 stars)",
    "negative_review_rate": "Share of their reviews that were negative",
    "installment_frequency": "Average number of payment installments used",
    "payment_value": "Total amount paid across all orders",
    "recent_90d_revenue": "Amount spent in the last 90 days",
    "previous_90d_revenue": "Amount spent in the 90 days before that",
    "purchase_frequency_trend_90d": "Change in how often they buy (recent vs. prior)",
    "review_trend_90d": "Change in their review scores (recent vs. earlier)",
    "recent_delivery_delay": "Average delivery delay in the last 90 days",
    "unique_categories": "Number of different product categories bought",
    "category_concentration": "How focused their purchases are on one category",
    "unique_sellers": "Number of different sellers bought from",
    "seller_concentration": "How focused their purchases are on one seller",
    "n_support_tickets": "Number of support tickets filed",
    "n_tickets_last_90_days": "Support tickets filed in the last 90 days",
    "pct_tickets_resolved": "Share of their support tickets that got resolved",
    "pct_tickets_escalated": "Share of their support tickets that got escalated",
    "avg_resolution_hours": "Average time to resolve their support tickets",
    "avg_ticket_satisfaction": "Average satisfaction rating on their support tickets",
    "n_campaigns_received": "Number of marketing messages received",
    "pct_campaigns_opened": "Share of marketing messages they opened",
    "pct_campaigns_clicked": "Share of marketing messages they clicked",
    "pct_campaigns_converted": "Share of marketing messages that led to a purchase",
    "days_since_last_campaign": "Days since their last marketing message",
    "has_been_targeted_for_retention": "Has previously received a retention offer",
    "has_repeat_purchase": "Has purchased more than once",
    "has_review": "Has left at least one review",
    "has_delivered_order": "Has had at least one order delivered",
    "has_recent_delivered_order": "Had a delivery in the last 90 days",
    "has_support_history": "Has contacted support before",
    "has_campaign_history": "Has received marketing messages before",
}


def plain_label(technical_name: str) -> str:
    """Translates a (possibly one-hot-expanded) transformed feature name
    into a stakeholder-readable label. Falls back to a readable guess
    for anything not explicitly mapped, rather than ever showing a raw
    sklearn/SHAP column name."""
    name = technical_name.split("__", 1)[-1]  # strip the ColumnTransformer prefix

    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]

    if name.startswith("preferred_payment_method_"):
        method = name.replace("preferred_payment_method_", "").replace("_", " ")
        return f"Usually pays by {method}"
    if name.startswith("customer_state_"):
        state = name.replace("customer_state_", "")
        return f"Located in {state}"

    return name.replace("_", " ").capitalize()  # last-resort readable fallback


def describe_value(feature_name: str, value, reference_median=None) -> str:
    """Turns a raw feature value into a natural sentence fragment,
    comparing against a reference (typically the median for customers
    who DID return) when available -- this is what makes an explanation
    like 'usual gap for returners is ~40 days' possible."""
    label = plain_label(feature_name)
    name = feature_name.split("__", 1)[-1]

    if name.startswith("has_") or name in ("has_been_targeted_for_retention",):
        return f"{label}: {'Yes' if value else 'No'}"

    if pd.isna(value):
        return f"{label}: not available"

    if isinstance(value, (int, float, np.integer, np.floating)):
        val_str = f"{value:,.1f}" if abs(value) < 1000 else f"{value:,.0f}"
        if reference_median is not None and not pd.isna(reference_median):
            return f"{label}: {val_str} (typical for returning customers: {reference_median:,.1f})"
        return f"{label}: {val_str}"

    return f"{label}: {value}"


# ============================================================
# SHAP computation
# ============================================================

def compute_shap(pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    background_transformed = preprocessor.transform(X_background)
    explain_transformed = preprocessor.transform(X_explain)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(model, background_transformed)
    shap_values = explainer.shap_values(explain_transformed)

    return shap_values, feature_names, explain_transformed


def global_importance(shap_values, feature_names, top_n=15) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df["plain_label"] = df["feature"].apply(plain_label)
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n)
    return df[["plain_label", "mean_abs_shap", "feature"]].reset_index(drop=True)


def explain_customer(customer_idx, X_explain_raw, shap_values, feature_names,
                      reference_medians, top_k=5) -> str:
    """The main deliverable: a plain-language paragraph for ONE customer,
    matching the format stakeholders actually asked for."""
    row_shap = shap_values[customer_idx]
    raw_row = X_explain_raw.iloc[customer_idx]

    ranked = sorted(zip(feature_names, row_shap), key=lambda x: abs(x[1]), reverse=True)
    contributions = [
        (name, val) for name, val in ranked
        if name.split("__", 1)[-1] not in EXCLUDE_FROM_CUSTOMER_EXPLANATION
    ][:top_k]

    lines = []
    for tech_name, shap_val in contributions:
        base_name = tech_name.split("__", 1)[-1]
        raw_value = raw_row.get(base_name, None)
        ref = reference_medians.get(base_name, None)
        direction = "increases" if shap_val > 0 else "decreases"
        desc = describe_value(tech_name, raw_value, ref)
        lines.append(f"  - {desc}  [{direction} churn risk]")

    return "\n".join(lines)


def main():
    engine = create_engine(DATABASE_URL)
    print("Loading data and the saved baseline model...")
    df = load_data(engine)
    pipeline = joblib.load(Path(__file__).parent.parent / "artifacts" / "models" / "baseline_logistic_regression.joblib")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + list(INDICATOR_SPECS.keys())
    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]

    # Background sample for the explainer (SHAP needs a reference
    # distribution, not the whole training set, for tractability)
    background = train[feature_cols].sample(n=min(500, len(train)), random_state=42)

    # Reference medians computed from customers who DID return (churned=0)
    # -- this is what makes "typical for returning customers: X" possible
    returned = train[train[LABEL_COLUMN] == 0]
    reference_medians = returned[NUMERIC_FEATURES].median().to_dict()

    # Explain a sample of validation customers
    sample = validation.sample(n=min(200, len(validation)), random_state=42).reset_index(drop=True)
    X_explain = sample[feature_cols]

    print("Computing SHAP values...")
    shap_values, feature_names, _ = compute_shap(pipeline, background, X_explain)

    print("\n=== Global feature importance (plain language) ===")
    importance = global_importance(shap_values, feature_names)
    for _, row in importance.iterrows():
        print(f"  {row['plain_label']:<55} avg impact: {row['mean_abs_shap']:.4f}")

    print("\n=== Example customer explanations ===")
    for i in range(min(3, len(sample))):
        cust_id = sample.loc[i, "customer_unique_id"]
        pred_proba = pipeline.predict_proba(X_explain.iloc[[i]])[0, 1]
        print(f"\nCustomer {cust_id} (predicted churn risk: {pred_proba:.1%}):")
        print(explain_customer(i, X_explain, shap_values, feature_names, reference_medians))

    reports_dir = Path(__file__).parent.parent / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(reports_dir / "shap_global_importance.csv", index=False)

    explanations = []
    for i in range(len(sample)):
        explanations.append({
            "customer_unique_id": sample.loc[i, "customer_unique_id"],
            "predicted_churn_risk": float(pipeline.predict_proba(X_explain.iloc[[i]])[0, 1]),
            "top_factors": explain_customer(i, X_explain, shap_values, feature_names, reference_medians),
        })
    with open(reports_dir / "shap_customer_explanations.json", "w") as f:
        json.dump(explanations, f, indent=2, default=str)

    print(f"\nGlobal importance saved to {reports_dir / 'shap_global_importance.csv'}")
    print(f"Customer explanations saved to {reports_dir / 'shap_customer_explanations.json'}")


if __name__ == "__main__":
    main()
