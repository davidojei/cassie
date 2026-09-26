"""
Phase 8 — baseline model: Logistic Regression.

Trains on analytics.customer_snapshot_features (built in Phase 7),
using the train/validation/test split already assigned there (temporal
+ customer-leakage safe, per docs/feature_engineering.md).

See docs/baseline_model.md for the full reasoning behind every
decision here: which columns were excluded as duplicates/leakage, the
class-imbalance framing, imputation strategy, and encoding choices.

Usage:
    python scripts/train_baseline_model.py
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://cassie:cassie@localhost:5432/cassie")

# See docs/baseline_model.md "Feature audit" table for why each
# excluded column is a duplicate/derivable/leaky, not just omitted.
NUMERIC_FEATURES = [
    "days_since_last_purchase", "orders_last_30_days", "orders_last_90_days",
    "orders_last_180_days", "lifetime_orders", "lifetime_revenue",
    "average_order_value", "max_order_value", "purchase_frequency",
    "purchase_gap_mean", "purchase_gap_std",
    "average_delivery_delay", "late_delivery_rate", "average_delivery_days",
    "average_review_score", "negative_review_rate",
    "installment_frequency", "payment_value",
    "recent_90d_revenue", "previous_90d_revenue",
    "purchase_frequency_trend_90d", "review_trend_90d", "recent_delivery_delay",
    "unique_categories", "category_concentration", "unique_sellers", "seller_concentration",
]
CATEGORICAL_FEATURES = ["preferred_payment_method", "customer_state"]
LABEL_COLUMN = "churned"

# Indicator columns for meaningfully-missing features (see doc table)
INDICATOR_SPECS = {
    "has_repeat_purchase": ["purchase_gap_mean", "purchase_gap_std"],
    "has_review": ["average_review_score", "negative_review_rate"],
    "has_delivered_order": ["average_delivery_delay", "late_delivery_rate", "average_delivery_days"],
    "has_recent_delivered_order": ["recent_delivery_delay"],
}


def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for indicator_name, cols in INDICATOR_SPECS.items():
        df[indicator_name] = df[cols].notna().any(axis=1).astype(int)
    df["review_trend_90d"] = df["review_trend_90d"].fillna(0.0)
    return df


def load_data(engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM analytics.customer_snapshot_features", engine)
    return add_missingness_indicators(df)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    indicator_cols = list(INDICATOR_SPECS.keys())

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("indicators", "passthrough", indicator_cols),
    ])


def build_pipeline() -> Pipeline:
    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def evaluate(pipeline: Pipeline, X, y, split_name: str) -> dict:
    proba_churn = pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X)

    # PR-AUC computed with "retained" (0) as the positive label — see
    # docs/baseline_model.md "Framing decision": that's the rare,
    # actually-hard-to-predict class here, not churn.
    pr_auc_retained = average_precision_score((y == 0).astype(int), 1 - proba_churn)
    roc_auc = roc_auc_score(y, proba_churn)
    cm = confusion_matrix(y, pred)
    report = classification_report(y, pred, output_dict=True, zero_division=0)

    print(f"\n=== {split_name} ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (retained=positive class): {pr_auc_retained:.4f}")
    print("Confusion matrix (rows=actual, cols=predicted, [0=retained, 1=churned]):")
    print(cm)
    print(classification_report(y, pred, zero_division=0))

    return {
        "roc_auc": roc_auc,
        "pr_auc_retained_positive": pr_auc_retained,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def main():
    engine = create_engine(DATABASE_URL)
    print("Loading analytics.customer_snapshot_features...")
    df = load_data(engine)
    print(f"  {len(df)} rows loaded")

    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + list(INDICATOR_SPECS.keys())
    X_train, y_train = train[feature_cols], train[LABEL_COLUMN]
    X_val, y_val = validation[feature_cols], validation[LABEL_COLUMN]
    X_test, y_test = test[feature_cols], test[LABEL_COLUMN]

    print(f"\nTrain: {len(X_train)} rows, churn rate {y_train.mean():.2%}")
    print(f"Validation: {len(X_val)} rows, churn rate {y_val.mean():.2%}")
    print(f"Test: {len(X_test)} rows, churn rate {y_test.mean():.2%}")

    print("\nFitting pipeline (impute + scale + one-hot, fit on train only)...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    results = {
        "train": evaluate(pipeline, X_train, y_train, "TRAIN"),
        "validation": evaluate(pipeline, X_val, y_val, "VALIDATION"),
    }
    # Test set is reported but should not drive any further decisions —
    # touching it repeatedly to tune anything would defeat its purpose.
    results["test"] = evaluate(pipeline, X_test, y_test, "TEST (report only, do not tune against this)")

    artifacts_dir = Path(__file__).parent.parent / "artifacts" / "models"
    reports_dir = Path(__file__).parent.parent / "artifacts" / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, artifacts_dir / "baseline_logistic_regression.joblib")
    with open(reports_dir / "baseline_model_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nModel saved to {artifacts_dir / 'baseline_logistic_regression.joblib'}")
    print(f"Metrics saved to {reports_dir / 'baseline_model_metrics.json'}")


if __name__ == "__main__":
    main()
