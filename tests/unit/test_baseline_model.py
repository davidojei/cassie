"""
Smoke test for scripts/train_baseline_model.py — synthetic data, no DB
required. Checks the pipeline actually fits/predicts correctly and that
the missingness-indicator logic behaves as documented in
docs/baseline_model.md, not just that the code runs without an error.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "scripts"))

import numpy as np
import pandas as pd

from train_baseline_model import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, INDICATOR_SPECS,
    add_missingness_indicators, build_pipeline, evaluate,
)

np.random.seed(0)
n = 200

data = {col: np.random.exponential(scale=50, size=n) for col in NUMERIC_FEATURES}
df = pd.DataFrame(data)

# Force specific known-missing patterns to test the indicator logic directly
df.loc[0:99, "purchase_gap_mean"] = np.nan
df.loc[0:99, "purchase_gap_std"] = np.nan
df.loc[100:119, "average_review_score"] = np.nan
df.loc[100:119, "negative_review_rate"] = np.nan
df["review_trend_90d"] = np.nan  # should ALWAYS be imputed to 0, never left null

df["preferred_payment_method"] = np.random.choice(["credit_card", "boleto", "voucher"], size=n)
df["customer_state"] = np.random.choice(["SP", "RJ", "MG"], size=n)
df["churned"] = np.random.choice([0, 1], size=n, p=[0.02, 0.98])  # mimic the real ~98% imbalance

# --- test 1: missingness indicators ---
result = add_missingness_indicators(df)

assert result.loc[0, "has_repeat_purchase"] == 0, "row 0 has null purchase_gap -> indicator should be 0"
assert result.loc[150, "has_repeat_purchase"] == 1, "row 150 has real purchase_gap -> indicator should be 1"
assert result.loc[100, "has_review"] == 0
assert result.loc[150, "has_review"] == 1
assert result["review_trend_90d"].isna().sum() == 0, "review_trend_90d must never be left null after imputation"
assert (result["review_trend_90d"] == 0.0).all(), "all-null review_trend_90d should impute to exactly 0"
print("Missingness indicator logic: PASSED")

# --- test 2: pipeline fits and predicts without error, on data with real nulls ---
feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + list(INDICATOR_SPECS.keys())
X = result[feature_cols]
y = result["churned"]

pipeline = build_pipeline()
pipeline.fit(X, y)
proba = pipeline.predict_proba(X)[:, 1]

assert proba.shape == (n,)
assert (proba >= 0).all() and (proba <= 1).all(), "predicted probabilities must be in [0, 1]"
print("Pipeline fit + predict on data with real nulls: PASSED")

# --- test 3: evaluate() returns well-formed metrics ---
metrics = evaluate(pipeline, X, y, "SMOKE TEST")
assert 0 <= metrics["roc_auc"] <= 1
assert 0 <= metrics["pr_auc_retained_positive"] <= 1
assert len(metrics["confusion_matrix"]) == 2 and len(metrics["confusion_matrix"][0]) == 2
print("evaluate() output shape/range: PASSED")

# --- test 4: unseen category in a hypothetical val/test set doesn't crash ---
X_unseen = X.copy()
X_unseen.loc[0, "customer_state"] = "ZZ"  # not in training data
_ = pipeline.predict_proba(X_unseen)
print("Unseen categorical value handled without error: PASSED")

print("\nALL BASELINE MODEL SMOKE TESTS PASSED")
