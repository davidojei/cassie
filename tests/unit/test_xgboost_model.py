"""
Smoke test for scripts/train_xgboost_model.py — synthetic data, no DB
required. Confirms the shared-preprocessing reuse actually works (the
main risk in this script: importing from train_baseline_model.py and
getting an incompatible feature set) and that sample_weight-based
imbalance handling produces valid predictions.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "scripts"))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from train_baseline_model import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, INDICATOR_SPECS, LABEL_COLUMN,
    add_missingness_indicators, build_preprocessor,
)
from train_xgboost_model import build_xgb_model, stratify_by_history

np.random.seed(0)
n = 400

def make_split(n):
    data = {col: np.random.exponential(scale=50, size=n) for col in NUMERIC_FEATURES}
    df = pd.DataFrame(data)
    df.loc[: n // 2, "purchase_gap_mean"] = np.nan
    df.loc[: n // 2, "purchase_gap_std"] = np.nan
    df["review_trend_90d"] = np.nan
    df["preferred_payment_method"] = np.random.choice(["credit_card", "boleto"], size=n)
    df["customer_state"] = np.random.choice(["SP", "RJ", "MG"], size=n)
    df["lifetime_orders"] = np.random.choice([1, 2, 3], size=n, p=[0.85, 0.10, 0.05])
    df[LABEL_COLUMN] = np.random.choice([0, 1], size=n, p=[0.02, 0.98])
    return add_missingness_indicators(df)

train_df = make_split(n)
val_df = make_split(n // 2)

feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + list(INDICATOR_SPECS.keys())
X_train, y_train = train_df[feature_cols], train_df[LABEL_COLUMN]
X_val, y_val = val_df[feature_cols], val_df[LABEL_COLUMN]

# --- test 1: preprocessor fit on train, transform both -- mirrors main()'s real flow ---
preprocessor = build_preprocessor()
X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)
print("Preprocessor fit/transform (train + validation): PASSED")

# --- test 2: model actually engages early stopping (not just "runs") ---
weights = compute_sample_weight(class_weight="balanced", y=y_train)
model = build_xgb_model()
model.fit(X_train_t, y_train, sample_weight=weights, eval_set=[(X_val_t, y_val)], verbose=False)

assert model.best_iteration is not None, "early stopping should set best_iteration"
assert model.best_iteration + 1 < model.n_estimators, (
    f"best_iteration ({model.best_iteration + 1}) should be well under the "
    f"{model.n_estimators} allowed -- if it hits the cap, early stopping never "
    f"actually triggered and this isn't testing what it claims to"
)
print(f"Early stopping engaged: selected {model.best_iteration + 1} of "
      f"{model.n_estimators} allowed trees: PASSED")

# --- test 3: reassembled pipeline (fitted preprocessor + fitted model) predicts correctly ---
pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
proba = pipeline.predict_proba(X_train)[:, 1]
assert proba.shape == (n,)
assert (proba >= 0).all() and (proba <= 1).all()
print("Reassembled pipeline (fitted preprocessor + fitted model) predicts correctly: PASSED")

# --- test 4: stratify_by_history still works against the reassembled pipeline ---
stratify_by_history(train_df, pipeline, feature_cols, "SMOKE TEST")
print("stratify_by_history(): PASSED")

print("\nALL XGBOOST SMOKE TESTS PASSED")
