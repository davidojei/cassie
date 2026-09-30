"""
Smoke test for scripts/explain_model.py — synthetic data, no DB or
saved model required. The main thing this checks: that every technical
feature name actually gets translated to plain language, never leaks
through raw (this is the entire point of this phase, so it gets the
most scrutiny, not just "does SHAP run").
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "scripts"))

import numpy as np
import pandas as pd

from train_baseline_model import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, INDICATOR_SPECS, LABEL_COLUMN,
    add_missingness_indicators, build_pipeline,
)
from explain_model import (
    plain_label, describe_value, compute_shap, global_importance, explain_customer,
    FEATURE_LABELS,
)

np.random.seed(0)
n = 300

data = {col: np.random.exponential(scale=50, size=n) for col in NUMERIC_FEATURES}
df = pd.DataFrame(data)
df.loc[:150, "purchase_gap_mean"] = np.nan
df.loc[:150, "purchase_gap_std"] = np.nan
df["review_trend_90d"] = np.nan
df["preferred_payment_method"] = np.random.choice(["credit_card", "boleto"], size=n)
df["customer_state"] = np.random.choice(["SP", "RJ", "MG"], size=n)
df[LABEL_COLUMN] = np.random.choice([0, 1], size=n, p=[0.1, 0.9])
df["customer_unique_id"] = [f"cust_{i}" for i in range(n)]

result = add_missingness_indicators(df)
feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + list(INDICATOR_SPECS.keys())
X, y = result[feature_cols], result[LABEL_COLUMN]

pipeline = build_pipeline()
pipeline.fit(X, y)

# --- test 1: every technical feature label maps to something WITHOUT
# leftover technical artifacts -- no "__", no "numeric_" prefix, no bare
# snake_case dumped straight through unmapped ---
sample_technical_names = (
    [f"numeric__{c}" for c in NUMERIC_FEATURES]
    + [f"indicators__{c}" for c in INDICATOR_SPECS]
    + ["categorical__preferred_payment_method_credit_card", "categorical__customer_state_SP"]
)
for tech_name in sample_technical_names:
    label = plain_label(tech_name)
    assert "__" not in label, f"leaked technical prefix in label: {tech_name} -> {label}"
    assert not label.islower() or " " in label, f"suspiciously raw label: {tech_name} -> {label}"
print(f"All {len(sample_technical_names)} technical names translate to clean plain labels: PASSED")

# every entry in FEATURE_LABELS must itself be a real sentence fragment,
# not just the technical name capitalized (i.e. someone actually wrote it)
for tech, label in FEATURE_LABELS.items():
    assert label != tech.replace("_", " ").capitalize(), (
        f"FEATURE_LABELS['{tech}'] looks auto-generated, not hand-written: {label}"
    )
print(f"All {len(FEATURE_LABELS)} explicit labels are hand-written, not auto-derived: PASSED")

# --- test 2: describe_value produces readable sentences, handles NaN ---
desc = describe_value("numeric__days_since_last_purchase", 165, reference_median=40.2)
assert "165" in desc and "40.2" in desc and "Days since their last purchase" in desc
print(f"describe_value() with reference median: '{desc}': PASSED")

desc_nan = describe_value("numeric__purchase_gap_mean", np.nan, reference_median=30.0)
assert "not available" in desc_nan
print(f"describe_value() handles NaN: '{desc_nan}': PASSED")

desc_bool = describe_value("indicators__has_repeat_purchase", 1, reference_median=None)
assert "Yes" in desc_bool
print(f"describe_value() handles boolean indicators: '{desc_bool}': PASSED")

# --- test 3: full SHAP pipeline runs and produces sane shapes ---
background = X.sample(50, random_state=1)
explain_set = X.sample(20, random_state=2).reset_index(drop=True)
shap_values, feature_names, _ = compute_shap(pipeline, background, explain_set)
assert shap_values.shape == (20, len(feature_names))
print(f"SHAP values computed: shape {shap_values.shape}: PASSED")

importance = global_importance(shap_values, feature_names, top_n=10)
assert len(importance) == 10
assert importance["mean_abs_shap"].is_monotonic_decreasing
assert all("__" not in lbl for lbl in importance["plain_label"])
print("Global importance table: sorted, plain-language, no leaked prefixes: PASSED")

# --- test 4: customer explanation is a readable multi-line string with
# direction markers, not a dump of numbers ---
ref_medians = X[NUMERIC_FEATURES].median().to_dict()
explanation = explain_customer(0, explain_set, shap_values, feature_names, ref_medians, top_k=5)
assert "increases churn risk" in explanation or "decreases churn risk" in explanation
assert "__" not in explanation
lines = explanation.strip().split("\n")
assert len(lines) == 5
print(f"Customer explanation ({len(lines)} factors, direction-labeled, no raw names): PASSED")
print(explanation)

print("\nALL SHAP EXPLAINABILITY SMOKE TESTS PASSED")

# --- test 5: has_been_targeted_for_retention is suppressed from customer
# explanations but the list still backfills to full length, not just shorter ---
from explain_model import EXCLUDE_FROM_CUSTOMER_EXPLANATION

# force this feature to have the single largest SHAP magnitude for a
# customer, so it would DEFINITELY appear at rank 1 if not suppressed
forced_shap = shap_values.copy()
targeted_idx = list(feature_names).index("numeric__has_been_targeted_for_retention")
forced_shap[0, targeted_idx] = 999.0  # artificially the biggest contributor

explanation_with_suppression = explain_customer(0, explain_set, forced_shap, feature_names, ref_medians, top_k=5)
assert "Has previously received a retention offer" not in explanation_with_suppression, (
    "suppressed feature leaked into the customer-facing explanation"
)
lines = explanation_with_suppression.strip().split("\n")
assert len(lines) == 5, f"suppression should backfill to full top_k, got {len(lines)} lines instead of 5"
print("Suppressed feature excluded from customer explanation AND list backfills to full length: PASSED")

# global importance is explicitly UNAFFECTED by the suppression list
importance_check = global_importance(forced_shap, feature_names, top_n=5)
assert importance_check.iloc[0]["plain_label"] == "Has previously received a retention offer", (
    "global importance should still surface this feature -- suppression is customer-facing only"
)
print("Global importance is unaffected by customer-facing suppression (as designed): PASSED")
