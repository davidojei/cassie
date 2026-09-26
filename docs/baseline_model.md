# Baseline Model (Phase 8) — Logistic Regression

## Feature audit before modeling — duplicates found and excluded

`analytics.customer_snapshot_features` has several columns that are
identical or linearly dependent, kept in the table for traceability to
the brief's §12/§13 naming (see `docs/feature_engineering.md`) but
excluded from the model's actual input (`X`) — feeding a linear model
duplicate/collinear columns produces unstable, arbitrarily-split
coefficients, which would directly undermine Phase 14's SHAP work:

| Kept for modeling | Excluded (identical or derivable) | Why |
|---|---|---|
| `days_since_last_purchase` | `days_since_previous_order` | identical formula |
| `orders_last_90_days` | `recent_90d_order_count` | identical formula |
| `recent_90d_revenue` | `revenue_last_90_days` | identical formula |
| `purchase_frequency` | `lifetime_purchase_frequency` | identical formula |
| `average_delivery_delay` | `historical_delivery_delay` | identical formula |
| `recent_90d_revenue`, `previous_90d_revenue` | `spend_trend_90d` | trend = recent − previous, exactly; keeping all three is perfectly collinear |

Also excluded, not because they're duplicates but because they're not
legitimate predictors:

- **`days_to_next_purchase`** — this is how the label was computed. It
  is not observable at prediction time and including it would be pure
  label leakage, not a subtle kind.
- **`customer_unique_id`, `snapshot_date`, `split`** — identifiers/meta,
  not features.
- **`customer_city`** — kept out of the baseline for cardinality
  reasons (hundreds of distinct values in a small dataset); `customer_state`
  is used instead. Worth revisiting with target encoding in a later
  model, not the logistic regression baseline.

## Framing decision: which class is actually "rare" here matters

`churned=1` is the **majority** class (~98.4% of rows) — the opposite
of the usual imbalanced-classification setup where the event of
interest is rare. Predicting "everyone churns" gets ~98% accuracy for
free and is completely useless. The actually hard, actually valuable
prediction is identifying the **~1.6% who return** — that's the
minority class, and it's the one worth a retention team's attention
(the R$18,623-customer style outcome we're trying to catch, not miss).

**Decision:** keep `churned` (1=churn) as the model's target — that
matches the business framing needed downstream (Phase 11's revenue-at-
risk = customer value × P(churn)) — but treat **accuracy as
meaningless** for this problem and evaluate primarily on:
- **PR-AUC**, computed with the minority class (`churned=0`, i.e.
  "retained") as the positive label — this is the metric that actually
  reflects whether the model can find the rare return event
- **ROC-AUC** as a secondary, more familiar reference metric
- A confusion matrix at the default 0.5 threshold, reported but
  expected to look poor given the imbalance — threshold tuning is a
  Phase 9+ concern, not solved here
- **`class_weight="balanced"`** in `LogisticRegression`, so training
  itself doesn't just learn to always predict the majority class

## Missing-value handling

Every imputation value is fit on **train only** and applied unchanged
to validation/test — fitting on the full dataset would leak
distributional information from validation/test into training.

| Feature(s) | Null rate | Strategy |
|---|---|---|
| `purchase_gap_mean`, `purchase_gap_std` | ~98% | train-median impute + `has_repeat_purchase` indicator (these are only defined for customers with 2+ purchases, which is most of the signal here) |
| `average_review_score`, `negative_review_rate` | ~0.85% | train-median impute + `has_review` indicator |
| `average_delivery_delay`, `late_delivery_rate`, `average_delivery_days` | ~4% | train-median impute + `has_delivered_order` indicator |
| `recent_delivery_delay` | ~47.5% | train-median impute + `has_recent_delivered_order` indicator (no delivered order in the trailing 90 days is common and meaningful, not an error) |
| `unique_categories`, `category_concentration`, `unique_sellers`, `seller_concentration`, `average_order_value`, `max_order_value` | ~1.14% | train-median impute (rare edge case, likely an order with no items — not investigated further here, low volume) |
| `review_trend_90d` | ~99.7% | impute 0 (neutral/no-detectable-trend) — requires both a recent AND a prior review to exist, inherently rare; kept for completeness, expected to contribute little |
| `purchase_frequency_trend_90d` | 0% | none needed — always computable as a count difference, even when both counts are 0 |

## Categorical encoding

`preferred_payment_method` and `customer_state`: one-hot encoded, fit
on train categories only. Any category appearing in validation/test but
not train gets an all-zero encoding (handled via
`OneHotEncoder(handle_unknown="ignore")`) rather than erroring.

## Numeric scaling

`StandardScaler`, fit on train only — logistic regression's
coefficients and regularization are scale-sensitive, unlike tree
models (relevant for Phase 9's XGBoost comparison, which won't need
this step).

## What this phase does NOT do

- No hyperparameter tuning (that's implicitly part of the Phase 9
  model-comparison work, not this baseline)
- No threshold optimization for the confusion matrix
- No handling of the `customer_city` cardinality question
- No calibration check (raw predicted probabilities aren't verified to
  be well-calibrated yet — matters for Phase 11's revenue-at-risk
  calculation, which multiplies P(churn) by customer value, so a
  miscalibrated probability directly distorts a dollar figure; flagged
  here as a known gap for a later phase, not solved now)
