# Model Comparison — XGBoost vs. Baseline (Phase 9)

## Design: isolate the model, not the pipeline

`scripts/train_xgboost_model.py` imports its feature list, missingness
handling, preprocessing (`build_preprocessor()`), and evaluation
function directly from `scripts/train_baseline_model.py` rather than
redefining them. This is deliberate: if the two models were evaluated
on different feature representations, a performance difference
couldn't be attributed to the model architecture — it could just as
easily be a preprocessing difference. Keeping everything identical
except the estimator itself is what makes "XGBoost beat logistic
regression by X" (or didn't) a claim actually worth trusting.

This required a small refactor of `train_baseline_model.py`: the
preprocessing `ColumnTransformer` construction was pulled out into its
own `build_preprocessor()` function so both scripts call the same code,
not two copies that could quietly drift apart over time.

## Imbalance handling: same methodology, not just "handled somehow"

The baseline used `class_weight="balanced"` inside `LogisticRegression`.
XGBoost's usual imbalance lever (`scale_pos_weight`) only reweights the
positive label (1) relative to negative — but here label 1 (`churned`)
is the *majority* class, so that parameter's usual "upweight the rare
positive" framing doesn't apply cleanly. Instead,
`compute_sample_weight(class_weight="balanced", y)` is used directly —
the same weighting *computation* sklearn's `class_weight="balanced"`
does internally, applied as explicit `sample_weight` at fit time. Same
imbalance-handling methodology across both models, not two different
strategies that would be another confound.

The smoke test (`tests/unit/test_xgboost_model.py`) asserts the weight
actually goes to the correct (minority) class regardless of which label
value happens to be rare — this direction is easy to get backwards when
adapting boilerplate written for the usual "positive=rare" case.

## Testing the cold-start hypothesis directly, not leaving it as speculation

`docs/baseline_model_results.md` proposed but did not test that
validation/test's weak performance is a cold-start effect — new
customers with little accumulated history. `stratify_by_history()`
splits validation and test by `lifetime_orders == 1` (no repeat-purchase
signal at all) vs. `>= 2` and reports ROC-AUC separately for each. If
the hypothesis is right, the `== 1` group should show notably worse
ROC-AUC than the `>= 2` group. If not, the weak validation performance
needs a different explanation.

## What to look for in the real results

1. **Does XGBoost's validation ROC-AUC/PR-AUC clear the baseline's
   (0.5431 / 0.0176)?** If not by a meaningful margin, that's a real
   finding — it would suggest the ceiling here is feature quality, not
   model architecture, and worth saying plainly rather than picking
   whichever number looks better.
2. **Does the cold-start stratification confirm the hypothesis?** A big
   gap between the two `lifetime_orders` groups supports it; a small
   gap means Phase 8's explanation needs revisiting.
3. **Train vs. validation gap for XGBoost** — trees can overfit more
   easily than a regularized linear model on a table this size; a much
   bigger train/validation gap than the baseline's would itself be
   worth naming as an overfitting concern, not just reported as "the
   number."
