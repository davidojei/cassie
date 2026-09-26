# Model Comparison — Results, Round 1 (Overfit XGBoost)

Real run, `analytics.customer_snapshot_features`. Kept as a documented
step, not deleted once a better version exists — the overfitting
diagnosis here is exactly what justified the regularized rerun that
follows it.

## Head-to-head

| Metric | Baseline (LogReg) | XGBoost (round 1, unregularized) |
|---|---|---|
| Train ROC-AUC | 0.6275 | **0.8684** |
| Train PR-AUC (retained) | 0.0340 | **0.2087** |
| Validation ROC-AUC | 0.5431 | 0.5267 (worse) |
| Validation PR-AUC (retained) | 0.0176 | 0.0174 (flat) |
| Test ROC-AUC | 0.5579 | 0.5365 (worse) |
| Test PR-AUC (retained) | 0.0393 | 0.0447 (marginal) |
| Train→Validation ROC-AUC gap | 0.08 | **0.34** |

## Verdict: overfitting, not a real improvement

XGBoost fit the training set dramatically better (PR-AUC 6x higher)
but validation performance is flat or slightly worse. A 0.34-point
train/validation gap — versus the baseline's 0.08 — is the standard
signature of a model with too much capacity for the signal actually
available, memorizing training-set noise rather than learning anything
that generalizes. **As configured (200 trees, depth 4, no
regularization, no early stopping), this XGBoost does not beat the
baseline where it matters.**

This is not "XGBoost is the wrong choice" — it's "this specific,
unconstrained configuration overfit." The fix is regularization and
early stopping against the validation set, not switching approaches.
See `docs/model_comparison_round2.md` for that rerun.

## Cold-start hypothesis: supported, modestly

| Split | `lifetime_orders == 1` | `lifetime_orders >= 2` | Gap |
|---|---|---|---|
| Validation | 0.5258 (n=17,836) | 0.5519 (n=187) | +0.026 |
| Test | 0.5195 (n=6,339) | 0.6330 (n=119) | +0.113 |

Consistent direction on both splits — customers with real purchase
history are more predictable than brand-new ones — supports the
cold-start explanation from `docs/baseline_model_results.md`. The
effect is modest on validation and the `>=2` subgroups are small
(especially test's 119), so this is supportive evidence, not proof.
Worth retesting once the regularized model is in place, since an
overfit model's stratified numbers aren't fully trustworthy either.
