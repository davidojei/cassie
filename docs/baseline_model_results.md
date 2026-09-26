# Baseline Model Results (Phase 8)

Real run against `analytics.customer_snapshot_features` (177,067 rows).
See `docs/baseline_model.md` for the modeling design; this doc is the
honest record of what actually happened, kept rather than only writing
up if it had looked good.

## Headline: this baseline is weak, and validation shows almost no signal

| Split | ROC-AUC | PR-AUC (retained=positive) | No-skill PR-AUC (class-0 prevalence) | Lift over random |
|---|---|---|---|---|
| Train | 0.6275 | 0.0340 | 1.63% | ~2.1x |
| Validation | 0.5431 | 0.0176 | 1.53% | **~1.15x — essentially no signal** |
| Test | 0.5579 | 0.0393 | 1.80% | ~2.2x (116 positives — small, noisy sample) |

ROC-AUC near 0.5 means barely-better-than-coin-flip ranking. The PR-AUC
comparison against the no-skill baseline (class prevalence, the correct
reference point for PR-AUC — unlike ROC-AUC's fixed 0.5, PR-AUC's floor
moves with how rare the positive class is) is the more honest read:
**on validation, the model provides almost no lift over randomly
guessing which customers will return.**

## Confusion matrices at the default 0.5 threshold

All three splits show the same pattern: `class_weight="balanced"` push
the model toward flagging far more customers as "retained" than are
actually retained, so recall for class 0 looks decent (56–73%) but
precision is very poor (2%) — for every customer correctly identified
as likely-to-return, roughly 50 are false alarms. This threshold isn't
tuned yet (out of scope for this phase, per `docs/baseline_model.md`),
but it's worth being clear that "decent recall" here is not a real win
given how bad the precision is.

## Leading explanation: this is a cold-start problem, not just weak features

The Phase 7 cross-split-leakage fix assigns each customer to a split
based on their **first appearance**. Validation customers first
purchased in Jan–Feb 2018; test customers in March 2018 — meaning by
construction, these customers have at most 1–2 months of history at
any snapshot in their own split. Train customers (first purchase
Jul–Dec 2017) had up to 6 months to accumulate history within train's
own window. Most of the feature set (`purchase_gap_mean`,
`has_repeat_purchase`, delivery/review history) is inherently sparse
or undefined for a customer who's only had one purchase — which is
most of validation/test by construction.

This is a real, sensible explanation for *why validation is worse than
train* — but it does **not** excuse train's own mediocre 0.63 ROC-AUC.
Even with full history available, a linear model on these features
isn't finding much signal. Two honest possibilities, not mutually
exclusive:

1. The underlying problem is genuinely hard — predicting which
   one-time Olist buyer becomes a rare repeat buyer may not have much
   signal in RFM-style behavioral features at all (no stated purchase
   intent, no browsing data, no marketing-touch data yet in this phase).
2. A linear model can't capture whatever interaction effects do exist
   — this is exactly what Phase 9's XGBoost comparison is designed to
   test, since tree-based models can capture non-linear interactions a
   logistic regression structurally cannot.

## What this result does and doesn't mean

- **Does NOT mean the pipeline is broken.** The leakage fixes, the
  duplicate-feature audit, the missingness indicators, the temporal
  split — all of that is still correct and verified. A correct
  pipeline can still produce a weak model; those are different claims.
- **Does mean** the baseline sets a low, honest bar: ROC-AUC ~0.54–0.63,
  PR-AUC lift over random of ~1.15–2.2x depending on split. Phase 9's
  XGBoost model needs to clear this bar meaningfully to justify the
  added complexity — if it doesn't, that's also a finding worth
  reporting, not a failure to hide.
- **Worth testing directly in Phase 9**: whether performance correlates
  with `lifetime_orders` / accumulated history length would confirm or
  rule out the cold-start explanation rather than leaving it as a
  hypothesis.

## Carried into Phase 9

- The cold-start hypothesis above is untested — worth a direct check
  (e.g. stratify validation performance by how much history each
  customer had at snapshot time) before assuming it's the full story.
- Threshold tuning and calibration are still open (per
  `docs/baseline_model.md`) — not attempted here since the ranking
  quality itself is the more fundamental issue right now.
