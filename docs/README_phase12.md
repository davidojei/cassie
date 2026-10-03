# Phase 12 — Scenario Engine (scenarios 1-3)

## What this phase does

Turns Phase 11's `revenue_at_risk` output into answerable business
questions: what does a campaign cost vs. return, how should a fixed
budget be spent, and what happens if risk-tier cutoffs change.

Scope note: a 4th scenario (churn-window what-if) was identified during
design but deferred — it requires re-running the feature/train/score
pipeline, not a cheap calculation, so it's its own follow-up rather
than bundled in here as if it were the same kind of thing.

## Key design decision: retention_effect_pct is an assumption, not data

`retention_interventions.success_probability` was generated as a
bounded random draw independent of actual customer outcomes
(`docs/synthetic_data_generation.md`) — specifically designed to avoid
leaking real outcomes into a synthetic field. Estimating a "retention
effect" from it would present a number this project manufactured as if
it were measured. Instead, `retention_effect_pct` is a clearly-labeled,
user-supplied input, with three illustrative presets (conservative 10%,
moderate 20%, optimistic 35%) — explicitly NOT derived from this
project's data or any external benchmark — plus free entry for any
value in (0, 1]. Every scenario output restates the chosen percentage
and its caveat alongside the model's own ROC-AUC caveat, so neither can
get silently dropped downstream.

## Scenarios implemented

**Campaign ROI** — target a group (top N by `revenue_at_risk`, a
`risk_tier`, or a `segment`), apply a cost-per-customer and a retention
effect, get total cost, total expected revenue saved, net benefit, and
ROI%.

**Budget allocation** — given a fixed budget and cost-per-customer,
greedily ranks all customers by expected revenue saved per dollar and
selects until the budget is spent. Also reports a marginal-value curve
at 0.5x-2x the stated budget, so the output shows whether more budget
still has room to pay off rather than presenting one number as
obviously correct.

**Risk-tier what-if** — re-buckets the existing (unchanged)
`predicted_churn_probability` column under different cutoffs, either
quantile-based (matching Phase 11's style) or absolute probability
thresholds (business-meaningful breakpoints). Reports a movement table
showing how many customers shift tiers vs. the current Phase 11
assignment. This is cheap — no model or feature recomputation, pure
re-bucketing of numbers that already exist.

## Smoke tests

- `campaign_roi`: revenue-saved and cost totals are finite; never
  targets more customers than exist in the input.
- `budget_allocation`: never targets more customers than the stated
  budget actually covers; total spent never exceeds the budget.
- Every scenario's output dict includes both caveats explicitly, never
  defaulted or omitted.

## Interface

All three scenarios are plain functions in `scripts/scenario_engine.py`
(`campaign_roi()`, `budget_allocation()`, `risk_tier_whatif()`),
importable as-is — intentionally built with no UI assumptions, so
Phase 13 (FastAPI) and Phase 14 (dashboard) can wrap them without this
phase having guessed what either interface needs. A CLI (`argparse`
subcommands) is included for quick manual checks.

## Results (real run against the Phase 11 output)

**Campaign ROI** — targeting the `high` risk tier (19,218 customers)
at $15/customer, moderate (20%) retention effect:

| Metric | Value |
|---|---|
| Customers targeted | 19,218 |
| Campaign cost | $288,270 |
| Expected revenue saved | $616,661.97 |
| Net expected benefit | $328,391.97 |
| ROI | +113.9% |

**Budget allocation** — $50,000 budget, $15/customer, conservative
(10%) retention effect: targeted 3,333 customers, spent $49,995,
expected revenue saved $222,476.28, net benefit $172,481.28.

Marginal-value curve confirms diminishing returns as intended — the
engine isn't just producing plausible-looking numbers, the shape is
right: going from $25k→$37.5k nets +$22.9k in additional benefit, but
$75k→$100k only nets +$11.8k more, since the highest expected-value
customers get captured first under the greedy ranking:

| Budget | Targeted | Revenue saved | Net benefit |
|---|---|---|---|
| $25,000 | 1,666 | $158,527.93 | $133,537.93 |
| $37,500 | 2,500 | $193,890.95 | $156,390.95 |
| $50,000 | 3,333 | $222,476.28 | $172,481.28 |
| $62,500 | 4,166 | $246,790.74 | $184,300.74 |
| $75,000 | 5,000 | $268,140.96 | $193,140.96 |
| $100,000 | 6,666 | $304,890.71 | $204,900.71 |

**Risk-tier what-if** — absolute cutoffs [0, 0.4, 0.7, 1.0] vs. the
current quantile-based tiers from Phase 11:

| New tier | Customers | Total revenue at risk | Mean churn probability |
|---|---|---|---|
| low | 29,894 | $1,588,831 | 0.293 |
| medium | 47,990 | $3,624,137 | 0.530 |
| high | 18,205 | $2,984,506 | 0.911 |

Movement table shows the new `medium` tier (28,826 customers, 0.4-0.7
probability) lines up almost exactly with the old quantile-based
`medium` tier (28,826 customers) — a sanity-check cross-confirmation
that both tiering schemes land in a similar place given the real
probability distribution (median 0.492, 80th percentile ~0.64).

## Open items / deferred

- Scenario 4 (churn-window what-if): needs an orchestration script
  that chains `build_features.py` → `train_baseline_model.py` →
  `score_customer_value.py` with a different `CHURN_WINDOW_DAYS`,
  reports results side-by-side with the 180-day baseline. Flagged as
  "takes minutes, not milliseconds, produces a genuinely different
  model" — not built this phase.
- `cost_per_customer` is currently a single flat number across all
  scenarios; a per-segment cost variant wasn't requested and wasn't
  built.

## Next phase

Phase 13: FastAPI.
