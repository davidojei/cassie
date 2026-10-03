# Phase 11 — Customer Value & Revenue at Risk

## What this phase does

Converts the model's churn probability into a dollar figure retention
teams can prioritize against:

```
revenue_at_risk = customer_value x predicted_churn_probability
```

Output: `outputs/customer_revenue_at_risk.csv`, one row per active
customer as of the scoring snapshot (2018-08-31, the latest date the
full dataset supports).

## Key design decisions

- **`customer_value = lifetime_revenue`**, deliberately kept simple —
  no annualization, no margin adjustment. The churn probability feeding
  this formula comes from a model with ROC-AUC ~0.55-0.65; multiplying
  an already-uncertain probability by an additional model-derived value
  estimate would compound two layers of uncertainty into a number that
  looks more precise than the pipeline actually supports.
- **`customer_costs` shown alongside, not netted** into a single
  "margin at risk" figure, for the same reason — two honestly-labeled
  numbers beat one falsely-precise one.
- **Scored on a new snapshot at `MAX_VALID_DATE` (2018-08-31)**, not a
  training snapshot — this is "what's the risk right now," not a
  historically-labelable point. Reuses `build_snapshot()` unchanged.
- **`churned` and `days_to_next_purchase` are dropped** from this
  snapshot — both are meaningless at the true end of the data (no
  future purchase exists to check against, so every customer would
  trivially show `churned=1` by construction).
- **`risk_tier`**: three tiers (low/medium/high), cutoffs at the median
  and 80th percentile of the real scoring distribution, confirmed after
  seeing actual numbers rather than guessed in advance. Deliberately no
  finer-grained tiering — the model's own weak ROC-AUC doesn't support
  more precision than that.
- **Every `revenue_at_risk` figure carries the model-caveat string** as
  an output column, so it can't get separated from the number in any
  downstream report or dashboard.

## Bug found this phase

**`customers` table fan-out from non-unique address history** — see
project state doc, "bug #6." `load_base_tables()`'s
`SELECT DISTINCT customer_unique_id, customer_state, customer_city`
assumed one address per customer; 122 customers had more than one
`(state, city)` pair on file, so `DISTINCT` silently returned multiple
rows per customer instead of collapsing to one. This fanned out into
every snapshot via `build_snapshot()`'s merge — not just a scoring
bug, a training-data bug present in `analytics.customer_snapshot_features`
since Phase 7 (766 duplicate rows, 103 customers, out of 177k rows).

Fixed by deduplicating `customers` to each customer's most frequent
`(state, city)` pair before the merge. Features rebuilt, model
retrained: ROC-AUC moved negligibly (0.645/0.555/0.568 vs. the prior
~0.55-0.65 range) — confirms small practical impact, but it was a real
correctness issue worth fixing regardless of size. Same root cause
category as bugs #1-#3 (cart-split orders, `lifetime_orders` raw-row
counting, synthetic-data join fan-out): joining/aggregating across a
one-to-many relationship without pre-aggregating first.

## Sanity checks run

- **Feature-stat diff** (scoring snapshot vs. last training snapshot):
  9 features showed a >5-point null-rate shift. All traced to plausible
  temporal effects at the true data boundary (e.g. `recent_delivery_delay`
  is 81% null in scoring vs. 2.5% in training, because orders placed in
  the last 90 days before the cutoff haven't had time to be delivered
  yet) rather than bugs.
- **Recency-skew check** (cold-start hypothesis, flagged unresolved in
  earlier phases): predicted churn probability rises smoothly and
  monotonically with customer tenure (newest quartile ~0.37, oldest
  ~0.63) — no flat or extreme newest-customer artifact. Does not
  resolve the cold-start question, but shows nothing alarming at this
  snapshot.
- **Smoke test**: no nulls in `revenue_at_risk`, probability in [0,1],
  no duplicate `customer_unique_id` rows (this is what caught bug #6).

## Results

| Metric | Value |
|---|---|
| Customers scored | 96,089 |
| `predicted_churn_probability` median | 0.492 |
| `revenue_at_risk` median | $48.71 |
| `revenue_at_risk` mean | $85.31 |
| Risk tier counts | low 48,045 / medium 28,826 / high 19,218 |

~98% churn rate across the dataset is expected, not a symptom of the
bug fix — ~97% of customers are single-order buyers, and under a
"no purchase within 180 days" definition a one-time buyer is churned
by construction almost regardless of snapshot date (consistent with
Phase 10's findings).

## Open items / not done this phase

- Aggregating `revenue_at_risk` by segment/state/tier for a summary
  view — planned, not yet built (see
  `docs/business_question_backlog.md`).
- No change to the model itself; Phase 11 consumes Phase 9's chosen
  logistic regression as-is.

## Next phase

Phase 12: scenario engine.
