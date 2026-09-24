# Database Schema Proposal

PostgreSQL, five schemas, one direction of flow: `raw` → `staging` →
`analytics` → `ml` / `business` (the latter two both read from
`analytics`, don't feed back into it).

## `raw`

Untouched loads, one table per source file, column names/types kept as
close to source as practical. No joins, no derived columns, no
deduplication.

- `raw.olist_customers`, `raw.olist_orders`, `raw.olist_order_items`,
  `raw.olist_order_payments`, `raw.olist_order_reviews`,
  `raw.olist_products`, `raw.olist_sellers`, `raw.olist_geolocation`,
  `raw.product_category_translation`
- `raw.customer_support`, `raw.marketing_campaigns`,
  `raw.customer_costs`, `raw.customer_segments`,
  `raw.retention_interventions` *(synthetic — same naming convention,
  distinguishable only by knowing the data dictionary; consider a
  `is_synthetic` marker or a separate `synthetic` schema if ambiguity
  becomes a real risk during Phase 3)*

## `staging`

Cleaned, typed, deduplicated, deterministic (no business logic yet).

- `staging.orders_clean` — one row per `order_id`, cast timestamps,
  dropped/flagged clearly invalid rows (see data-quality plan)
- `staging.order_items_clean`, `staging.payments_clean`,
  `staging.reviews_clean`, `staging.customers_clean` (with
  `customer_unique_id` as the primary grouping key surfaced explicitly)

## `analytics`

Business-meaningful, query-ready views/tables. This is what the ML
feature pipeline and the dashboard both read from.

- **`analytics.customer_snapshot_features`** — built in Phase 7
  (`scripts/build_features.py`), documented in
  `docs/feature_engineering.md`. Supersedes the two lines originally
  planned here (`customer_360` for features,
  `customer_retention_features` for churn labels) — those were merged
  into one table, because computing features and labels separately per
  snapshot risks exactly the kind of temporal/customer-leakage bug
  found and fixed in Phase 7 (see that doc). One row per
  (`customer_unique_id`, `snapshot_date`): RFM, behavior, delivery, CX,
  payment, product, seller, geography, and trend features (brief
  §12–13), plus the leakage-safe `churned` label and a `split` column
  (train/validation/test, assigned by each customer's first-appearance
  cohort, not by snapshot date alone).
- `analytics.customer_monthly_metrics` — one row per
  `customer_unique_id` × month — not yet built
- `analytics.customer_revenue_metrics` — not yet built
- `analytics.customer_risk_metrics` — post-model, joined predictions +
  SHAP summary (populated once `ml.churn_predictions` exists) — not yet
  built
- `analytics.segment_metrics` — not yet built

## `ml`

- `ml.model_runs` — one row per training run: model type, params, metric
  scores, temporal cutoff, git commit / code version
- `ml.churn_predictions` — one row per `customer_unique_id` × model_run:
  probability, predicted class, calibration bucket

## `business`

- `business.revenue_at_risk` — one row per `customer_unique_id` (or
  aggregated by segment/geography/category/tier/risk-bucket, per §16):
  customer value, churn probability, revenue at risk, expected
  recoverable revenue
- `business.intervention_scenarios` — one row per scenario run (what-if
  discount/targeting/ROI queries), stores inputs + outputs for
  reproducibility
- `business.decisions` — the decision log (§36): `decision_id`,
  `timestamp`, `question`, `evidence`, `assumptions`, `recommendation`,
  `financial_impact`, `model_version`, `data_period`

## Open questions for Phase 3

- Exact column-level DDL (types, constraints, indexes) — written when
  the schema is actually created, after the Phase 2 data audit confirms
  real column types/ranges rather than assuming them here.
- Whether `raw` synthetic tables need their own schema for clarity.
- Partitioning/indexing strategy for `analytics.customer_monthly_metrics`
  if it gets large.
