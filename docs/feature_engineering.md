# Feature Engineering (Phase 7)

Implements `docs/database_schema_proposal.md`'s
`analytics.customer_retention_features` as
`analytics.customer_snapshot_features`, built by `scripts/build_features.py`.

## The core design problem this solves

The brief (§11) requires no random splitting and no future information
leaking into features. For a churn model, that means: for every row
used in training, the features must be computable from data that would
have existed *before* the point being labeled, and the label must be
determined only from what happened *after* that point.

**Approach: snapshot-based feature generation.** Rather than one
feature row per customer using all their history, this generates
multiple rows per customer — one per monthly "snapshot date" — where:

- every feature uses only orders/reviews/payments with
  `order_purchase_timestamp <= snapshot_date`
- the label (`churned`) uses only purchase events strictly *after*
  `snapshot_date`

This also multiplies the training data (9 snapshots × active customers
per snapshot, versus one row per customer), which matters given how
small the true repeat-buyer population is (2.85% of customers, per
`docs/churn_definition.md`).

## Snapshot dates and why

Monthly, **2017-07-01 through 2018-03-01** (9 snapshots). Two
constraints set this range:

1. **No censoring.** Every snapshot + 180-day churn window must fall
   within real, observed data. The last full month of real order
   volume is August 2018 (`docs/business_exploration_findings.md`,
   finding 9 — September 2018 is a one-order artifact of the dataset's
   collection cutoff, not real). So the latest usable snapshot is
   `2018-08-31 − 180 days ≈ 2018-03-04`, rounded down to 2018-03-01.
2. **Real history to draw features from.** The dataset starts
   2016-09-04, but Sept–Dec 2016 is a sparse ramp-up period (as low as
   1 order in December 2016, per the same findings doc). 2017-07-01 was
   chosen as the earliest snapshot to give even the first snapshot
   several months of realistic order volume as feature history, while
   still leaving 6 snapshots for the training split.

Because every snapshot satisfies constraint 1 by construction, **no
per-customer censoring check is needed** — this is a direct
consequence of the snapshot range, not a separate rule to remember or
enforce later.

## Split — temporal, not random

| Split | Snapshots |
|---|---|
| train | 2017-07, 08, 09, 10, 11, 12 |
| validation | 2018-01, 02 |
| test | 2018-03 |

Earlier history trains, later periods validate/test — directly
following §11's instruction, and avoiding the standard failure mode
of random-splitting a temporal churn dataset (where a model can
effectively see the future via correlated snapshots of the same
customer landing in both train and test).

## A customer only appears in a snapshot if they've bought at least once by then

Recency, frequency, and monetary features are undefined for someone
with zero purchase history as of the snapshot date — they're excluded
from that snapshot's rows, not given a zero/null row. They may still
appear in later snapshots once they have a first purchase.

## Feature list — mapped to brief §12/§13

All features from §12's minimum list are implemented. A few needed an
explicit definition where the brief named a metric without fully
specifying it — documented here so it's clear this was a decision, not
an oversight:

- **`average_delivery_delay`**: defined as `delivered_date −
  estimated_delivery_date` (positive = later than promised), distinct
  from `average_delivery_days` (raw purchase-to-delivery duration,
  same metric as the `late_delivery_outlier` flag in
  `staging.orders_clean`, but averaged rather than thresholded here).
  These are two different things the brief's naming doesn't
  distinguish, so both are computed under separate names.
- **`category_concentration` / `seller_concentration`**: implemented
  as an HHI-style index (sum of squared revenue shares, 0–1, higher =
  more concentrated on one category/seller) rather than a simple
  top-1 share — standard concentration measure, and consistent with
  how seller/geography concentration was already measured in Phase 5.
- **`preferred_payment_method`**: mode of `payment_type` across the
  customer's orders as of the snapshot.
- **Trend features (§12) and advanced trend features (§13)** are
  implemented as the same underlying recent-90-days-vs-previous-90-days
  comparison, exposed both as a delta (`spend_trend_90d`,
  `purchase_frequency_trend_90d`, `review_trend_90d`) and as the raw
  paired values (`recent_90d_revenue` / `previous_90d_revenue`, etc.)
  — the raw pairs are kept, not just the delta, because SHAP
  explainability (Phase 14) will be more interpretable against "recent
  vs. historical" pairs than a single trend number.

## Cross-split customer leakage — found and fixed after the first real run

The first run against real data (96,096 customers) revealed a real
problem the design above didn't account for: a customer active by an
early snapshot is "active" (by the >=1-past-purchase rule) in every
later snapshot too, since nothing ever removes them. Verified directly:
**100% of the 38,547 train customers also appeared in the test split.**

This is a distinct issue from temporal leakage (which the snapshot
design does correctly prevent — each row's features/label only use
data before/after its own snapshot date). This is *customer-level*
leakage: a model can learn customer-specific patterns from a person's
train-split rows and then get evaluated on that same person in test,
inflating apparent generalization.

**Fix:** each customer is assigned exclusively to the split of their
**first appearance** (earliest qualifying snapshot). They keep all
their rows within that split's snapshot window — still several
observations over time, which is normal, valid panel-data structure —
but are excluded from any other split entirely, even though they'd
otherwise still qualify as "active" there.

Practical effect: validation and test now only contain customers who
are *new* as of that period (first purchase in Jan/Feb 2018 for
validation, March 2018 for test), not "everyone active by then." This
is smaller than the original row counts but is the version that
actually measures what it claims to measure — generalization to
customers the model hasn't seen, not memorization dressed up as
generalization.

## Known open items — not resolved in this pass

1. **Run against real data, cross-split leakage found and fixed** (see
   above) — this replaced the original "not yet run" status. Row counts
   post-fix haven't been re-verified against a second real run yet;
   next step is rerunning `build_features.py` with the fix in place and
   confirming zero train/test customer overlap.
2. **The R$18,623 outlier customer** flagged in
   `docs/business_exploration_findings.md` hasn't been inspected yet.
   It will now show up as an extreme `lifetime_revenue` /
   `max_order_value` value in whichever snapshots include it — worth
   checking once real output exists, before Phase 8 trains anything on
   it un-investigated.
3. **`unique_categories`/`unique_sellers`/concentration features use
   item price only**, excluding freight — freight is a shipping cost,
   not product revenue, so this was a deliberate choice, but it means
   `category_concentration`'s revenue base differs slightly from
   `lifetime_revenue`'s (which includes freight). Not a bug, just
   worth knowing the two aren't directly comparable in scale.
