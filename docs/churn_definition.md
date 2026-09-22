# Churn Definition

Decided from the real inter-purchase gap distribution in
`docs/data_quality_report.md` (Phase 2 audit), computed exactly (not
interpolated) on purchase events (not raw orders), per the project
brief's requirement (§7) not to assume a churn definition up front.

## Context that shapes this decision

97.15% of customers in this dataset never make a second purchase at
all. Whatever window is chosen, it only ever applies to the 2.85% of
customers who return (2,740 customers, measured on purchase events
after correcting for cart-splitting). "Churned" isn't a meaningful
transition state for the other 97% — it's closer to "never activated."
Stated explicitly so the eventual model and dashboard don't imply the
business can meaningfully move 97% of customers out of a "churned"
bucket.

## Unit of measurement: purchase events, not raw orders

Olist assigns a separate `order_id` per checkout, and a cart split
across multiple sellers produces multiple `order_id`s at or near the
same timestamp (confirmed below). All repeat-purchase analysis in this
document uses distinct `(customer_unique_id, order_purchase_timestamp)`
pairs — "purchase events" — as the unit of "a purchase." This is now
the standing definition for the whole project, used in
`docs/data_dictionary.md` and enforced in feature engineering (Phase 7).

## Chosen definition

> A customer is classified as **churned** if no subsequent purchase
> event occurs within **180 days** after their last purchase event.

## Why 180 days — exact coverage, not estimated

Computed directly as `(gaps <= window).mean()` on the purchase-event
gap distribution (2,740 repeat customers, 3,053 gaps, mean 85.3 days,
median 38 days):

| Window | Exact coverage | Genuine repeat gaps mislabeled as churn |
|---|---|---|
| 90 days | 66.72% | 1,016 |
| 120 days | 73.11% | 821 |
| **180 days** | **82.31%** | **540** |
| 270 days | 91.48% | 260 |

## Alternatives considered and rejected

- **90 days** — rejected. Mislabels 1,016 of 3,053 genuine repeat gaps
  (a third of returning-customer behavior) as churn. Too aggressive
  given the actual spread of purchase cadence in this dataset.
- **120 days** — considered, close second. Rejected: still misses
  73.11% coverage, and the jump from 120→180 recovers 9 points of
  coverage (821 → 540 mislabeled) for a window that's still well within
  a plausible "customer hasn't come back" business interpretation.
- **270 days** — rejected as primary definition. Highest coverage
  (91.48%) but calling a customer "active" for the better part of a
  year with no purchase weakens the churn flag's practical usefulness
  for a retention team. Kept as a documented secondary/looser threshold
  for "dormant vs. permanently lost," not the primary label.

## Cart-splitting: confirmed mechanism, partially explained

Of the 999 zero-day-gap order pairs, 292 (29.2%) share the exact same
purchase timestamp — clear cart splits. Cross-checked against
`seller_id`: **62.4% of all zero-gap pairs involve completely different
sellers**, supporting cart-splitting-by-seller as a real, mechanical
cause of Olist's per-order-per-checkout structure.

**Open item:** the remaining 37.6% of zero-gap pairs share at least one
seller in common and are not explained by the seller-split mechanism.
These could be genuine same-day repeat purchases, or a different split
cause (e.g. inventory/fulfillment-driven order splitting within one
seller) not yet investigated. Not resolved here — flagged for Phase 7
if it turns out to matter for feature quality, and named in the risk
register as an unresolved data-generating-process question rather than
silently assumed away.

## Leakage rule

Whatever window is used, feature computation must only use information
that would have been available *as of* the observation date — no
feature may look forward past the point being labeled. Enforced via the
temporal train/validation/test split (Phase 8), not just by care during
feature engineering.

## Status

Resolved. No open items block Phase 7 except the cart-split-remainder
question above, which is non-blocking (affects a minority of an already
small repeat-purchase population, not the churn window decision).
