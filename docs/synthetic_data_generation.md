# Synthetic Enterprise Data — Generation Design

Fills the gap left since Phase 1/3: `docs/data_dictionary.md` §2 and
`sql/schema/01_raw_schema.sql` always planned for this data, and it was
never actually generated. Built now, prompted by Phase 9's finding that
the churn model has hit a ceiling — but this is infrastructure the
project needed regardless (Phase 11's costs, Phase 12's interventions,
Phase 16's segments), not a one-off experiment.

## The one rule everything else follows

**Every synthetic field is generated from real, already-observed
customer behavior — delivery lateness, review scores, spend, purchase
recency. Never from the future churn label.** `scripts/generate_synthetic_data.py`'s
`load_real_behavior()` is the only function allowed to touch real data,
and its query has no access to anything about what happens after a
customer's last observed purchase. `tests/unit/test_generate_synthetic_data.py`
asserts the behavior input has no label-like column, as a hard check
against this rule quietly breaking later.

Why this matters: if support-ticket rates were wired directly to "will
this customer come back," the churn model would appear to improve for
a fake reason — the improvement would just be a roundabout way of
leaking the answer into the input. If this data helps the model, it
needs to be because these synthetic signals correlate with the *same
underlying behavior* the model already has (weak) access to — a real,
if modest, effect. If it doesn't help, that's an equally honest and
useful finding: it would mean the ceiling isn't about missing feature
categories, it's about the *quantity* of behavioral signal available
per customer, which more tables can't fix.

## A real bug found and fixed here: join fan-out

The first version of `load_real_behavior()` joined `order_items` and
`payments` directly to `orders` in one query. Both are one-to-many
relative to an order (multiple items, multiple payment installments),
so joining both at once fans out: an order with 3 items and 2
installments produces 3×2=6 duplicated rows, silently multiplying
`total_revenue` and `n_late_deliveries` by 6x for that order. This
surfaced as a `ValueError: lam < 0 or lam is NaN` crash (a customer
whose orders had zero item rows produced a NaN revenue percentile) —
the crash was the visible symptom; the inflated-but-not-crashing
numbers for customers with multi-item, multi-payment orders would have
been the dangerous, silent version.

**This is the third time this exact bug category has appeared in this
project** — Phase 3's cart-split double-counting and Phase 7's
order-vs-purchase-event counting were the same underlying mistake:
joining a one-to-many relation without pre-aggregating to the right
grain first. Fixed here the same way as those: `order_items`,
`reviews`, and `payments` are each pre-aggregated to one row per
`order_id` in a CTE before joining to `orders`. Proven with a concrete
test (`tests/unit/test_generate_synthetic_data.py`, test 7) — an
in-memory database with a deliberate 3-item, 2-payment order, asserting
the result is exactly 100 (the real item total), not 600 (what the
fan-out would produce).

## A second, subtler bug found after real model results: span_days leakage

After integrating this data into the churn model (Phase 9 follow-up),
validation ROC-AUC jumped from 0.5431 to 0.6552 — but so did train
performance, dramatically, for **logistic regression specifically**. A
plain linear model suddenly fitting training data much better is a red
flag for leaked information, not a cause for celebration, so this was
traced rather than accepted at face value.

**The bug**: `generate_customer_support()` and
`generate_marketing_campaigns()` originally sized their event-dating
window using `span_days = (last_purchase − first_purchase)` — the
customer's TRUE, full purchase lifespan, including purchases that
happen after any early snapshot. A fixed number of synthetic
events then got spread across that window. Concretely: a true
one-time buyer (churned) has `last_purchase ≈ first_purchase`, so
`span_days` is tiny (~60 days) — the same event count gets packed
densely into a short window. A genuine repeat buyer (not churned) who
returns 8 months later has a much larger `span_days` — the same event
count gets spread thin. Result: `n_campaigns_received` and
`n_support_tickets`, measured as of an early snapshot, came out
systematically higher for customers who turned out to be one-time
buyers — not because of anything realistic about marketing or support
operations, but because the generator used each customer's **actual
future purchase behavior** to decide how tightly to pack their
synthetic timestamps. That's the label leaking in sideways, through
`last_purchase` as a proxy, past the "never correlate with the label
directly" rule — the rule held, the proxy got through anyway.

**The fix**: both generators now use `EVENT_HORIZON_DAYS` (a fixed,
customer-independent 730-day window from `first_purchase`, capped at
`MAX_VALID_DATE`) instead of anything derived from `last_purchase`.
Proven concretely, not just asserted: `tests/unit/test_generate_synthetic_data.py`
test 8 builds two "twin" customers, identical in every real-behavior
input except `last_purchase` (one a true one-timer, one a long-span
repeat buyer), and asserts their generated campaign dates are now
byte-for-byte identical given the same RNG state — proof that
`last_purchase` no longer influences event timing at all.

**Why this matters beyond just fixing it**: this is a good example of
how leakage can sneak past an explicit rule ("never use the label")
through a variable that isn't the label but is causally entangled with
it. `last_purchase` isn't `churned`, but it's *how churned gets
computed* — using it to shape any other generated field is exactly as
dangerous as using the label directly, just harder to spot.

## Table-by-table generation logic

### `customer_support`
Ticket probability rises with `n_late_deliveries` and falls with
`avg_review_score` — a realistic pattern (complaints follow bad
delivery/product experiences). `issue_type` skews toward
`delivery_delay` for customers with late deliveries. `resolution_hours`
is log-normal, faster for high-priority tickets. `customer_satisfaction`
degrades with resolution time and unresolved status.
**Assumption to flag**: ticket count uses a simplified Poisson draw
capped at 4 per customer, not a fully realistic queueing model — good
enough for directional signal, not a claim of operational realism.

### `marketing_campaigns`
Campaign *frequency* scales with `revenue_percentile` (real businesses
market more to valuable segments — legitimate). Open/click/convert
rates are **independent random cascades** (32%/22%/12%), deliberately
**not** tied to any real behavior — there's no honest way to derive
marketing engagement from the Olist data, so making one up would be
worse than admitting it's unconstrained. **Assumption to flag**: these
rates are illustrative industry-typical numbers, not derived from
anything specific to Cassie's fictional business.

### `customer_costs`
`support_cost` is a real, derived quantity (sum of that customer's
*generated* ticket `resolution_hours` × an illustrative $1.50/hr rate)
— not independently random, legitimately traceable.
`discount_cost` derives from real voucher-payment usage.
`annual_service_cost` and `estimated_processing_cost` are simple
percentages of real revenue. **Assumption to flag**: `acquisition_cost`
is the one largely unconstrained field — no real acquisition-channel
data exists anywhere in Olist to derive it from, so it's a bounded
random draw ($15-60). Said plainly rather than dressed up as derived.

### `customer_segments`
`customer_tier` (bronze/silver/gold/platinum) and `segment` are cut
directly from real `revenue_percentile` — exactly how a real business
would define these tiers, not an arbitrary label.
`account_manager` is assigned only to gold/platinum tiers, from a fixed
pool of 7 fake names.

### `retention_interventions`
Simulates a basic real-world ops process: a customer who's gone quiet
150+ days (measured safely before the dataset's end date, so nothing
here depends on future information) has a 15% chance of having
received some past retention offer. `success_probability` is a bounded
random draw (8-22%, a realistic-if-illustrative win-back rate range) —
**deliberately not** derived from whether the customer actually
returned in the real data, since that would be direct label leakage
dressed up as a "probability" field.

## Honest expectations for the churn model

This is very likely **not** a ceiling-buster. The synthetic tables are
*derived from* the same underlying real behavior (delivery experience,
spend, recency) the churn model can already weakly see — they don't
add independent new information, they repackage existing information
into new columns. A genuinely new signal source (actual marketing
engagement data, actual support transcripts, actual browsing behavior)
would need to come from outside this dataset entirely, which isn't
available here.

**What this data is more reliably useful for**: Phase 11's revenue-at-
risk calculation needs real (or realistic) cost figures to compute
anything meaningful; Phase 12's scenario engine needs intervention
cost/success data to model what-if discount scenarios; Phase 16's
business-analyst mode needs segments to answer segment-level questions
at all. Those phases were blocked without this data regardless of
what it does for churn prediction.

## Reproducibility

Fixed random seed (42, `numpy.random.default_rng`) — reruns produce
identical synthetic data. If the generation logic changes, the seed
alone won't preserve old output; that's expected, not a bug to chase.
