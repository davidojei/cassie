# Business Exploration Findings

From `notebooks/02_business_exploration.ipynb`, run against the loaded
`staging` schema (95,420 customers with revenue attributable — a few
thousand short of the 96,096 total due to canceled/unavailable orders
with no items, not a data-quality issue on its own).

## 1. Repeat customers ARE disproportionately valuable — worth acting on

2.87% of customers (2,734) generate **11.27% of revenue**, at an
average of **R$696 vs R$162** for one-time buyers — a repeat customer
is worth **4.3x** a one-timer on average. Their revenue share (11.27%)
is nearly 4x their population share (2.87%). This matters for Phase 11
(customer value) and Phase 16 (targeting): retention spend aimed at
converting a one-time buyer into a repeat one has a real, quantified
payoff, not just a plausible story.

## 2. This is NOT a classic 80/20 whale business — worth naming explicitly

| Top % of customers | % of revenue |
|---|---|
| 1% | 11.8% |
| 5% | 29.2% |
| 10% | 41.0% |
| 20% | 56.1% |
| 50% | 82.0% |

Top 20% generating only 56% (not the ~80% a typical Pareto business
shows) means revenue is fairly broadly spread across a large number of
moderate-value customers, not concentrated in a small number of
whales. **Practical implication for Phase 16's scenario engine**: an
"only target our biggest spenders" strategy will underperform here
compared to a broader-based retention approach — worth stating as a
finding, not assuming the usual 80/20 framing applies.

## 3. Customer value distribution — one thing to check before Phase 11

Median R$108.63, 90th percentile R$341, 99th percentile R$1,258 — but
the **max is R$18,623**, roughly 15x the 99th percentile. Before using
raw revenue for value tiers, confirm whether this (and any other
extreme outliers) is a legitimate large/bulk buyer or a data artifact
(e.g. a business account, duplicate charge, or an order-item pricing
error). **Open item for Phase 7/11**, not resolved here.

## 4. Geographic concentration is real — named as a risk

**SP (São Paulo) alone is 37.38% of revenue.** SP + RJ + MG together
account for **62.5%** of revenue from just 3 of Brazil's states. This
is a legitimate business-continuity risk worth naming alongside the
churn risk register: a regional disruption (logistics, competition,
economic) in the São Paulo area would materially move overall revenue.

## 5. Category revenue is reasonably diversified

Top category (health_beauty) is only 9.26% of revenue; the top 15
categories span 9.26% down to 2.02% with no single category
dominating. Lower concentration risk here than geography or sellers.

## 6. Seller concentration — a second named risk, adjacent to but distinct from churn

**Top 1% of sellers (30 of 3,095) generate 25.2% of revenue. Top 10%
(309 sellers) generate 66.7%.** This is more concentrated than customer
revenue and matters for the platform's health independent of customer
churn: losing a handful of top sellers would be a bigger single shock
than losing any comparable slice of customers. Outside this project's
churn scope strictly, but worth one line in the executive presentation
(Phase 22) as a related risk Cassie's analysis surfaced.

## 7. Payment behavior

Credit card dominates (73.92% of payments), with a notable average of
**3.5 installments per credit card payment** — consistent with common
Brazilian retail behavior. Boleto (bank slip, 19.04%) is the main
alternative. Installment count is a candidate feature for Phase 7 (may
correlate with order value or customer segment) — not tested yet, just
noted as available.

## 8. Delivery lateness → satisfaction: strong, clean signal

**Late-delivery-outlier orders average a 1.80 review score vs. 4.18 for
on-time orders** — more than a 2-point drop on a 5-point scale, over
850 affected orders. This directly answers the business-question
backlog's "are delivery delays associated with churn?" at the
satisfaction-proxy level: yes, strongly. Actual churn correlation
(does a bad review predict non-return) is still to be tested in
feature engineering (Phase 7), but this is strong enough evidence to
prioritize delivery-lateness features early rather than deprioritize
them.

## 9. Monthly revenue trend — two items for Phase 8's temporal split

- **November 2017 is a clear outlier peak** (R$1.18M, 7,451 orders vs.
  ~R$770K the month before) — consistent with Black Friday seasonality
  in Brazil. Worth being aware of when designing the temporal
  train/validation/test split so a seasonal spike doesn't land
  entirely in one split and skew it.
- **September 2018 has only 1 order (R$166.46)** — this is not a real
  collapse in business, it's the dataset's collection cutoff landing
  mid-month. **Decision: exclude September 2018 (and treat August 2018
  as the effective last full month) from any trend or temporal-split
  analysis** — including a partial month as if it were representative
  would distort any month-over-month feature or model evaluation.
- Similarly, **September–December 2016 is a sparse ramp-up period**
  (as low as 1 order in December 2016) — worth excluding or
  down-weighting in any analysis sensitive to volume, for the same
  reason.

## Summary of decisions made here

1. September 2018 (partial month) and the Sept–Dec 2016 ramp-up period
   are excluded from temporal analysis going forward — flagged for
   Phase 8's split design.
2. The R$18,623 max-value customer (and similar extreme outliers) needs
   inspection before Phase 11 builds value tiers on raw revenue —
   open item, not resolved.
3. Two business risks beyond customer churn are now on record: SP-state
   revenue concentration (62.5% in 3 states) and seller concentration
   (top 10% of sellers = 66.7% of revenue). Both worth a line in the
   final executive presentation even though they're outside the
   project's core churn-model scope.
