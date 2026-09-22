# Business Question Backlog

Every question Cassie must eventually be able to answer with evidence,
grouped by category and mapped to the tool/capability that will answer
it once built. This is the acceptance bar for Phase 15–19 (LLM tools,
Business Analyst mode, RAG, UAT, evaluation).

## Factual / descriptive

- What is our current churn rate? → `get_churn_rate()`
- How has churn changed over time? → `get_churn_trend()`
- Which customer segment is contributing most to revenue at risk? →
  `get_churn_by_segment()` + `get_revenue_at_risk()`
- Which regions have the worst retention? → segment/geography cut of
  `analytics.customer_360` + `analytics.segment_metrics`
- Are delivery delays associated with churn? → SHAP global explanation +
  correlation check (framed as association, not causation)

## Customer investigation

- Why is Customer X considered high risk? → `get_customer_explanation()`
  (SHAP local explanation)
- Which customers should our retention team contact first? →
  `get_high_value_at_risk_customers()` (prioritization score, §17)

## Scenario / financial

- How much revenue is at risk? → `get_revenue_at_risk()`
- What happens if we give high-risk customers a 15% discount? →
  `calculate_retention_scenario()` → `calculate_discount_scenario()`
- What if our intervention success rate is only 20%? → same, with
  `success_probability` parameter overridden
- What if we only target customers worth more than ₦500,000? →
  `calculate_targeting_scenario()`
- What if intervention cost increases by 30%? →
  `calculate_intervention_roi()`
- Which customer segment gives us the best expected retention ROI? →
  `calculate_intervention_roi()` grouped by segment

## Policy / definitional

- What is our churn definition, and why? → `get_kpi_definition()` +
  `docs/*` (RAG)
- What's our retention discount policy / SLA policy / refund policy? →
  RAG over `business_policy.md` etc.

## Meta / evidentiary (the ones that matter most for trust)

- What evidence supports that conclusion?
- What assumptions did you use?
- Show me the calculation.
- What data is missing before we can answer this confidently?

These four are not separate tools — they're a standing requirement on
every other answer (cite the tool/metric used, distinguish observed
fact from model prediction from assumption from recommendation, and
say plainly when the data doesn't support an answer). Tracked
explicitly in the LLM evaluation set (§33) so this behavior is tested,
not just aspired to.

## Requirements-translation mode

- "Turn this business problem into requirements" → Business Analyst mode
  (§29): generates business requirement, functional/non-functional
  requirements, user stories, acceptance criteria, business rules, KPIs,
  risks, assumptions, dependencies for a given prompt.

---

This backlog is the input to the 50-question LLM evaluation set (§33)
and the 20+ UAT cases (§31) — both expand on these, they don't replace
them.
