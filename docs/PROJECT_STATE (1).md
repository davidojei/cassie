# Cassie — Project State (for resuming in a new chat)

Paste this whole file into a new chat to resume. Last updated: end of
Phase 9 + synthetic-data detour, about to start Phase 10 (SHAP).

## What Cassie is, in one paragraph

Cassie is a decision-support platform for a fictional retailer,
Meridian Commerce Group, that turns raw order/customer data into an
answer to one question: why are customers leaving, which ones matter
most to save, how much revenue is actually at risk, and what should
the business do about it. Built as a portfolio piece for Riskgratis
Technologies (Nigerian ERP/data & analytics consulting) to demonstrate
SQL/data modeling, ML judgment (including knowing when the data
doesn't support more than a certain result), a business-rules/scenario
layer, and an LLM interface — not just a polished-looking churn
dashboard.

## Environment facts (easy to forget, costly to get wrong)

- Repo: `C:\Users\USER\Desktop\ML PORTFOLIO\cassie`
- **Cassie has its OWN `.venv`** inside that folder, on **Python 3.13**
  (not 3.14 — ipykernel/xgboost don't have 3.14 wheels yet). Separate
  from the `HORUS` project's venv — do not reuse that one.
- Postgres via `docker compose up -d db` — **host port 5442** (not the
  default 5432, which was already taken by something else on this
  machine). `.env` has `DATABASE_URL=postgresql://cassie:cassie@localhost:5442/cassie`.
- Jupyter: launch `jupyter lab` from inside the activated `.venv` so it
  picks up the right kernel automatically.

## Phase status: 10 of 22 core phases done, plus a synthetic-data detour

**Done:** 1 (repo/env/docs), 2 (data audit — done rigorously, includes
churn-window decision), 3 (DB schema, raw+staging loaded), 4 (automated
data quality checks), 5 (business exploration findings), 6 (churn
definition — folded into Phase 2's rigor pass), 7 (feature engineering
— snapshot-based), 8 (baseline logistic regression), 9 (XGBoost +
model comparison), **10 (SHAP explainability, plain-language layer)**,
**plus** a synthetic enterprise-data layer (support/marketing/costs/
segments/interventions) that was always planned but never built until
partway through this work.

**Not started:** 11 (customer value/revenue at risk — NEXT), 12
(scenario engine), 13 (FastAPI), 14 (dashboard), 15 (LLM tools), 16
(Business Analyst mode), 17 (RAG), 18 (UAT), 19 (LLM evaluation), 20
(deployment), 21 (documentation), 22 (final executive presentation).

## Key decisions already made (don't re-litigate these without reason)

- **Churn definition**: no purchase event within **180 days** of last
  purchase event. Chosen from real gap-distribution data, not assumed.
  See `docs/churn_definition.md`.
- **"A purchase" = a distinct `(customer_unique_id, purchase_timestamp)`
  purchase event, never a raw order row** — Olist's cart-splitting
  (one checkout → multiple `order_id`s) inflates raw order counts.
  Enforced via `staging.customer_purchase_events`.
- **Train/validation/test split is by customer's FIRST-APPEARANCE
  cohort**, not just by snapshot date — otherwise the same customer
  leaks across all three splits (found and fixed in Phase 7, verified
  0% overlap).
- **Final model choice: baseline logistic regression, NOT XGBoost.**
  Tried XGBoost properly (including catching and fixing an overfit
  first attempt), it converges to the same weak ceiling as the linear
  model — added complexity isn't earning its keep. See
  `docs/model_comparison_conclusion.md`.
- **Current honest model performance**: ROC-AUC ~0.55-0.65 depending on
  split — weak by conventional standards, genuinely (if modestly)
  improved by the synthetic data layer once a leakage bug in it was
  found and fixed. This is NOT a strong model — every dollar-figure
  phase downstream (11, 12) needs to carry that caveat honestly, not
  present false precision.
- **Cold-start hypothesis (do new customers predict worse?) is
  UNRESOLVED** — inconsistent direction between validation and test on
  small subgroups. Don't state it as confirmed.
- **`average_order_value` and `max_order_value` dropped from the
  model's feature set** (Phase 10) — both collapse to the exact same
  number as `lifetime_revenue` for any single-order customer, ~97% of
  the dataset, causing arbitrary/contradictory SHAP explanations for
  the majority of customers. `lifetime_revenue` kept as the one "how
  much spent" signal. See `docs/baseline_model.md`'s second audit.
- **`has_been_targeted_for_retention` stays in the model** (real,
  temporally-safe signal, currently the #1 feature by SHAP importance)
  **but is suppressed from customer-facing explanations specifically**
  (`EXCLUDE_FROM_CUSTOMER_EXPLANATION` in `scripts/explain_model.py`)
  — telling a retention team "the reason they're at risk is that we
  already flagged them as at risk" isn't useful, even though the model
  legitimately uses the signal. Global importance still shows it.

## Real bugs found and fixed this project (worth remembering the pattern)

Three separate instances of the same bug category — joining/counting
across a one-to-many relationship without pre-aggregating first, which
silently inflates numbers:
1. Cart-split orders inflating raw purchase counts (Phase 3/7)
2. `lifetime_orders` counting raw order rows instead of dedup'd events
   (Phase 7, caught via a null-rate sanity check)
3. Join fan-out in synthetic data generation (order_items × payments)

Two more, different in kind:
4. `span_days` leakage in synthetic data generation — event dates were
   sized using each customer's TRUE future `last_purchase`, leaking
   return behavior sideways through a proxy variable. Found because
   logistic regression's train performance jumped suspiciously. Fixed
   with a fixed, customer-independent event horizon.
5. **Subgroup collinearity, found via SHAP (Phase 10)** —
   `average_order_value`/`max_order_value`/`lifetime_revenue` aren't
   duplicates in general, but collapse to identical values for the
   ~97% single-order majority, causing a real customer explanation to
   show the same dollar figure increasing AND decreasing risk
   simultaneously. Not a data bug — a modeling/feature-set bug that
   only became visible by reading actual model output for real
   customers, not by inspecting the feature table.

Worth watching for both patterns again in any new phase.

## Standing preferences for this project

- Direct, honest assessments — don't spin weak results as good.
- Document failed/weak experiments, don't hide them.
- Visual/diagrammatic explanations where they help.
- **SHAP/explanations use plain, stakeholder-friendly language**, not
  raw feature names/SHAP values — e.g. "Hasn't purchased in 165 days
  (typical gap for returners is ~40 days)", not
  `days_since_last_purchase: 165`. Implemented via `FEATURE_LABELS` and
  `describe_value()` in `scripts/explain_model.py` — extend that
  dictionary, don't bypass it, if new features get added later.
- Every deliverable gets a smoke test that's actually run before
  handing it over, not just written and assumed correct.
- **Notify when this chat's context is getting full and produce an
  updated version of this file** — standing instruction, not one-off.

## File map (what exists in `docs/`, `scripts/`, `sql/`, `tests/`)

Repo skeleton, schema DDL, data-quality plan/report, churn definition,
business exploration findings, feature engineering design (+ the
cross-split-leakage and counting-bug fixes documented inline),
baseline model design + results (+ the subgroup-collinearity fix),
model comparison (round 1 overfit → round 2 regularized → conclusion,
updated again after the synthetic data detour), synthetic data
generation design (+ fan-out and span_days leakage fixes documented
inline). Scripts mirror this: `load_raw_data.py`, `run_quality_checks.py`,
`build_features.py`, `check_feature_nulls.py`, `train_baseline_model.py`,
`train_xgboost_model.py`, `generate_synthetic_data.py`,
`explain_model.py`. Tests for all of the above live in `tests/unit/`.

## Immediate next step

Phase 11: customer value + revenue at risk. This is the first phase
that turns the model's output into a dollar figure
(`customer_value × P(churn)`), which means the "this model is weak,
~0.55-0.65 ROC-AUC" caveat has to be built into how those numbers are
presented from the start — not a strong-model assumption bolted on
later. Will need `staging`/`analytics` revenue data plus the model's
predicted probabilities; does NOT use `customer_costs`/`customer_segments`
for prediction (excluded from the model, per Phase 9's leakage
decision) but SHOULD use them here, since this phase isn't a temporal
prediction task — full-history "current" values are legitimate here.
