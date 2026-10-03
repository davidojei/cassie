"""
tests/unit/test_scenario_engine.py

Unit tests for scripts/scenario_engine.py. Uses a small synthetic
fixture DataFrame rather than the real customer_revenue_at_risk.csv --
unit tests shouldn't depend on production data (it changes every time
the pipeline reruns), and a hand-built fixture lets every edge case
(zero-probability customer, missing segment, duplicate probability
values) be constructed deliberately instead of hoping the real data
happens to contain it.

This also gives real test coverage to the `top_n` and `segment`
selection paths in campaign_roi, which had only been exercised
manually via `risk_tier` before this suite existed.

Run with: pytest tests/unit/test_scenario_engine.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from scenario_engine import (  # noqa: E402
    RETENTION_EFFECT_PRESETS,
    budget_allocation,
    campaign_roi,
    compute_expected_revenue_saved,
    resolve_retention_effect,
    risk_tier_whatif,
    select_targets,
)


@pytest.fixture
def fixture_df():
    return pd.DataFrame({
        "customer_unique_id": [f"cust_{i}" for i in range(10)],
        "customer_value": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "predicted_churn_probability": [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9],
        "revenue_at_risk": [10, 40, 90, 160, 250, 300, 420, 560, 720, 900],
        "segment": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "risk_tier": ["low", "low", "low", "medium", "medium", "medium", "medium", "high", "high", "high"],
    })


# --- resolve_retention_effect ---

def test_resolve_retention_effect_preset():
    assert resolve_retention_effect("moderate") == RETENTION_EFFECT_PRESETS["moderate"]


def test_resolve_retention_effect_numeric_string():
    assert resolve_retention_effect("0.27") == pytest.approx(0.27)


def test_resolve_retention_effect_rejects_out_of_range():
    with pytest.raises(ValueError):
        resolve_retention_effect("1.5")
    with pytest.raises(ValueError):
        resolve_retention_effect("0")


def test_resolve_retention_effect_rejects_garbage():
    with pytest.raises(ValueError):
        resolve_retention_effect("not-a-number")


# --- select_targets ---

def test_select_targets_top_n(fixture_df):
    result = select_targets(fixture_df, "top_n", n=3)
    assert len(result) == 3
    # top 3 by revenue_at_risk: cust_9 (900), cust_8 (720), cust_7 (560)
    assert set(result["customer_unique_id"]) == {"cust_9", "cust_8", "cust_7"}


def test_select_targets_risk_tier(fixture_df):
    result = select_targets(fixture_df, "risk_tier", tier="high")
    assert len(result) == 3
    assert (result["risk_tier"] == "high").all()


def test_select_targets_risk_tier_invalid_raises(fixture_df):
    with pytest.raises(ValueError):
        select_targets(fixture_df, "risk_tier", tier="nonexistent")


def test_select_targets_segment(fixture_df):
    result = select_targets(fixture_df, "segment", segment="A")
    assert len(result) == 4
    assert (result["segment"] == "A").all()


def test_select_targets_segment_invalid_raises(fixture_df):
    with pytest.raises(ValueError):
        select_targets(fixture_df, "segment", segment="nonexistent")


def test_select_targets_unknown_type_raises(fixture_df):
    with pytest.raises(ValueError):
        select_targets(fixture_df, "bogus_type")


# --- compute_expected_revenue_saved ---

def test_compute_expected_revenue_saved_formula(fixture_df):
    out = compute_expected_revenue_saved(fixture_df, 0.2)
    expected = fixture_df["customer_value"] * fixture_df["predicted_churn_probability"] * 0.2
    pd.testing.assert_series_equal(out["expected_revenue_saved"], expected, check_names=False)


def test_compute_expected_revenue_saved_does_not_mutate_input(fixture_df):
    original_cols = set(fixture_df.columns)
    compute_expected_revenue_saved(fixture_df, 0.2)
    assert set(fixture_df.columns) == original_cols


# --- campaign_roi: all three selection types ---

def test_campaign_roi_top_n(fixture_df):
    summary, targeted = campaign_roi(fixture_df, "top_n", cost_per_customer=10, retention_effect_pct=0.2, n=3)
    assert summary["n_targeted"] == 3
    assert summary["total_campaign_cost"] == 30
    assert summary["total_expected_revenue_saved"] > 0
    assert np.isfinite(summary["net_expected_benefit"])
    assert "model_caveat" in summary and "retention_effect_caveat" in summary


def test_campaign_roi_risk_tier(fixture_df):
    summary, targeted = campaign_roi(fixture_df, "risk_tier", cost_per_customer=10, retention_effect_pct=0.2, tier="high")
    assert summary["n_targeted"] == 3
    assert (targeted["risk_tier"] == "high").all()


def test_campaign_roi_segment(fixture_df):
    summary, targeted = campaign_roi(fixture_df, "segment", cost_per_customer=10, retention_effect_pct=0.2, segment="B")
    assert summary["n_targeted"] == 3
    assert (targeted["segment"] == "B").all()


def test_campaign_roi_rejects_zero_cost(fixture_df):
    with pytest.raises(ValueError):
        campaign_roi(fixture_df, "top_n", cost_per_customer=0, retention_effect_pct=0.2, n=3)


def test_campaign_roi_rejects_negative_cost(fixture_df):
    with pytest.raises(ValueError):
        campaign_roi(fixture_df, "top_n", cost_per_customer=-5, retention_effect_pct=0.2, n=3)


def test_campaign_roi_known_values(fixture_df):
    # cust_9: value=1000, prob=0.9, effect=0.2 -> expected_revenue_saved = 180
    summary, targeted = campaign_roi(fixture_df, "top_n", cost_per_customer=10, retention_effect_pct=0.2, n=1)
    assert targeted.iloc[0]["customer_unique_id"] == "cust_9"
    assert targeted.iloc[0]["expected_revenue_saved"] == pytest.approx(1000 * 0.9 * 0.2)
    assert summary["total_expected_revenue_saved"] == pytest.approx(180.0)
    assert summary["total_campaign_cost"] == 10
    assert summary["net_expected_benefit"] == pytest.approx(170.0)


# --- budget_allocation ---

def test_budget_allocation_never_exceeds_budget(fixture_df):
    summary, selected, curve = budget_allocation(fixture_df, total_budget=35, cost_per_customer=10, retention_effect_pct=0.2)
    assert summary["total_spent"] <= 35
    assert summary["n_targeted"] == 3  # floor(35/10)


def test_budget_allocation_rejects_zero_cost(fixture_df):
    with pytest.raises(ValueError):
        budget_allocation(fixture_df, total_budget=100, cost_per_customer=0, retention_effect_pct=0.2)


def test_budget_allocation_selects_highest_value_first(fixture_df):
    # With flat cost_per_customer, ranking by expected_revenue_saved_per_dollar
    # is equivalent to ranking by expected_revenue_saved -- highest-value
    # customers (highest customer_value x probability) should be selected first.
    summary, selected, curve = budget_allocation(fixture_df, total_budget=10, cost_per_customer=10, retention_effect_pct=0.2)
    assert selected.iloc[0]["customer_unique_id"] == "cust_9"


def test_budget_allocation_curve_is_monotonic_nondecreasing(fixture_df):
    summary, selected, curve = budget_allocation(fixture_df, total_budget=50, cost_per_customer=10, retention_effect_pct=0.2)
    # More budget should never produce less total revenue saved.
    assert curve["total_revenue_saved"].is_monotonic_increasing or curve["total_revenue_saved"].duplicated().any() is False


def test_budget_allocation_zero_budget_targets_nobody(fixture_df):
    summary, selected, curve = budget_allocation(fixture_df, total_budget=5, cost_per_customer=10, retention_effect_pct=0.2)
    assert summary["n_targeted"] == 0
    assert summary["total_spent"] == 0


# --- risk_tier_whatif ---

def test_risk_tier_whatif_absolute(fixture_df):
    out, summary, movement = risk_tier_whatif(fixture_df, cutoffs=[0, 0.4, 0.7, 1.0], labels=["low", "medium", "high"], method="absolute")
    assert set(out["risk_tier_scenario"].dropna().unique()) <= {"low", "medium", "high"}
    assert len(summary) <= 3


def test_risk_tier_whatif_quantile(fixture_df):
    out, summary, movement = risk_tier_whatif(fixture_df, cutoffs=[0, 0.5, 0.8, 1.0], labels=["low", "medium", "high"], method="quantile")
    assert "risk_tier_scenario" in out.columns


def test_risk_tier_whatif_mismatched_labels_raises(fixture_df):
    with pytest.raises(ValueError):
        risk_tier_whatif(fixture_df, cutoffs=[0, 0.5, 1.0], labels=["low", "medium", "high"], method="absolute")


def test_risk_tier_whatif_unknown_method_raises(fixture_df):
    with pytest.raises(ValueError):
        risk_tier_whatif(fixture_df, cutoffs=[0, 0.5, 1.0], labels=["low", "high"], method="bogus")


def test_risk_tier_whatif_duplicate_quantile_edges_raises_clear_error():
    # Deliberately constructed so repeated probability values collide at a
    # quantile cutoff -- this is the exact scenario the duplicates="drop"
    # handling + wrapped error message exists for.
    df = pd.DataFrame({
        "customer_unique_id": [f"c{i}" for i in range(6)],
        "customer_value": [100] * 6,
        "predicted_churn_probability": [0.5, 0.5, 0.5, 0.5, 0.5, 0.9],
        "revenue_at_risk": [50] * 6,
    })
    # Asking for 4 bins out of data that's 5/6 identical values can't
    # produce 4 distinct quantile edges -- should raise a clear, wrapped
    # error rather than an opaque pandas traceback.
    with pytest.raises(ValueError, match="quantile bins"):
        risk_tier_whatif(df, cutoffs=[0, 0.25, 0.5, 0.75, 1.0], labels=["a", "b", "c", "d"], method="quantile")


def test_risk_tier_whatif_movement_table_present_when_risk_tier_exists(fixture_df):
    out, summary, movement = risk_tier_whatif(fixture_df, cutoffs=[0, 0.4, 0.7, 1.0], labels=["low", "medium", "high"], method="absolute")
    assert movement is not None


def test_risk_tier_whatif_movement_table_absent_without_risk_tier_column(fixture_df):
    df_no_tier = fixture_df.drop(columns=["risk_tier"])
    out, summary, movement = risk_tier_whatif(df_no_tier, cutoffs=[0, 0.4, 0.7, 1.0], labels=["low", "medium", "high"], method="absolute")
    assert movement is None
