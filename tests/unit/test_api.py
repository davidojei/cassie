"""
tests/unit/test_api.py

Tests for api/main.py. Uses a small synthetic fixture CSV written to a
temp directory and monkeypatches the API's data path to it -- same
philosophy as Phase 12's tests, not the real production CSV.

IMPORTANT: TestClient must be used as a context manager
(`with TestClient(app) as client:`) for FastAPI's lifespan startup
event to actually run and load the data. Without `with`, /health will
report data_loaded=False even when the fixture file exists -- this
bit the first manual test run of this API and is easy to repeat by
accident, hence this note.

Run with: pytest tests/unit/test_api.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Build a small fixture CSV with the same columns score_customer_value.py produces
    n = 30
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "customer_unique_id": [f"cust_{i}" for i in range(n)],
        "customer_value": rng.uniform(50, 2000, n).round(2),
        "predicted_churn_probability": rng.uniform(0.05, 0.95, n).round(4),
    })
    df["revenue_at_risk"] = df["customer_value"] * df["predicted_churn_probability"]
    df["segment"] = rng.choice(["standard", "low_value", "high_value"], n)
    df["risk_tier"] = pd.qcut(df["predicted_churn_probability"], q=[0, 0.5, 0.8, 1.0], labels=["low", "medium", "high"])
    df["caveat"] = "test caveat"

    fixture_path = tmp_path / "customer_revenue_at_risk.csv"
    df.to_csv(fixture_path, index=False)

    import api.main as api_main
    monkeypatch.setattr(api_main, "REVENUE_AT_RISK_PATH", fixture_path)

    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture
def client_no_data(tmp_path, monkeypatch):
    """A client pointed at a path with no CSV -- tests the 503 path."""
    import api.main as api_main
    monkeypatch.setattr(api_main, "REVENUE_AT_RISK_PATH", tmp_path / "does_not_exist.csv")
    with TestClient(api_main.app) as c:
        yield c


# --- health ---

def test_health_reports_loaded_data(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["data_loaded"] is True
    assert body["n_customers"] == 30
    assert "model_caveat" in body


def test_health_reports_missing_data(client_no_data):
    r = client_no_data.get("/health")
    assert r.status_code == 200
    assert r.json()["data_loaded"] is False


def test_data_dependent_endpoint_503_when_no_data(client_no_data):
    r = client_no_data.get("/customers")
    assert r.status_code == 503
    assert "score_customer_value.py" in r.json()["detail"]


# --- customers ---

def test_list_customers_pagination(client):
    r = client.get("/customers?limit=5&offset=0")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 30
    assert len(body["customers"]) == 5


def test_list_customers_filter_by_risk_tier(client):
    r = client.get("/customers?risk_tier=high&limit=100")
    assert r.status_code == 200
    body = r.json()
    assert all(c["risk_tier"] == "high" for c in body["customers"])


def test_list_customers_filter_by_min_revenue(client):
    r = client.get("/customers?min_revenue_at_risk=1000&limit=100")
    assert r.status_code == 200
    body = r.json()
    assert all(c["revenue_at_risk"] >= 1000 for c in body["customers"])


def test_get_customer_found(client):
    r = client.get("/customers/cust_0")
    assert r.status_code == 200
    assert r.json()["customer_unique_id"] == "cust_0"
    assert "model_caveat" in r.json()


def test_get_customer_not_found(client):
    r = client.get("/customers/does_not_exist")
    assert r.status_code == 404


# --- campaign-roi ---

def test_campaign_roi_top_n(client):
    r = client.post("/scenarios/campaign-roi", json={
        "selection_type": "top_n", "n": 5, "cost_per_customer": 10, "effect": "moderate",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["n_targeted"] == 5
    assert "model_caveat" in body and "retention_effect_caveat" in body


def test_campaign_roi_risk_tier(client):
    r = client.post("/scenarios/campaign-roi", json={
        "selection_type": "risk_tier", "tier": "high", "cost_per_customer": 10, "effect": "moderate",
    })
    assert r.status_code == 200


def test_campaign_roi_segment(client):
    r = client.post("/scenarios/campaign-roi", json={
        "selection_type": "segment", "segment": "high_value", "cost_per_customer": 10, "effect": "moderate",
    })
    assert r.status_code == 200


def test_campaign_roi_missing_required_field_returns_400(client):
    # top_n selected but n omitted -- a business-logic error raised inside
    # scenario_engine.py, not a pydantic schema error, so this is 400.
    r = client.post("/scenarios/campaign-roi", json={
        "selection_type": "top_n", "cost_per_customer": 10, "effect": "moderate",
    })
    assert r.status_code == 400


def test_campaign_roi_invalid_tier_returns_400(client):
    r = client.post("/scenarios/campaign-roi", json={
        "selection_type": "risk_tier", "tier": "bogus", "cost_per_customer": 10, "effect": "moderate",
    })
    assert r.status_code == 400


def test_campaign_roi_zero_cost_returns_422(client):
    # cost_per_customer <= 0 is caught by pydantic's Field(gt=0) before the
    # handler runs at all -- this is 422, not 400. Different failure point
    # from the business-logic errors above.
    r = client.post("/scenarios/campaign-roi", json={
        "selection_type": "top_n", "n": 5, "cost_per_customer": 0, "effect": "moderate",
    })
    assert r.status_code == 422


def test_campaign_roi_invalid_effect_string_returns_400(client):
    r = client.post("/scenarios/campaign-roi", json={
        "selection_type": "top_n", "n": 5, "cost_per_customer": 10, "effect": "not-a-number",
    })
    assert r.status_code == 400


# --- budget-allocation ---

def test_budget_allocation_success(client):
    r = client.post("/scenarios/budget-allocation", json={
        "total_budget": 500, "cost_per_customer": 10, "effect": "conservative",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["total_spent"] <= 500
    assert "marginal_value_curve" in body


def test_budget_allocation_zero_budget_returns_422(client):
    r = client.post("/scenarios/budget-allocation", json={
        "total_budget": 0, "cost_per_customer": 10, "effect": "conservative",
    })
    assert r.status_code == 422


# --- risk-tier-whatif ---

def test_risk_tier_whatif_absolute(client):
    r = client.post("/scenarios/risk-tier-whatif", json={
        "cutoffs": [0, 0.4, 0.7, 1.0], "labels": ["low", "medium", "high"], "method": "absolute",
    })
    assert r.status_code == 200
    body = r.json()
    assert "tier_summary" in body
    assert "movement_vs_current_tiers" in body


def test_risk_tier_whatif_mismatched_labels_returns_400(client):
    r = client.post("/scenarios/risk-tier-whatif", json={
        "cutoffs": [0, 0.5, 1.0], "labels": ["low", "medium", "high"], "method": "absolute",
    })
    assert r.status_code == 400


# --- admin ---

def test_reload_data(client):
    r = client.post("/admin/reload-data")
    assert r.status_code == 200
    assert r.json()["data_loaded"] is True
