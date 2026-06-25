"""Integration smoke tests for the FastAPI service."""
from fastapi.testclient import TestClient

from churn.inference import service


client = TestClient(service.app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_probability_when_model_loaded():
    if service._model is None:
        # Mirrors the manual: /ready should fail when model isn't loaded.
        r = client.get("/ready")
        assert r.status_code == 503
        return

    payload = {
        "customer_id": 42,
        "tenure_months": 6,
        "monthly_charges": 70.0,
        "total_charges": 420.0,
        "contract_type": "month-to-month",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_predicted"] in (0, 1)


def test_predict_rejects_invalid_contract():
    payload = {
        "customer_id": 42,
        "tenure_months": 6,
        "monthly_charges": 70.0,
        "total_charges": 420.0,
        "contract_type": "lifetime",  # invalid
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422  # pydantic validation, before model is hit
