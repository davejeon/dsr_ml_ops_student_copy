"""
test_service.py — Integration tests for the iris FastAPI service.

These tests hit the RUNNING service over HTTP — they require the server
to already be up. They are NOT unit tests; they validate the full stack:
Pydantic validation → model inference → JSON serialisation → HTTP response.

Prerequisites:
  1. Run day2/Session_1/train.py to generate the .joblib files.
  2. Start the service in a separate terminal:
       uvicorn service:app --app-dir day2/Session_1 --port 8000
  3. Then run these tests:
       pytest day2/Session_3/test_service.py -v

If the service is not running, every test in this file is SKIPPED
(not failed) — the conftest skip logic handles that cleanly.

Latency SLO: p99 response time for /predict should be under 100 ms on
any modern laptop. We test a single call — for proper SLO testing use
Locust or k6.
"""

import time

import pytest
import requests

BASE_URL = "http://localhost:8000"
TIMEOUT  = 2.0   # seconds — if the service doesn't respond in 2 s, fail


def _service_is_running() -> bool:
    """Return True if the service responds on /health within TIMEOUT."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# ── Skip the entire module if the service is not up ───────────────────────────
# Using a module-level mark means each test shows as SKIPPED with a reason,
# rather than FAILED — distinguishing "test failed" from "env not ready".
pytestmark = pytest.mark.skipif(
    not _service_is_running(),
    reason=(
        "Service is not running on localhost:8000. "
        "Start it with: uvicorn service:app --app-dir day2/Session_1 --port 8000"
    ),
)

# ── Canonical payloads ─────────────────────────────────────────────────────────
SETOSA_PAYLOAD = {
    "septal_length": 5.1,
    "sepal_width":   3.5,
    "petal_length":  1.4,
    "petal_width":   0.2,
}
VIRGINICA_PAYLOAD = {
    "septal_length": 6.7,
    "sepal_width":   3.0,
    "petal_length":  5.2,
    "petal_width":   2.3,
}
VERSICOLOR_PAYLOAD = {
    "septal_length": 5.7,
    "sepal_width":   2.8,
    "petal_length":  4.1,
    "petal_width":   1.3,
}


# ── 1. Smoke / health tests ────────────────────────────────────────────────────

class TestHealthEndpoints:
    """
    Minimal checks that confirm the process is alive and the model is loaded.
    These are the same checks a load balancer or Kubernetes probe would make.
    """

    def test_health_returns_200(self):
        r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_health_returns_ok_status(self):
        r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert r.json() == {"status": "ok"}

    def test_ready_returns_200_when_model_loaded(self):
        r = requests.get(f"{BASE_URL}/ready", timeout=TIMEOUT)
        assert r.status_code == 200, (
            f"Service returned {r.status_code} — model may not have loaded yet. "
            f"Response: {r.text}"
        )

    def test_ready_response_contains_status_and_uptime(self):
        r = requests.get(f"{BASE_URL}/ready", timeout=TIMEOUT)
        body = r.json()
        assert body["status"] == "ready"
        assert "model_uptime_seconds" in body
        assert body["model_uptime_seconds"] >= 0


# ── 2. Inference correctness tests ────────────────────────────────────────────

class TestPredictEndpoint:
    """
    The /predict endpoint must return the right structure and sensible values
    for known iris samples.
    """

    def test_predict_returns_200_with_valid_payload(self):
        r = requests.post(f"{BASE_URL}/predict", json=SETOSA_PAYLOAD, timeout=TIMEOUT)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    def test_predict_response_has_required_fields(self):
        r = requests.post(f"{BASE_URL}/predict", json=SETOSA_PAYLOAD, timeout=TIMEOUT)
        body = r.json()
        assert "predicted_class" in body
        assert "probabilities"   in body
        assert "model_version"   in body

    def test_predict_probabilities_contain_all_three_classes(self):
        r    = requests.post(f"{BASE_URL}/predict", json=SETOSA_PAYLOAD, timeout=TIMEOUT)
        probs = r.json()["probabilities"]
        assert "Iris-setosa"     in probs
        assert "Iris-versicolor" in probs
        assert "Iris-virginica"  in probs

    def test_predict_probabilities_sum_to_one(self):
        r     = requests.post(f"{BASE_URL}/predict", json=SETOSA_PAYLOAD, timeout=TIMEOUT)
        probs = r.json()["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.001, f"Probabilities sum to {total}, expected 1.0"

    def test_canonical_setosa_is_predicted_correctly(self):
        r = requests.post(f"{BASE_URL}/predict", json=SETOSA_PAYLOAD, timeout=TIMEOUT)
        assert r.json()["predicted_class"] == "Iris-setosa", (
            f"Expected Iris-setosa, got {r.json()['predicted_class']}"
        )

    def test_canonical_virginica_is_predicted_correctly(self):
        r = requests.post(f"{BASE_URL}/predict", json=VIRGINICA_PAYLOAD, timeout=TIMEOUT)
        assert r.json()["predicted_class"] == "Iris-virginica", (
            f"Expected Iris-virginica, got {r.json()['predicted_class']}"
        )

    def test_canonical_versicolor_is_predicted_correctly(self):
        r = requests.post(f"{BASE_URL}/predict", json=VERSICOLOR_PAYLOAD, timeout=TIMEOUT)
        assert r.json()["predicted_class"] == "Iris-versicolor", (
            f"Expected Iris-versicolor, got {r.json()['predicted_class']}"
        )

    def test_model_version_is_present(self):
        r = requests.post(f"{BASE_URL}/predict", json=SETOSA_PAYLOAD, timeout=TIMEOUT)
        version = r.json().get("model_version")
        assert version is not None and version != "", (
            "model_version field is missing or empty — needed for audit trails"
        )


# ── 3. Latency tests ───────────────────────────────────────────────────────────

class TestLatency:
    """
    Single-request latency SLO. The model is loaded in memory; inference
    should be near-instant on a laptop for a 4-feature logistic regression.

    100 ms is generous — a stricter production SLO might be 20 ms p99.
    We use a single call here; for load testing use Locust or k6.
    """

    SLO_MS = 100

    def test_predict_responds_within_slo(self):
        start = time.perf_counter()
        requests.post(f"{BASE_URL}/predict", json=SETOSA_PAYLOAD, timeout=TIMEOUT)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < self.SLO_MS, (
            f"Response took {elapsed_ms:.1f} ms — exceeds {self.SLO_MS} ms SLO. "
            "Check for model loading on request (should load at startup only)."
        )


# ── 4. Input validation tests ──────────────────────────────────────────────────

class TestInputValidation:
    """
    Pydantic + FastAPI reject bad inputs before they reach the model.
    These tests confirm that the validation layer is working correctly —
    not that the model handles bad inputs gracefully.
    """

    def test_negative_feature_value_returns_422(self):
        bad = {**SETOSA_PAYLOAD, "septal_length": -1.0}
        r   = requests.post(f"{BASE_URL}/predict", json=bad, timeout=TIMEOUT)
        assert r.status_code == 422, (
            f"Expected 422 for negative feature, got {r.status_code}"
        )

    def test_zero_feature_value_returns_422(self):
        bad = {**SETOSA_PAYLOAD, "petal_width": 0.0}
        r   = requests.post(f"{BASE_URL}/predict", json=bad, timeout=TIMEOUT)
        assert r.status_code == 422

    def test_feature_above_max_returns_422(self):
        bad = {**SETOSA_PAYLOAD, "sepal_width": 25.0}   # max is 20
        r   = requests.post(f"{BASE_URL}/predict", json=bad, timeout=TIMEOUT)
        assert r.status_code == 422

    def test_missing_field_returns_422(self):
        incomplete = {
            "septal_length": 5.1,
            "sepal_width":   3.5,
            # petal_length and petal_width missing
        }
        r = requests.post(f"{BASE_URL}/predict", json=incomplete, timeout=TIMEOUT)
        assert r.status_code == 422

    def test_string_feature_value_returns_422(self):
        bad = {**SETOSA_PAYLOAD, "petal_length": "long"}
        r   = requests.post(f"{BASE_URL}/predict", json=bad, timeout=TIMEOUT)
        assert r.status_code == 422

    def test_empty_body_returns_422(self):
        r = requests.post(f"{BASE_URL}/predict", json={}, timeout=TIMEOUT)
        assert r.status_code == 422

    def test_422_response_contains_validation_detail(self):
        """FastAPI's 422 response should include a 'detail' field explaining the error."""
        bad = {**SETOSA_PAYLOAD, "petal_width": -5.0}
        r   = requests.post(f"{BASE_URL}/predict", json=bad, timeout=TIMEOUT)
        assert "detail" in r.json(), "422 response should have a 'detail' field"


# ── 5. Helper endpoint tests ───────────────────────────────────────────────────

class TestHelperEndpoints:

    def test_example_endpoint_returns_200(self):
        r = requests.get(f"{BASE_URL}/predict/example", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_example_endpoint_contains_all_feature_keys(self):
        r    = requests.get(f"{BASE_URL}/predict/example", timeout=TIMEOUT)
        body = r.json()
        example = body.get("example_request", body)
        for key in ["septal_length", "sepal_width", "petal_length", "petal_width"]:
            assert key in example, f"Key '{key}' missing from /predict/example response"

    def test_docs_endpoint_returns_200(self):
        """Swagger UI must be reachable — useful for developer experience smoke tests."""
        r = requests.get(f"{BASE_URL}/docs", timeout=TIMEOUT)
        assert r.status_code == 200
