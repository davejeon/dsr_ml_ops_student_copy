"""
test_service.py — Basic integration tests for the iris FastAPI service.

Covers the three things worth testing on any ML service:
  1. The service is alive (/health).
  2. A known-good request returns the right prediction (/predict).
  3. A bad request is rejected cleanly (422, not a 500 crash).

Prerequisites:
  Start the service in a separate terminal before running these tests:
    uvicorn service:app --app-dir day2/Session_1 --port 8000

  Then run:
    pytest day2/Session_3/test_service.py -v

If the service is not running, all tests in this file are SKIPPED.
"""

import pytest
import requests

BASE_URL = "http://localhost:8000"
TIMEOUT  = 2.0


def _service_is_running() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=TIMEOUT).status_code == 200
    except requests.exceptions.ConnectionError:
        return False


pytestmark = pytest.mark.skipif(
    not _service_is_running(),
    reason=(
        "Service not running on localhost:8000. "
        "Start it with: uvicorn service:app --app-dir day2/Session_1 --port 8000"
    ),
)


def test_health_check():
    """The service must respond 200 on /health — the most basic liveness check."""
    r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_correct_class_for_setosa():
    """
    A canonical setosa measurement must come back as Iris-setosa.
    This validates the full stack: HTTP → Pydantic → model → JSON response.
    """
    payload = {
        "septal_length": 5.1,
        "sepal_width":   3.5,
        "petal_length":  1.4,
        "petal_width":   0.2,
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
    assert r.status_code == 200
    assert r.json()["predicted_class"] == "Iris-setosa"


def test_invalid_input_is_rejected_with_422():
    """
    A negative feature value should be rejected by Pydantic validation
    before the model ever sees it. The response must be 422, not a 500 crash.
    """
    bad_payload = {
        "septal_length": -1.0,   # invalid: must be > 0
        "sepal_width":   3.5,
        "petal_length":  1.4,
        "petal_width":   0.2,
    }
    r = requests.post(f"{BASE_URL}/predict", json=bad_payload, timeout=TIMEOUT)
    assert r.status_code == 422, (
        f"Expected 422 for invalid input, got {r.status_code}. "
        "Pydantic validation may not be working."
    )
