"""Model-quality gate — the test that gives CI/CT teeth.

In a real repo this would load the latest trained model and a held-out
golden dataset, then assert AUC and per-segment fairness. Here we
just sketch the structure.
"""
import json
import os

import pytest

from churn.config import settings


@pytest.mark.skipif(
    not os.path.exists("metrics.json"),
    reason="metrics.json not produced — run `make train` first",
)
def test_holdout_auc_meets_threshold():
    with open("metrics.json") as f:
        metrics = json.load(f)
    assert metrics["auc"] >= settings.min_auc, (
        f"AUC regression: {metrics['auc']:.3f} < {settings.min_auc:.3f}"
    )


@pytest.mark.skipif(
    not os.path.exists("metrics.json"),
    reason="metrics.json not produced — run `make train` first",
)
def test_accuracy_sanity():
    with open("metrics.json") as f:
        metrics = json.load(f)
    # Sanity floor — guards against an "all zeros" model on balanced data.
    assert metrics["accuracy"] >= 0.6
