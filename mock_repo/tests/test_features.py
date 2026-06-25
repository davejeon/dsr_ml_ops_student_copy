"""Unit tests — fast, no I/O, no model."""
import pandas as pd

from churn.features.build import FEATURE_COLUMNS, build_features


def test_build_features_returns_expected_columns():
    df = pd.DataFrame([
        {
            "customer_id": 1,
            "tenure_months": 12,
            "monthly_charges": 50.0,
            "total_charges": 600.0,
            "contract_type": "one-year",
            "churned": 0,
        }
    ])
    out = build_features(df)
    assert list(out.columns) == FEATURE_COLUMNS
    assert out.loc[0, "contract_one_year"] == 1
    assert out.loc[0, "avg_charge_per_month"] == 50.0


def test_avg_charge_handles_zero_tenure():
    df = pd.DataFrame([
        {
            "customer_id": 2,
            "tenure_months": 0,
            "monthly_charges": 70.0,
            "total_charges": 0.0,
            "contract_type": "month-to-month",
            "churned": 1,
        }
    ])
    out = build_features(df)
    assert out.loc[0, "avg_charge_per_month"] == 0.0  # no divide-by-zero
