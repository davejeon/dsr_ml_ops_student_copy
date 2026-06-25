"""Data-layer tests — schema + simple distribution checks."""
import pandas as pd
import pandera as pa
import pytest

from churn.data.validate import validate


def _good_row(**overrides):
    base = {
        "customer_id": 1,
        "tenure_months": 12,
        "monthly_charges": 50.0,
        "total_charges": 600.0,
        "contract_type": "one-year",
        "churned": 0,
    }
    base.update(overrides)
    return base


def test_validate_accepts_good_data():
    df = pd.DataFrame([_good_row(customer_id=i) for i in range(5)])
    validate(df)  # should not raise


def test_validate_rejects_unknown_contract():
    df = pd.DataFrame([_good_row(contract_type="lifetime")])
    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)


def test_validate_rejects_negative_charges():
    df = pd.DataFrame([_good_row(total_charges=-1.0)])
    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)
