"""Schema + distribution validation (Section 5 of the manual).

Runs in two places:
  * before training (catches upstream schema changes)
  * before serving  (catches request payloads that don't look like training data)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, Check


CustomerSchema = pa.DataFrameSchema(
    {
        "customer_id": Column(int, unique=True),
        "tenure_months": Column(int, Check.in_range(0, 120)),
        "monthly_charges": Column(float, Check.in_range(0.0, 1000.0)),
        "total_charges": Column(float, Check.ge(0.0)),
        "contract_type": Column(str, Check.isin(["month-to-month", "one-year", "two-year"])),
        "churned": Column(int, Check.isin([0, 1])),
    },
    strict=True,
    coerce=True,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Raise SchemaError on bad data; return the (coerced) frame on success."""
    return CustomerSchema.validate(df, lazy=True)


def main(path: str) -> None:
    df = pd.read_csv(path)
    validate(df)
    print(f"OK — {len(df)} rows passed schema validation.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/customers.csv")
