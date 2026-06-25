"""Feature engineering. SHARED by training and serving — this is the
single biggest defence against train/serve skew (Section 5)."""
from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_month_to_month",
    "contract_one_year",
    "contract_two_year",
    "avg_charge_per_month",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["tenure_months"] = df["tenure_months"].astype(float)
    out["monthly_charges"] = df["monthly_charges"].astype(float)
    out["total_charges"] = df["total_charges"].astype(float)

    # one-hot contract
    out["contract_month_to_month"] = (df["contract_type"] == "month-to-month").astype(int)
    out["contract_one_year"] = (df["contract_type"] == "one-year").astype(int)
    out["contract_two_year"] = (df["contract_type"] == "two-year").astype(int)

    # derived feature — example of "feature engineering"
    out["avg_charge_per_month"] = (
        df["total_charges"] / df["tenure_months"].clip(lower=1)
    ).astype(float)

    return out[FEATURE_COLUMNS]
