"""Central config — read from env vars, never hard-coded in code."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    seed: int = int(os.getenv("CHURN_SEED", "42"))
    raw_data_path: Path = Path(os.getenv("CHURN_RAW_DATA", "data/raw/customers.csv"))
    model_path: Path = Path(os.getenv("CHURN_MODEL_PATH", "model.joblib"))
    mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT", "churn-baseline")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    # CI gate — train.py / test_model.py both read this.
    min_auc: float = float(os.getenv("CHURN_MIN_AUC", "0.80"))


settings = Settings()
