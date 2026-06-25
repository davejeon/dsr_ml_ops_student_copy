"""FastAPI serving layer (Section 6).

Endpoints follow the manual's "essential endpoints" table:
  GET  /health   liveness
  GET  /ready    readiness (model loaded)
  POST /predict  inference
  GET  /metrics  Prometheus-style counters (stub)
"""
from __future__ import annotations

from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn.config import settings
from churn.features.build import build_features

app = FastAPI(title="churn-svc", version="0.1.0")

# Loaded once at process start — avoids per-request disk hits.
try:
    _model = joblib.load(settings.model_path)
except FileNotFoundError:
    _model = None

# Naive in-memory counters — real deployments expose prometheus_client.
_counters = {"predictions_total": 0, "errors_total": 0}


class CustomerFeatures(BaseModel):
    customer_id: int = Field(..., ge=0)
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0.0, le=1000.0)
    total_charges: float = Field(..., ge=0.0)
    contract_type: Literal["month-to-month", "one-year", "two-year"]


class Prediction(BaseModel):
    customer_id: int
    churn_probability: float
    churn_predicted: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=Prediction)
def predict(payload: CustomerFeatures) -> Prediction:
    if _model is None:
        _counters["errors_total"] += 1
        raise HTTPException(status_code=503, detail="model not loaded")

    df = pd.DataFrame([payload.model_dump()])
    # churned column not required for inference — add a dummy so the
    # shared validate() schema doesn't fail. In production, split the
    # schema into "input" vs "training" variants.
    df["churned"] = 0

    features = build_features(df)
    proba = float(_model.predict_proba(features)[0, 1])
    _counters["predictions_total"] += 1
    return Prediction(
        customer_id=payload.customer_id,
        churn_probability=proba,
        churn_predicted=int(proba >= 0.5),
    )


@app.get("/metrics")
def metrics() -> str:
    # Minimal Prometheus exposition format.
    lines = [f"{k} {v}" for k, v in _counters.items()]
    return "\n".join(lines)
