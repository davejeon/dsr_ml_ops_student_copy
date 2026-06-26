"""
service.py — FastAPI inference service for the iris classifier.

Run:
    uvicorn service:app --reload --port 8000

Then try:
    curl http://localhost:8000/health
    curl http://localhost:8000/ready

    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"septal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

Or open the auto-generated docs at:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc

You must run train.py first to generate iris_classifier.joblib and
label_encoder.joblib in this directory.
"""

import pathlib
import time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Paths 
HERE = pathlib.Path(__file__).resolve().parent
MODEL_PATH   = HERE / "iris_classifier.joblib"
ENCODER_PATH = HERE / "label_encoder.joblib"

FEATURE_COLS = ["septal_length", "sepal_width", "petal_length", "petal_width"]

# Application state 
# Stored at module level so all requests share the same loaded objects.
# Loading on first request would add latency; loading at startup is standard.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release on shutdown."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    state["model"]      = joblib.load(MODEL_PATH)
    state["encoder"]    = joblib.load(ENCODER_PATH)
    state["loaded_at"]  = time.time()
    print(f"Model loaded from {MODEL_PATH}")
    yield
    state.clear()
    print("Model unloaded.")


app = FastAPI(
    title="Iris Classifier API",
    description=(
        "Classifies iris flowers into Setosa / Versicolor / Virginica "
        "given four measurements. Trained on the classic iris dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Request / response schemas 

class IrisFeatures(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "septal_length": 5.1,
        "sepal_width":   3.5,
        "petal_length":  1.4,
        "petal_width":   0.2,
    }}}

    septal_length: float = Field(..., gt=0, le=20, description="Sepal length in cm")
    sepal_width:   float = Field(..., gt=0, le=20, description="Sepal width in cm")
    petal_length:  float = Field(..., gt=0, le=20, description="Petal length in cm")
    petal_width:   float = Field(..., gt=0, le=20, description="Petal width in cm")


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    model_version: str = "1.0.0"


#  Endpoints

@app.get("/health", tags=["ops"],
         summary="Liveness — is the process alive?")
def health():
    """Returns 200 as long as the process is running."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"],
         summary="Readiness — is the model loaded?")
def ready():
    """Returns 200 only once the model has been loaded into memory.
    A load balancer or Kubernetes readiness probe should call this endpoint
    before routing traffic to the pod.
    """
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    uptime = round(time.time() - state["loaded_at"], 1)
    return {"status": "ready", "model_uptime_seconds": uptime}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"],
          summary="Classify an iris flower")
def predict(features: IrisFeatures):
    """
    Accepts four measurements and returns:
    - **predicted_class**: the most likely species name
    - **probabilities**: per-class probability for all three species

    All four feature values must be positive floats in the range (0, 20].
    Pydantic validates this automatically before the model is called.
    """
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([[
        features.septal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    model   = state["model"]
    encoder = state["encoder"]

    proba        = model.predict_proba(X)[0]           # shape: (3,)
    class_idx    = int(np.argmax(proba))
    class_name   = encoder.inverse_transform([class_idx])[0]
    probabilities = {
        str(cls): round(float(p), 4)
        for cls, p in zip(encoder.classes_, proba)
    }

    return PredictionResponse(
        predicted_class=class_name,
        probabilities=probabilities,
    )


@app.get("/predict/example", tags=["inference"],
         summary="Show an example request body")
def predict_example():
    """Returns a sample payload you can copy-paste into POST /predict."""
    return {
        "example_request": {
            "septal_length": 5.1,
            "sepal_width":   3.5,
            "petal_length":  1.4,
            "petal_width":   0.2,
        },
        "expected_response": {
            "predicted_class": "Iris-setosa",
            "probabilities": {
                "Iris-setosa":     0.97,
                "Iris-versicolor": 0.02,
                "Iris-virginica":  0.01,
            },
        },
    }
