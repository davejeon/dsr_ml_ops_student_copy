"""Training entrypoint: load → validate → featurize → fit → evaluate → log.

Mirrors the MLflow snippet in Section 4 of the manual. Run with:

    python -m churn.models.train
"""
from __future__ import annotations

import json
import os
import random

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn.config import settings
from churn.data.load import load_raw
from churn.data.validate import validate
from churn.features.build import build_features


def _set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    _set_seeds(settings.seed)

    df = validate(load_raw())
    X = build_features(df)
    y = df["churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.seed, stratify=y
    )

    params = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**params, random_state=settings.seed)),
    ])

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("seed", settings.seed)
        mlflow.log_param("n_train", len(X_train))

        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y_test, proba))
        acc = float(accuracy_score(y_test, preds))

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Quality gate — fails the training stage early if regression detected.
        if auc < settings.min_auc:
            raise RuntimeError(
                f"AUC {auc:.3f} below required minimum {settings.min_auc:.3f}"
            )

        joblib.dump(model, settings.model_path)
        with open("metrics.json", "w") as f:
            json.dump({"auc": auc, "accuracy": acc}, f, indent=2)

        print(f"Trained: auc={auc:.3f} accuracy={acc:.3f} -> {settings.model_path}")


if __name__ == "__main__":
    main()
