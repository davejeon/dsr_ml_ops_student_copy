
from __future__ import annotations

import io
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from airflow.decorators import dag, task

# Resolve the iris.csv path relative to this DAG file so it works
# regardless of the current working directory when Airflow executes tasks.
DATA_PATH = str(
    pathlib.Path(__file__).resolve().parent.parent.parent.parent
    / "Session_1" / "iris.csv"
)


@dag(
    dag_id="iris_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,       # manual trigger only
    catchup=False,
    tags=["iris", "demo"],
    doc_md="""
    ## Iris ML Pipeline

    End-to-end pipeline for the iris dataset:
    `load_data → validate_data → featurize → train_model → evaluate`
    """,
)
def iris_pipeline():

    @task
    def load_data() -> str:
        """Read iris.csv and return it as a JSON string (passed via XCom)."""
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df)} rows | columns: {df.columns.tolist()}")
        return df.to_json(orient="records")

    @task
    def validate_data(raw_json: str) -> str:
        """Assert schema, nulls, and numeric ranges. Raises on failure."""
        df = pd.read_json(io.StringIO(raw_json), orient="records")

        expected_cols = {"septal_length", "sepal_width",
                         "petal_length", "petal_width", "class"}
        missing = expected_cols - set(df.columns)
        assert not missing, f"Missing columns: {missing}"
        assert df.isnull().sum().sum() == 0, "Null values detected"
        assert df["septal_length"].between(0, 20).all(),             "septal_length contains out-of-range values"
        assert df["sepal_width"].between(0, 20).all(),             "sepal_width contains out-of-range values"

        print(
            f"Validation passed | {len(df)} rows | "
            f"{df['class'].nunique()} classes: {df['class'].unique().tolist()}"
        )
        return raw_json

    @task
    def featurize(raw_json: str) -> dict:
        """Encode the class label and split into features / targets."""
        df = pd.read_json(io.StringIO(raw_json), orient="records")
        le = LabelEncoder()
        X = df.drop(columns=["class"]).values.tolist()
        y = le.fit_transform(df["class"]).tolist()
        print(
            f"Feature matrix: ({len(X)} rows x {len(X[0])} cols) | "
            f"Classes: {le.classes_.tolist()}"
        )
        return {"X": X, "y": y, "classes": le.classes_.tolist()}

    @task
    def train_model(features: dict) -> dict:
        """Train a LogisticRegression and return predictions for evaluation."""
        X = np.array(features["X"])
        y = np.array(features["y"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = LogisticRegression(max_iter=300, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test).tolist()
        print(
            f"Trained on {len(X_train)} samples | "
            f"Test set size: {len(X_test)}"
        )
        return {
            "y_test": y_test.tolist(),
            "preds": preds,
            "classes": features["classes"],
        }

    @task
    def evaluate(results: dict) -> float:
        """Compute and print accuracy + classification report."""
        acc = accuracy_score(results["y_test"], results["preds"])
        report = classification_report(
            results["y_test"],
            results["preds"],
            target_names=results["classes"],
        )
        print(f"\n{'=' * 50}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"{'=' * 50}")
        print(report)
        return float(acc)

    # ── Wire tasks together (defines the DAG edges) ────────────────────────────
    raw    = load_data()
    valid  = validate_data(raw)
    feats  = featurize(valid)
    result = train_model(feats)
    evaluate(result)


iris_pipeline()   # instantiate — Airflow scans for this at module level
