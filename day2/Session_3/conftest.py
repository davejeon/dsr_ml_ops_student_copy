"""
conftest.py — pytest fixtures shared across all Session 3 test files.

pytest automatically discovers this file and makes the fixtures available
to every test in the same directory without explicit imports.
"""

import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

#  Paths 
REPO_ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH    = REPO_ROOT / "day1" / "Session_1" / "iris.csv"
SESSION_1    = REPO_ROOT / "day2" / "Session_1"
MODEL_PATH   = SESSION_1 / "iris_classifier.joblib"
ENCODER_PATH = SESSION_1 / "label_encoder.joblib"

FEATURE_COLS = ["septal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COL   = "class"
CLASSES      = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


# Data fixtures 

@pytest.fixture(scope="session")
def iris_df():
    """The full iris CSV as a DataFrame. Loaded once for the whole test run."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="session")
def iris_X_y(iris_df):
    """Feature matrix and encoded label vector ready for sklearn."""
    le = LabelEncoder()
    X = iris_df[FEATURE_COLS].values
    y = le.fit_transform(iris_df[TARGET_COL].values)
    return X, y, le


@pytest.fixture(scope="session")
def train_test_data(iris_X_y):
    """Stratified 80/20 split — same split used in train.py (random_state=42)."""
    X, y, le = iris_X_y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, le


#  Model fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def trained_model():
    """
    Load the pre-trained model from Session 1.

    If the file doesn't exist the test that uses this fixture will error with
    a clear message rather than a cryptic FileNotFoundError.
    """
    if not MODEL_PATH.exists():
        pytest.skip(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run day2/Session_1/train.py first."
        )
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def label_encoder():
    """Load the pre-trained LabelEncoder from Session 1."""
    if not ENCODER_PATH.exists():
        pytest.skip(
            f"Encoder artifact not found at {ENCODER_PATH}. "
            "Run day2/Session_1/train.py first."
        )
    return joblib.load(ENCODER_PATH)


@pytest.fixture(scope="session")
def fresh_model(train_test_data):
    """
    A model trained fresh from the iris data — used for tests that need
    to control the training process independently of the saved artifact.
    """
    X_train, _, y_train, _, _ = train_test_data
    pipeline = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", LogisticRegression(max_iter=300, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
