"""
train.py — Train a LogisticRegression classifier on the iris dataset and
save the fitted model + label encoder to disk for the FastAPI service.

Usage:
    python train.py

Outputs:
    iris_classifier.joblib   — the trained pipeline (scaler + classifier)
    label_encoder.joblib     — the LabelEncoder mapping class names ↔ integers
"""

import pathlib

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths 
HERE      = pathlib.Path(__file__).resolve().parent
DATA_PATH = HERE.parent.parent / "day1" / "Session_1" / "iris.csv"

FEATURE_COLS = ["septal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COL   = "class"

# Load data 
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH.name}")

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

# Encode labels 
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Classes: {le.classes_.tolist()}")

# Train / test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Build pipeline: standardise → classify 
# Wrapping in a sklearn Pipeline means the scaler and classifier are saved
# together as a single object — the service loads one file and calls .predict().
pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(max_iter=300, random_state=42)),
])
pipeline.fit(X_train, y_train)

# Evaluate 
preds = pipeline.predict(X_test)
acc   = accuracy_score(y_test, preds)
print(f"\nTest accuracy: {acc:.4f}")
print(classification_report(y_test, preds, target_names=le.classes_))

# Save artifacts 
model_path = HERE / "iris_classifier.joblib"
le_path    = HERE / "label_encoder.joblib"

joblib.dump(pipeline, model_path)
joblib.dump(le,       le_path)

print(f"Model saved   : {model_path}")
print(f"Encoder saved : {le_path}")
