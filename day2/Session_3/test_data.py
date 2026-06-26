"""
test_data.py — Basic data validation tests for the iris dataset.

Checks the three things you should always verify before training:
  1. The expected columns are there.
  2. There are no missing values.
  3. The class labels are what we expect.

Run with:
    pytest day2/Session_3/test_data.py -v
"""

import pandas as pd
import pytest

from conftest import CLASSES, DATA_PATH, FEATURE_COLS, TARGET_COL


def test_required_columns_are_present(iris_df):
    """All feature columns and the target column must exist in the CSV."""
    expected = set(FEATURE_COLS + [TARGET_COL])
    missing  = expected - set(iris_df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_no_missing_values(iris_df):
    """No NaN values anywhere — missing data would silently corrupt training."""
    null_counts = iris_df.isnull().sum()
    problems    = null_counts[null_counts > 0]
    assert problems.empty, f"Columns with null values:\n{problems}"


def test_only_known_class_labels(iris_df):
    """
    The target column must contain only the three known iris species.
    An unknown label (e.g. a typo or a new species) would break the
    LabelEncoder used in train.py.
    """
    observed = set(iris_df[TARGET_COL].unique())
    unknown  = observed - set(CLASSES)
    assert not unknown, f"Unknown labels in dataset: {unknown}"


def test_dataset_has_enough_rows(iris_df):
    """Need enough rows for a meaningful train/test split."""
    assert len(iris_df) >= 100, (
        f"Only {len(iris_df)} rows — need at least 100 for reliable training"
    )
