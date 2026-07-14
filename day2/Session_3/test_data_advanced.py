"""
test_data.py — Data layer tests for the iris dataset.

These tests run at the START of the CT (Continuous Training) pipeline,
before any model is trained. If they fail, training is blocked — there is
no point training on bad data.

What they catch:
  - Missing or renamed columns (schema drift from upstream ETL).
  - Null / NaN values that would silently corrupt model training.
  - Out-of-range feature values (sensor errors, unit mismatches).
  - Unexpected class labels (a new species appeared, or a label was renamed).
  - A severely imbalanced or shifted label distribution.
  - Too few rows to train reliably.

Run with:
    pytest day2/Session_3/test_data.py -v
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

# conftest.py provides: iris_df, FEATURE_COLS, TARGET_COL, CLASSES
from conftest import CLASSES, DATA_PATH, FEATURE_COLS, TARGET_COL


# ── 1. Schema tests ────────────────────────────────────────────────────────────

class TestSchema:
    """The CSV must have exactly the columns we expect, with no extras missing."""

    def test_expected_columns_present(self, iris_df):
        expected = set(FEATURE_COLS + [TARGET_COL])
        assert expected.issubset(set(iris_df.columns)), (
            f"Missing columns: {expected - set(iris_df.columns)}"
        )

    def test_feature_columns_are_numeric(self, iris_df):
        for col in FEATURE_COLS:
            assert pd.api.types.is_numeric_dtype(iris_df[col]), (
                f"Column '{col}' should be numeric, got {iris_df[col].dtype}"
            )

    def test_target_column_is_string(self, iris_df):
        assert iris_df[TARGET_COL].dtype == object, (
            f"Expected string target column, got {iris_df[TARGET_COL].dtype}"
        )


# ── 2. Completeness tests ──────────────────────────────────────────────────────

class TestCompleteness:
    """No nulls, no empty strings, enough rows to train on."""

    def test_no_null_values(self, iris_df):
        null_counts = iris_df.isnull().sum()
        has_nulls   = null_counts[null_counts > 0]
        assert has_nulls.empty, f"Null values found:\n{has_nulls}"

    def test_sufficient_row_count(self, iris_df):
        # We need enough data to train and evaluate reliably.
        # The original dataset has 150 rows; warn if we get far fewer.
        assert len(iris_df) >= 100, (
            f"Only {len(iris_df)} rows found — expected at least 100 for reliable training"
        )

    def test_no_duplicate_rows(self, iris_df):
        n_dupes = iris_df.duplicated().sum()
        # The iris dataset has a known small number of near-duplicates;
        # flag if > 5% are exact duplicates (likely a pipeline bug).
        assert n_dupes / len(iris_df) < 0.05, (
            f"{n_dupes} duplicate rows ({n_dupes/len(iris_df):.1%}) — "
            "possible ETL duplication error"
        )


# ── 3. Range / validity tests ──────────────────────────────────────────────────

class TestFeatureRanges:
    """
    Physical constraints for iris measurements.
    Values outside these ranges indicate sensor errors or unit mismatches
    (e.g. cm vs mm), not real biological variation.
    """

    # Biologically plausible ranges for iris measurements (in cm)
    RANGES = {
        "septal_length": (4.0, 8.0),
        "sepal_width":   (2.0, 5.0),
        "petal_length":  (1.0, 7.0),
        "petal_width":   (0.1, 3.0),
    }

    @pytest.mark.parametrize("col,bounds", RANGES.items())
    def test_feature_within_plausible_range(self, iris_df, col, bounds):
        lo, hi = bounds
        out_of_range = iris_df[(iris_df[col] < lo) | (iris_df[col] > hi)]
        assert out_of_range.empty, (
            f"Column '{col}' has {len(out_of_range)} values outside "
            f"[{lo}, {hi}]:\n{out_of_range[[col]].head()}"
        )

    @pytest.mark.parametrize("col", FEATURE_COLS)
    def test_feature_is_positive(self, iris_df, col):
        non_positive = iris_df[iris_df[col] <= 0]
        assert non_positive.empty, (
            f"Column '{col}' has {len(non_positive)} non-positive values — "
            "measurements must be > 0"
        )


# ── 4. Label / class tests ─────────────────────────────────────────────────────

class TestLabels:
    """Class labels must be exactly the three known iris species."""

    def test_only_known_classes_present(self, iris_df):
        observed = set(iris_df[TARGET_COL].unique())
        expected = set(CLASSES)
        unknown  = observed - expected
        assert not unknown, (
            f"Unknown class labels found: {unknown}. "
            f"Expected only: {expected}"
        )

    def test_all_classes_represented(self, iris_df):
        observed = set(iris_df[TARGET_COL].unique())
        missing  = set(CLASSES) - observed
        assert not missing, (
            f"Some classes have no samples: {missing}. "
            "Cannot train a classifier that predicts all species."
        )

    def test_class_distribution_roughly_balanced(self, iris_df):
        """
        Each class should have between 25% and 45% of total rows.
        The reference iris dataset is perfectly balanced at 33.3% each.
        A large imbalance suggests a data pipeline error.
        """
        counts = iris_df[TARGET_COL].value_counts(normalize=True)
        for cls, proportion in counts.items():
            assert 0.25 <= proportion <= 0.45, (
                f"Class '{cls}' has proportion {proportion:.1%} — "
                "expected roughly 33% (25–45% is acceptable)"
            )

    def test_each_class_has_enough_samples(self, iris_df):
        """Need at least 10 samples per class for a meaningful train/test split."""
        counts = iris_df[TARGET_COL].value_counts()
        for cls, count in counts.items():
            assert count >= 10, (
                f"Class '{cls}' has only {count} samples — "
                "need at least 10 per class"
            )


# ── 5. Statistical property tests ─────────────────────────────────────────────

class TestStatisticalProperties:
    """
    Sanity checks on feature statistics.
    These catch upstream changes like feature scaling being applied
    before saving (mean would shift to ~0) or a feature being accidentally
    filled with a constant (zero variance).
    """

    def test_features_have_nonzero_variance(self, iris_df):
        for col in FEATURE_COLS:
            variance = iris_df[col].var()
            assert variance > 0.01, (
                f"Column '{col}' has near-zero variance ({variance:.4f}) — "
                "possibly a constant-fill bug upstream"
            )

    def test_petal_length_separates_setosa(self, iris_df):
        """
        Setosa has distinctly shorter petals than the other two species.
        This is a domain knowledge check: if setosa petal lengths overlap
        with versicolor, the data has been corrupted or mislabelled.
        """
        setosa_max     = iris_df[iris_df[TARGET_COL] == "Iris-setosa"]["petal_length"].max()
        versicolor_min = iris_df[iris_df[TARGET_COL] == "Iris-versicolor"]["petal_length"].min()
        assert setosa_max < versicolor_min, (
            f"Setosa petal lengths (max={setosa_max}) overlap with "
            f"versicolor (min={versicolor_min}) — possible mislabelling"
        )
