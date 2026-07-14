"""
test_model.py — Model quality and behavioural tests for the iris classifier.

These tests run AFTER training, before the model is registered or deployed.
They form the quality gate in the CT (Continuous Training) pipeline.

Two kinds of tests:

  Model tests   — objective metrics on held-out data.
                  They fail if accuracy / confidence drops below acceptable levels.

  Behavioural tests — does the model make biological sense?
                  They catch overfitting to noise, spurious correlations, and
                  preprocessing bugs that happen to preserve accuracy on the
                  test set but produce nonsense predictions in the real world.

Run with:
    pytest day2/Session_3/test_model.py -v

Requires: day2/Session_1/train.py to have been run first.
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix

from conftest import CLASSES, FEATURE_COLS


# ── 1. Quality gates ───────────────────────────────────────────────────────────

class TestModelQuality:
    """
    Hard thresholds the model must meet on the held-out test set.
    If a new training run produces a model below these thresholds, it
    should NOT be registered or deployed.
    """

    ACCURACY_THRESHOLD = 0.90  # iris is an easy dataset; < 90% is suspicious

    def test_overall_accuracy_above_threshold(self, trained_model, train_test_data):
        _, X_test, _, y_test, _ = train_test_data
        preds = trained_model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        assert acc >= self.ACCURACY_THRESHOLD, (
            f"Overall accuracy {acc:.3f} is below the {self.ACCURACY_THRESHOLD} threshold. "
            "The model should not be deployed."
        )

    def test_per_class_recall_above_threshold(self, trained_model, train_test_data):
        """
        Overall accuracy can mask a model that completely ignores one class.
        Check that each species is recalled at least 80% of the time.
        """
        _, X_test, _, y_test, le = train_test_data
        preds = trained_model.predict(X_test)
        cm    = confusion_matrix(y_test, preds)

        # cm[i, i] / cm[i, :].sum() = recall for class i
        for i, cls_name in enumerate(le.classes_):
            recall = cm[i, i] / cm[i, :].sum()
            assert recall >= 0.80, (
                f"Recall for '{cls_name}' is {recall:.2%} — below 80%. "
                "The model ignores this class too often."
            )

    def test_model_returns_probabilities_for_all_classes(
        self, trained_model, train_test_data
    ):
        """predict_proba must return a probability for every known class."""
        X_train, _, _, _, le = train_test_data
        # Use one sample
        proba = trained_model.predict_proba(X_train[:1])
        assert proba.shape == (1, len(le.classes_)), (
            f"Expected shape (1, {len(le.classes_)}), got {proba.shape}"
        )

    def test_probabilities_sum_to_one(self, trained_model, train_test_data):
        X_train, X_test, _, _, _ = train_test_data
        X_all = np.vstack([X_train, X_test])
        probas = trained_model.predict_proba(X_all)
        row_sums = probas.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"Probabilities do not sum to 1.0. Min={row_sums.min():.6f}, "
            f"Max={row_sums.max():.6f}"
        )

    def test_model_is_not_trivially_predicting_one_class(
        self, trained_model, train_test_data
    ):
        """
        A degenerate model could reach reasonable accuracy by predicting
        the majority class on every input. Check that all three classes
        appear in predictions on the test set.
        """
        _, X_test, _, _, le = train_test_data
        preds          = trained_model.predict(X_test)
        predicted_set  = set(preds)
        expected_count = len(le.classes_)
        assert len(predicted_set) == expected_count, (
            f"Model only predicted {len(predicted_set)} of {expected_count} classes "
            f"on the test set: {predicted_set}. Possible degenerate model."
        )


# ── 2. Behavioural / invariance tests ─────────────────────────────────────────

class TestModelBehaviour:
    """
    Domain-knowledge checks: does the model behave in ways that make
    biological sense, even on inputs it hasn't seen?

    These tests do NOT require labelled data — they construct synthetic
    inputs that differ along one dimension and assert a directional effect.
    """

    # Canonical examples from the iris dataset (known correct predictions)
    SETOSA_SAMPLE     = [5.1, 3.5, 1.4, 0.2]   # short petals → setosa
    VIRGINICA_SAMPLE  = [6.7, 3.0, 5.2, 2.3]   # long petals → virginica
    VERSICOLOR_SAMPLE = [5.7, 2.8, 4.1, 1.3]   # medium → versicolor

    def _predict_class(self, model, label_encoder, features):
        idx = model.predict([features])[0]
        return label_encoder.inverse_transform([idx])[0]

    def _predict_proba(self, model, label_encoder, features, cls):
        probas     = model.predict_proba([features])[0]
        class_idx  = list(label_encoder.classes_).index(cls)
        return probas[class_idx]

    def test_canonical_setosa_is_classified_correctly(
        self, trained_model, label_encoder
    ):
        pred = self._predict_class(trained_model, label_encoder, self.SETOSA_SAMPLE)
        assert pred == "Iris-setosa", (
            f"Expected Iris-setosa for canonical setosa sample, got '{pred}'"
        )

    def test_canonical_virginica_is_classified_correctly(
        self, trained_model, label_encoder
    ):
        pred = self._predict_class(trained_model, label_encoder, self.VIRGINICA_SAMPLE)
        assert pred == "Iris-virginica", (
            f"Expected Iris-virginica for canonical virginica sample, got '{pred}'"
        )

    def test_longer_petals_increase_virginica_probability(
        self, trained_model, label_encoder
    ):
        """
        Virginica has the longest petals of the three species.
        Increasing petal_length (index 2) from a versicolor-like value
        towards a virginica-like value should increase P(virginica).
        """
        # Start from a mid-range flower and increase petal length step by step
        base        = [6.0, 3.0, 3.5, 1.2]   # ambiguous: versicolor territory
        long_petals = [6.0, 3.0, 6.0, 2.2]   # clearly virginica territory

        p_base = self._predict_proba(trained_model, label_encoder, base,       "Iris-virginica")
        p_long = self._predict_proba(trained_model, label_encoder, long_petals, "Iris-virginica")

        assert p_long > p_base, (
            f"P(virginica) should increase with longer petals, but "
            f"it went from {p_base:.3f} → {p_long:.3f}"
        )

    def test_shorter_petals_increase_setosa_probability(
        self, trained_model, label_encoder
    ):
        """
        Setosa has the shortest, narrowest petals. A sample with very
        short petals should have high P(setosa).
        """
        short_petals = [5.0, 3.5, 1.2, 0.2]   # clearly setosa
        long_petals  = [5.0, 3.5, 5.0, 1.8]   # clearly not setosa

        p_short = self._predict_proba(trained_model, label_encoder, short_petals, "Iris-setosa")
        p_long  = self._predict_proba(trained_model, label_encoder, long_petals,  "Iris-setosa")

        assert p_short > p_long, (
            f"P(setosa) should be higher for short petals ({p_short:.3f}) "
            f"than long petals ({p_long:.3f})"
        )

    def test_model_confidence_is_high_for_clear_setosa(
        self, trained_model, label_encoder
    ):
        """
        The canonical setosa sample should be classified with high confidence.
        If the model is uncertain (< 90%) on the textbook setosa example,
        something is wrong with the preprocessing or training.
        """
        proba = self._predict_proba(
            trained_model, label_encoder, self.SETOSA_SAMPLE, "Iris-setosa"
        )
        assert proba >= 0.90, (
            f"Model is only {proba:.1%} confident on a textbook setosa sample. "
            "Expected >= 90% — check for preprocessing or training issues."
        )

    def test_perturbation_does_not_catastrophically_change_prediction(
        self, trained_model, label_encoder
    ):
        """
        Adding tiny noise (0.01 cm, within measurement error) to a setosa
        sample should not change the predicted class. A fragile decision
        boundary suggests overfitting.
        """
        original  = self.SETOSA_SAMPLE
        perturbed = [v + 0.01 for v in original]  # +1 mm noise

        pred_orig = self._predict_class(trained_model, label_encoder, original)
        pred_pert = self._predict_class(trained_model, label_encoder, perturbed)

        assert pred_orig == pred_pert, (
            f"Prediction changed from '{pred_orig}' to '{pred_pert}' with only "
            "0.01 cm of noise — the decision boundary is unstable."
        )
