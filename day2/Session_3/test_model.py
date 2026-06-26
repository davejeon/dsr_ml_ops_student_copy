"""
test_model.py — Basic model quality and behavioural tests.

Two things to check after training:
  1. The model meets a minimum accuracy bar on held-out data.
  2. The model makes biological sense (longer petals → virginica).

Run with:
    pytest day2/Session_3/test_model.py -v

Requires: day2/Session_1/train.py to have been run first.
"""

from sklearn.metrics import accuracy_score


ACCURACY_THRESHOLD = 0.90


def test_accuracy_above_threshold(trained_model, train_test_data):
    """
    The model must reach 90% accuracy on the test set.
    If it drops below this, something went wrong with training
    and the model should not be deployed.
    """
    _, X_test, _, y_test, _ = train_test_data
    acc = accuracy_score(y_test, trained_model.predict(X_test))
    assert acc >= ACCURACY_THRESHOLD, (
        f"Accuracy {acc:.1%} is below the {ACCURACY_THRESHOLD:.0%} threshold — "
        "do not deploy this model"
    )


def test_canonical_setosa_prediction(trained_model, label_encoder):
    """
    A textbook setosa flower (short petals, narrow) must be classified
    as setosa. If this fails, the model or its preprocessing is broken.
    """
    # [septal_length, sepal_width, petal_length, petal_width]
    setosa_features = [[5.1, 3.5, 1.4, 0.2]]
    predicted_idx   = trained_model.predict(setosa_features)[0]
    predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
    assert predicted_class == "Iris-setosa", (
        f"Expected Iris-setosa for a canonical setosa sample, got '{predicted_class}'"
    )


def test_longer_petals_predict_virginica(trained_model, label_encoder):
    """
    Virginica has the longest petals of the three species.
    A sample with clearly long petals should be classified as virginica.
    This is a directional check — it catches models with inverted features
    or incorrect label mappings.
    """
    virginica_features = [[6.7, 3.0, 5.2, 2.3]]
    predicted_idx      = trained_model.predict(virginica_features)[0]
    predicted_class    = label_encoder.inverse_transform([predicted_idx])[0]
    assert predicted_class == "Iris-virginica", (
        f"Expected Iris-virginica for a long-petal sample, got '{predicted_class}'"
    )
