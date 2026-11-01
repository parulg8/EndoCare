import numpy as np

from src.evaluate_models import classification_metrics, expected_calibration_error, specificity_score


def test_specificity_and_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert specificity_score(y_true, y_pred) == 1.0
    metrics = classification_metrics(y_true, y_pred, y_score)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == 1.0


def test_ece_range():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_score = np.linspace(0.05, 0.95, 10)
    ece = expected_calibration_error(y_true, y_score, n_bins=5)
    assert 0.0 <= ece <= 1.0
