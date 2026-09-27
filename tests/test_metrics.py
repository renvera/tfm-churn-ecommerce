import numpy as np

from src.evaluation.metrics import evaluate_binary


def test_evaluate_binary_returns_expected_keys():
    y_true = np.array([0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.8, 0.6, 0.3, 0.9])
    result = evaluate_binary(y_true, proba)
    assert set(result.keys()) == {
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
    }


def test_perfect_predictions_give_perfect_scores():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.0, 0.1, 0.9, 1.0])
    result = evaluate_binary(y_true, proba)
    assert result["roc_auc"] == 1.0
    assert result["pr_auc"] == 1.0