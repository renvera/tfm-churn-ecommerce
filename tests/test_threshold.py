import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

from src.evaluation.threshold import best_threshold_f1, threshold_for_min_recall


def test_best_threshold_f1_on_separable_data_gives_perfect_f1():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    thr = best_threshold_f1(y, p)
    assert f1_score(y, (p >= thr).astype(int)) == pytest.approx(1.0)


def test_threshold_for_min_recall_meets_the_recall_target():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 500), 0, 1)
    thr = threshold_for_min_recall(y, p, min_recall=0.8)
    assert recall_score(y, (p >= thr).astype(int)) >= 0.8


def test_threshold_for_min_recall_rejects_invalid_target():
    with pytest.raises(ValueError):
        threshold_for_min_recall([0, 1], [0.2, 0.8], min_recall=1.5)