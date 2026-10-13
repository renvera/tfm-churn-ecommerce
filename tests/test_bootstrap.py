import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.bootstrap import cluster_bootstrap_ci


def _data(n_customers=200, rows_per_customer=3, seed=0, signal=1.0):
    rng = np.random.default_rng(seed)
    customers = np.repeat(np.arange(n_customers), rows_per_customer)
    y = rng.integers(0, 2, len(customers))
    proba = np.clip(0.5 + signal * (y - 0.5) * 0.4 + rng.normal(0, 0.25, len(y)), 0, 1)
    return y, proba, customers


def test_perfect_classifier_has_degenerate_ci():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    ci = cluster_bootstrap_ci(y, y.astype(float), np.arange(8), n_boot=50)
    assert ci["roc_auc"] == (1.0, 1.0)
    assert ci["pr_auc"] == (1.0, 1.0)


def test_ci_contains_point_estimate_and_is_ordered():
    y, p, c = _data()
    ci = cluster_bootstrap_ci(y, p, c, n_boot=200)
    point = roc_auc_score(y, p)
    lo, hi = ci["roc_auc"]
    assert lo < hi
    assert lo <= point <= hi


def test_same_seed_gives_same_result():
    y, p, c = _data()
    assert cluster_bootstrap_ci(y, p, c, seed=7) == cluster_bootstrap_ci(y, p, c, seed=7)


def test_resamples_customers_not_rows():
    """Con un solo cliente no hay variabilidad entre clientes que remuestrear:
    todas las muestras son idénticas y el IC se reduce a un punto."""
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.3, 0.6, 0.8, 0.2])
    ci = cluster_bootstrap_ci(y, p, np.zeros(6, dtype=int), n_boot=30)
    assert ci["roc_auc"][0] == pytest.approx(ci["roc_auc"][1])
