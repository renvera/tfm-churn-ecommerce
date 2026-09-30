"""
src/evaluation/threshold.py

Selección de threshold sobre un conjunto de validación. 0.5 no es
necesariamente el punto de corte óptimo (sección 12 del doc técnico).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def _pr_arrays(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall tienen un elemento más que thresholds
    return precision[:-1], recall[:-1], thresholds


def best_threshold_f1(y_true, proba) -> float:
    """Threshold que maximiza F1 en los datos dados."""
    precision, recall, thresholds = _pr_arrays(y_true, proba)
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0
    )
    return float(thresholds[int(np.argmax(f1))])


def threshold_for_min_recall(y_true, proba, min_recall: float) -> float:
    """
    Threshold que maximiza la precision exigiendo un recall mínimo.
    Útil cuando el negocio prefiere no dejar escapar churners.
    """
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall debe estar en (0, 1].")
    precision, recall, thresholds = _pr_arrays(y_true, proba)
    candidates = np.where(recall >= min_recall)[0]
    best = candidates[np.argmax(precision[candidates])]
    return float(thresholds[best])