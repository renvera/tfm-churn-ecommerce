"""
src/evaluation/metrics.py

Métricas de evaluación para clasificación binaria con probabilidades,
según lo definido en el documento maestro (sección 9 y 12).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calcula PR-AUC, ROC-AUC, precision, recall y F1 para un threshold dado.

    PR-AUC y ROC-AUC no dependen del threshold (usan proba directamente);
    precision/recall/F1 sí, y 0.5 no es necesariamente el threshold óptimo
    -- eso se decide más adelante (sección 12 del doc técnico).
    """
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }