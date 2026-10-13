"""
src/evaluation/bootstrap.py

Intervalos de confianza por bootstrap por cliente.

Un mismo cliente aparece en varios snapshots, así que las filas no son
independientes: se remuestrean clientes completos (con reemplazo) y se
recalculan las métricas ponderando cada fila por las veces que su cliente
sale en la muestra.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def cluster_bootstrap_ci(
    y_true,
    proba,
    customers,
    n_boot: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """
    IC (percentiles) de ROC-AUC y PR-AUC remuestreando clientes.

    Returns
    -------
    dict
        {"roc_auc": (lo, hi), "pr_auc": (lo, hi)}
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    _, codes = np.unique(np.asarray(customers), return_inverse=True)
    codes = codes.ravel()
    n_cust = int(codes.max()) + 1

    rng = np.random.default_rng(seed)
    roc, pr = [], []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_cust, n_cust), minlength=n_cust)
        w = counts[codes]
        # Una muestra sin ambas clases no permite calcular AUC: se descarta.
        if len(np.unique(y_true[w > 0])) < 2:
            continue
        roc.append(roc_auc_score(y_true, proba, sample_weight=w))
        pr.append(average_precision_score(y_true, proba, sample_weight=w))

    if not roc:
        raise ValueError("Ninguna muestra bootstrap contiene ambas clases.")

    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {
        "roc_auc": tuple(float(v) for v in np.percentile(roc, q)),
        "pr_auc": tuple(float(v) for v in np.percentile(pr, q)),
    }
