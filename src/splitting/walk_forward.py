"""
src/splitting/walk_forward.py

Folds de backtest rodante (walk-forward) con embargo, sobre fechas de
snapshot (cutoff_date).

Para cada snapshot de test posible, se entrena con todo lo anterior
(ventana expansiva) dejando un embargo de `embargo_days` entre el último
corte de train y el de test. Con embargo >= horizonte de churn, la ventana
de etiqueta de cualquier fila de train termina antes del corte de test
(ventanas disjuntas), así que train nunca "ve" el periodo evaluado.
"""

from __future__ import annotations

import pandas as pd


def walk_forward_folds(
    snapshot_dates,
    embargo_days: int,
    min_train_snapshots: int = 9,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Devuelve [(train_end, test_date), ...], un fold por cada snapshot de
    test viable.

    - train_end: el último snapshot con fecha <= test_date - embargo_days.
      Train usa todos los snapshots con fecha <= train_end.
    - Solo se generan folds con al menos `min_train_snapshots` snapshots
      de train.
    - Cada snapshot aparece como test como máximo una vez, así que los
      periodos de test no se solapan entre folds.
    """
    if embargo_days < 0:
        raise ValueError("embargo_days no puede ser negativo.")
    if min_train_snapshots < 1:
        raise ValueError("min_train_snapshots debe ser >= 1.")

    dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(snapshot_dates)))))
    embargo = pd.Timedelta(days=embargo_days)

    folds = []
    for test_date in dates:
        eligible = dates[dates <= test_date - embargo]
        if len(eligible) < min_train_snapshots:
            continue
        folds.append((eligible.max(), test_date))
    return folds
