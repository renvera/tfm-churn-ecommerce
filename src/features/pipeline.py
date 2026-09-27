"""
src/features/pipeline.py

Combina features RFM/comportamentales con la etiqueta de churn en un
único dataset por cliente + fecha de corte, respetando la separación
temporal: las features solo usan el pasado (<= cutoff_date), la etiqueta
solo usa el futuro (cutoff_date, cutoff_date + horizon_days].
"""

from __future__ import annotations
from src.features.temporal import add_seasonal_features

import pandas as pd

from src.features.rfm import build_rfm_features
from src.labeling.churn import build_churn_labels


def build_dataset(
    df_sales: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int = 90,
) -> pd.DataFrame:
    """
    Construye el dataset cliente + snapshot: features (pasado) unidas
    a la etiqueta de churn (futuro), para una única fecha de corte.

    Returns
    -------
    pd.DataFrame
        Una fila por cliente con todas las columnas de features más
        churn, days_since_last_purchase, horizon_days.
    """
    features = build_rfm_features(df_sales, cutoff_date=cutoff_date)
    labels = build_churn_labels(
        df_sales, cutoff_date=cutoff_date, horizon_days=horizon_days
    )

    label_cols = labels[["Customer ID", "churn"]]

    dataset = features.merge(label_cols, on="Customer ID", how="inner")

    if len(dataset) != len(features):
        raise ValueError(
            "Inconsistencia entre features y labels: "
            f"{len(features)} clientes en features, "
            f"{len(dataset)} tras el merge con labels. "
            "Revisa que ambas funciones usen el mismo cutoff_date "
            "y el mismo df_sales."
        )

    dataset = add_seasonal_features(dataset, date_col="cutoff_date")

    return dataset