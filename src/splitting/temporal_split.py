"""
src/splitting/temporal_split.py

Split temporal del dataset cliente+snapshot: train, validation y test
se definen por fecha de snapshot (cutoff_date), nunca de forma aleatoria.

Limitación conocida y documentada: un mismo Customer ID puede aparecer
en varios snapshots (y por tanto en más de un split). Esto no constituye
leakage temporal -- cada fila usa solo pasado en X y solo futuro en y,
respetando su propio cutoff_date -- pero introduce dependencia entre
observaciones del mismo cliente, que debe mencionarse como limitación
metodológica en el TFM.
"""

from __future__ import annotations

import pandas as pd


def temporal_split(
    dataset: pd.DataFrame,
    date_col: str,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en train/valid/test según cutoff_date.

    - train: date_col <= train_end
    - valid: train_end < date_col <= valid_end
    - test:  date_col > valid_end
    """
    if train_end >= valid_end:
        raise ValueError("train_end debe ser anterior a valid_end.")

    train = dataset[dataset[date_col] <= train_end].copy()
    valid = dataset[
        (dataset[date_col] > train_end) & (dataset[date_col] <= valid_end)
    ].copy()
    test = dataset[dataset[date_col] > valid_end].copy()

    return train, valid, test


def summarize_split(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = "cutoff_date",
) -> pd.DataFrame:
    """Resumen rápido para verificar el split antes de modelar."""
    rows = []
    for name, df in [("train", train), ("valid", valid), ("test", test)]:
        rows.append(
            {
                "split": name,
                "filas": len(df),
                "clientes_unicos": df["Customer ID"].nunique(),
                "snapshots": df[date_col].nunique(),
                "fecha_min": df[date_col].min(),
                "fecha_max": df[date_col].max(),
                "%churn": round(df["churn"].mean() * 100, 2)
                if len(df) > 0
                else None,
            }
        )
    return pd.DataFrame(rows)