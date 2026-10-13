"""
src/splitting/temporal_split.py

Split temporal del dataset cliente+snapshot: train, validation y test
se definen por fecha de snapshot (cutoff_date), nunca de forma aleatoria.

Embargo (gap_days): la etiqueta de una fila con corte T se calcula con las
compras de (T, T + horizon]. Si el siguiente split empieza antes de
T + horizon, la etiqueta de la fila de train se apoya en compras que caen
dentro del periodo del split siguiente (solapamiento de ventanas de
etiqueta). Para evitarlo, `gap_days` (normalmente igual al horizonte de
churn) descarta los snapshots que caen dentro de ese margen tras
train_end y tras valid_end. Con gap_days=0 se conserva el comportamiento
anterior, que sí presenta ese solapamiento.

Limitación conocida y documentada: un mismo Customer ID puede aparecer
en varios snapshots (y por tanto en más de un split). Las features de cada
fila usan solo pasado y su etiqueta solo futuro respecto a su propio
cutoff_date, pero introduce dependencia entre observaciones del mismo
cliente, que debe mencionarse como limitación metodológica en el TFM.
"""

from __future__ import annotations

import pandas as pd


def temporal_split(
    dataset: pd.DataFrame,
    date_col: str,
    train_end: pd.Timestamp,
    valid_end: pd.Timestamp,
    gap_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en train/valid/test según cutoff_date, con un
    embargo opcional de `gap_days` entre splits.

    - train: date_col <= train_end
    - valid: date_col > train_end  y  date_col >= train_end + gap_days
             y  date_col <= valid_end
    - test:  date_col > valid_end  y  date_col >= valid_end + gap_days

    Las filas cuyo corte cae dentro de cada embargo no se asignan a ningún
    split. Con gap_days >= horizon_days, la etiqueta de cada fila de un
    split (ventana (T, T + horizon]) queda totalmente antes del primer
    corte del split siguiente: un corte situado exactamente en
    T + horizon se conserva, porque las ventanas son disjuntas.
    """
    if gap_days < 0:
        raise ValueError("gap_days no puede ser negativo.")
    if train_end >= valid_end:
        raise ValueError("train_end debe ser anterior a valid_end.")

    gap = pd.Timedelta(days=gap_days)
    valid_start = train_end + gap
    test_start = valid_end + gap

    if valid_start > valid_end:
        raise ValueError(
            "gap_days es demasiado grande: el embargo tras train_end "
            "deja valid vacío (train_end + gap_days > valid_end)."
        )

    train = dataset[dataset[date_col] <= train_end].copy()
    valid = dataset[
        (dataset[date_col] > train_end)
        & (dataset[date_col] >= valid_start)
        & (dataset[date_col] <= valid_end)
    ].copy()
    test = dataset[
        (dataset[date_col] > valid_end) & (dataset[date_col] >= test_start)
    ].copy()

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