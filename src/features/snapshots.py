"""
src/features/snapshots.py

Genera múltiples snapshots mensuales y construye el dataset final
cliente + snapshot, filtrando en cada corte a clientes con compra
reciente (últimos N días antes del corte).
"""

from __future__ import annotations

import pandas as pd

from src.features.pipeline import build_dataset


def generate_snapshot_dates(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    horizon_days: int,
    freq: str = "MS",
) -> pd.DatetimeIndex:
    """
    Genera fechas de corte mensuales entre min_date y la última fecha
    que permite observar el horizonte completo (max_date - horizon_days),
    evitando censura por la derecha.
    """
    last_valid_cutoff = max_date - pd.Timedelta(days=horizon_days)
    if last_valid_cutoff < min_date:
        raise ValueError(
            "El horizonte es más largo que el rango de datos disponible."
        )
    return pd.date_range(start=min_date, end=last_valid_cutoff, freq=freq)


def filter_active_customers(
    df_sales: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    lookback_days: int = 365,
) -> set:
    """
    Devuelve el conjunto de Customer ID con al menos una compra en
    (cutoff_date - lookback_days, cutoff_date].
    """
    start = cutoff_date - pd.Timedelta(days=lookback_days)
    mask = (df_sales["InvoiceDate"] > start) & (
        df_sales["InvoiceDate"] <= cutoff_date
    )
    return set(df_sales.loc[mask, "Customer ID"].unique())


def build_multi_snapshot_dataset(
    df_sales: pd.DataFrame,
    snapshot_dates: pd.DatetimeIndex,
    horizon_days: int = 90,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Construye el dataset completo cliente + snapshot, iterando sobre
    varias fechas de corte. En cada corte, se queda solo con clientes
    activos en los últimos `lookback_days` días.

    Returns
    -------
    pd.DataFrame
        Una fila por (cliente, snapshot). Incluye todas las columnas de
        build_dataset (features + churn + cutoff_date).
    """
    frames = []
    for cutoff in snapshot_dates:
        active_ids = filter_active_customers(
            df_sales, cutoff_date=cutoff, lookback_days=lookback_days
        )
        if not active_ids:
            continue

        snapshot_dataset = build_dataset(
            df_sales, cutoff_date=cutoff, horizon_days=horizon_days
        )
        snapshot_dataset = snapshot_dataset[
            snapshot_dataset["Customer ID"].isin(active_ids)
        ]
        frames.append(snapshot_dataset)

    if not frames:
        raise ValueError("Ningún snapshot generó clientes activos.")

    return pd.concat(frames, ignore_index=True)