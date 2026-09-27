"""
src/labeling/churn.py

Construcción de etiquetas temporales de churn a nivel de cliente.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd


def build_churn_labels(
    df_sales: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int = 90,
) -> pd.DataFrame:
    """
    Construye una etiqueta de churn para cada cliente.

    Un cliente se considera churn si no realiza ninguna compra válida
    durante los horizon_days posteriores a cutoff_date.

    Parameters
    ----------
    df_sales:
        DataFrame de ventas limpias. Debe contener:
        Customer ID, InvoiceDate e Invoice.

    cutoff_date:
        Fecha final del periodo de observación.

    horizon_days:
        Número de días posteriores al cutoff utilizados para observar
        si el cliente vuelve a comprar.

    Returns
    -------
    pd.DataFrame
        Una fila por cliente con:
        - Customer ID
        - last_purchase_date
        - days_since_last_purchase
        - churn
        - cutoff_date
        - horizon_days
    """
    required_columns = {"Customer ID", "InvoiceDate"}
    missing_columns = required_columns - set(df_sales.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(missing_columns)}"
        )

    if horizon_days <= 0:
        raise ValueError("horizon_days debe ser mayor que cero.")

    df = df_sales.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    cutoff_date = pd.Timestamp(cutoff_date)
    horizon_end = cutoff_date + timedelta(days=horizon_days)

    observation = df[df["InvoiceDate"] <= cutoff_date].copy()

    future_purchases = df[
        (df["InvoiceDate"] > cutoff_date)
        & (df["InvoiceDate"] <= horizon_end)
    ].copy()

    last_purchase = (
        observation.groupby("Customer ID")["InvoiceDate"]
        .max()
        .rename("last_purchase_date")
    )

    future_purchase_count = (
        future_purchases.groupby("Customer ID")
        .size()
        .rename("future_purchase_count")
    )

    labels = last_purchase.to_frame()
    labels = labels.join(future_purchase_count, how="left")
    labels["future_purchase_count"] = (
        labels["future_purchase_count"].fillna(0).astype(int)
    )

    labels["churn"] = (
        labels["future_purchase_count"] == 0
    ).astype(int)

    labels["cutoff_date"] = cutoff_date
    labels["horizon_days"] = horizon_days
    labels["days_since_last_purchase"] = (
        cutoff_date - labels["last_purchase_date"]
    ).dt.days

    return labels.reset_index()