"""
src/features/rfm.py

Construcción de features RFM (Recency, Frequency, Monetary) y variables
comportamentales adicionales, a nivel de cliente, respetando la fecha de
corte T: solo se usan compras con InvoiceDate <= T.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "Customer ID",
    "Invoice",
    "InvoiceDate",
    "Quantity",
    "Price",
    "Country",
}

_OUTPUT_COLUMNS = [
    "Customer ID",
    "recency",
    "frequency",
    "monetary",
    "n_products",
    "n_countries",
    "avg_days_between_purchases",
    "cutoff_date",
]


def build_rfm_features(
    df_sales: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Construye features RFM y comportamentales por cliente, usando
    únicamente compras con InvoiceDate <= cutoff_date.

    Parameters
    ----------
    df_sales:
        DataFrame de ventas limpias (sin cancelaciones). Debe contener:
        Customer ID, Invoice, InvoiceDate, Quantity, UnitPrice, Country.
        StockCode es opcional (se usa para n_products si está presente).

    cutoff_date:
        Fecha de corte T. Solo se usan compras con InvoiceDate <= T.

    Returns
    -------
    pd.DataFrame
        Una fila por cliente con:
        - Customer ID
        - recency: días desde la última compra hasta el corte
        - frequency: número de facturas distintas
        - monetary: gasto total (Quantity * UnitPrice)
        - n_products: productos distintos (si hay StockCode)
        - n_countries: países distintos
        - avg_days_between_purchases: intervalo medio entre facturas
          (NaN si el cliente tiene una sola compra observable)
        - cutoff_date
    """
    missing_columns = REQUIRED_COLUMNS - set(df_sales.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(missing_columns)}"
        )

    cutoff_date = pd.Timestamp(cutoff_date)

    df = df_sales.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    observation = df[df["InvoiceDate"] <= cutoff_date].copy()

    if observation.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    observation["line_amount"] = (
        observation["Quantity"] * observation["Price"]
    )

    last_purchase = (
        observation.groupby("Customer ID")["InvoiceDate"]
        .max()
        .rename("last_purchase_date")
    )
    recency = (cutoff_date - last_purchase).dt.days.rename("recency")

    frequency = (
        observation.groupby("Customer ID")["Invoice"]
        .nunique()
        .rename("frequency")
    )

    monetary = (
        observation.groupby("Customer ID")["line_amount"]
        .sum()
        .rename("monetary")
    )

    n_countries = (
        observation.groupby("Customer ID")["Country"]
        .nunique()
        .rename("n_countries")
    )

    features = pd.concat([recency, frequency, monetary, n_countries], axis=1)

    if "StockCode" in observation.columns:
        n_products = (
            observation.groupby("Customer ID")["StockCode"]
            .nunique()
            .rename("n_products")
        )
        features = features.join(n_products)

    # Intervalo medio entre compras: se calcula sobre fechas de factura
    # únicas por cliente (evita que varias líneas de la misma factura
    # cuenten como "compras" distintas).
    invoice_dates = (
        observation.groupby(["Customer ID", "Invoice"])["InvoiceDate"]
        .min()
        .reset_index()
        .sort_values(["Customer ID", "InvoiceDate"])
    )
    invoice_dates["gap_days"] = (
        invoice_dates.groupby("Customer ID")["InvoiceDate"].diff().dt.days
    )
    avg_gap = (
        invoice_dates.groupby("Customer ID")["gap_days"]
        .mean()
        .rename("avg_days_between_purchases")
    )
    features = features.join(avg_gap)

    features["cutoff_date"] = cutoff_date

    return features.reset_index()