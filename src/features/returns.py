"""
src/features/returns.py

Features de devoluciones (cancelaciones) por cliente y fecha de corte.

`get_clean_sales` descarta las cancelaciones para quedarse solo con ventas
reales, así que estas features se construyen aparte, a partir de las
cancelaciones del DataFrame sin filtrar (`get_cancellations`).

Sin leakage: para cada fila (cliente, cutoff_date) solo se usan
cancelaciones con InvoiceDate <= cutoff_date, igual que en rfm.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.cleaning import filter_valid_customers, remove_system_noise

_CANCELLATION_COLUMNS = ["Customer ID", "InvoiceDate", "Quantity", "Price"]


def get_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve solo las líneas de cancelación (Invoice con prefijo 'C') con
    Customer ID válido y sin ruido de sistema (TEST001).

    Recibe el DataFrame crudo, antes de `get_clean_sales`, que las elimina.
    """
    is_cancellation = df["Invoice"].str.startswith("C", na=False)
    cancellations = df[is_cancellation].copy()
    cancellations = filter_valid_customers(cancellations)
    cancellations = remove_system_noise(cancellations)
    return cancellations


def add_return_features(
    dataset: pd.DataFrame,
    cancellations: pd.DataFrame,
    date_col: str = "cutoff_date",
) -> pd.DataFrame:
    """
    Añade al dataset (una fila por cliente + fecha de corte) las columnas:

    - return_amount: importe devuelto hasta el corte (|Quantity| * Price de
      las cancelaciones con InvoiceDate <= corte), 0 si no hay devoluciones.
    - monetary_net: monetary - return_amount. Puede ser negativo si lo
      devuelto supera lo comprado en la ventana observada.
    - return_ratio: return_amount / monetary; 0 si monetary <= 0.

    Parameters
    ----------
    dataset:
        Debe contener Customer ID, `date_col` y monetary.
    cancellations:
        Líneas de cancelación (ver `get_cancellations`) con Customer ID,
        InvoiceDate, Quantity y Price.
    date_col:
        Columna con la fecha de corte de cada fila.

    Returns
    -------
    pd.DataFrame
        Copia de `dataset` (mismo índice) con las tres columnas nuevas.
    """
    required = {"Customer ID", date_col, "monetary"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Faltan columnas en dataset: {sorted(missing)}")
    missing_canc = set(_CANCELLATION_COLUMNS) - set(cancellations.columns)
    if missing_canc:
        raise ValueError(f"Faltan columnas en cancellations: {sorted(missing_canc)}")

    out = dataset.copy()

    canc = cancellations[_CANCELLATION_COLUMNS].copy()
    canc["InvoiceDate"] = pd.to_datetime(canc["InvoiceDate"])
    canc["_return_amount"] = (
        pd.to_numeric(canc["Quantity"]).abs() * pd.to_numeric(canc["Price"])
    )

    keys = out[["Customer ID", date_col]].copy()
    keys["_row"] = np.arange(len(out))

    pairs = keys.merge(
        canc[["Customer ID", "InvoiceDate", "_return_amount"]],
        on="Customer ID",
        how="inner",
    )
    # Solo pasado: cancelaciones hasta el corte inclusive.
    pairs = pairs[pairs["InvoiceDate"] <= pairs[date_col]]

    per_row = pairs.groupby("_row")["_return_amount"].sum()
    return_amount = per_row.reindex(range(len(out)), fill_value=0.0).astype(float)

    out["return_amount"] = return_amount.to_numpy()
    out["monetary_net"] = out["monetary"] - out["return_amount"]
    out["return_ratio"] = np.where(
        out["monetary"] > 0, out["return_amount"] / out["monetary"], 0.0
    )
    return out