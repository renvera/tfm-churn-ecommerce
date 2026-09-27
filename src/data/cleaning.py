"""
src/data/cleaning.py
Reglas de limpieza validadas en notebooks/01_data_audit.ipynb.
"""

import pandas as pd


def add_cancellation_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Marca facturas de cancelación (Invoice con prefijo 'C')."""
    df = df.copy()
    df["is_cancellation"] = df["Invoice"].str.startswith("C")
    return df


def filter_valid_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas sin Customer ID (ruido: ajustes internos, no ventas)."""
    return df[df["Customer ID"].notna()].copy()


def remove_system_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina StockCode == 'TEST001' (productos de prueba del sistema)."""
    return df[df["StockCode"] != "TEST001"].copy()


def get_clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de limpieza completo: Customer ID válido, sin cancelaciones,
    sin ruido de sistema. Resultado: transacciones de venta reales.
    """
    df = add_cancellation_flag(df)
    df = filter_valid_customers(df)
    df = remove_system_noise(df)
    return df[~df["is_cancellation"]].copy()


def build_customer_profile(df_valid: pd.DataFrame, wholesale_percentile: float = 0.99) -> pd.DataFrame:
    """
    Construye perfil por cliente (quantity, spend, invoices, países) y
    marca mayoristas según percentil de Quantity o Spend.
    """
    profile = (
        df_valid.assign(spend=lambda d: d["Quantity"] * d["Price"])
        .groupby("Customer ID")
        .agg(
            total_quantity=("Quantity", "sum"),
            total_spend=("spend", "sum"),
            n_invoices=("Invoice", "nunique"),
            n_countries=("Country", "nunique"),
        )
        .reset_index()
    )

    q_threshold = profile["total_quantity"].quantile(wholesale_percentile)
    spend_threshold = profile["total_spend"].quantile(wholesale_percentile)

    profile["is_wholesale"] = (
        (profile["total_quantity"] > q_threshold) | (profile["total_spend"] > spend_threshold)
    )
    return profile