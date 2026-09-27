"""
tests/test_cleaning.py
Pruebas unitarias para src/data/cleaning.py, basadas en los hallazgos
validados en notebooks/01_data_audit.ipynb.
"""

import pandas as pd
import pytest

from src.data.cleaning import (
    add_cancellation_flag,
    filter_valid_customers,
    remove_system_noise,
    get_clean_sales,
    build_customer_profile,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Invoice": ["489434", "C489435", "489436", "489437", "489438"],
        "StockCode": ["85048", "79323P", "TEST001", "22041", "21232"],
        "Description": ["A", "B", "This is a test product.", "C", "D"],
        "Quantity": [12, -5, 5, 100, 20000],
        "Price": [6.95, 6.75, 0.0, 2.10, 1.50],
        "Customer ID": [13085.0, 13085.0, 14103.0, None, 17850.0],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "France", "United Kingdom"],
    })


def test_add_cancellation_flag(sample_df):
    result = add_cancellation_flag(sample_df)
    assert result["is_cancellation"].tolist() == [False, True, False, False, False]


def test_filter_valid_customers(sample_df):
    result = filter_valid_customers(sample_df)
    assert result["Customer ID"].isna().sum() == 0
    assert len(result) == 4  # se elimina la fila con Customer ID nulo


def test_remove_system_noise(sample_df):
    result = remove_system_noise(sample_df)
    assert "TEST001" not in result["StockCode"].values
    assert len(result) == 4


def test_get_clean_sales_excludes_cancellations_noise_and_nulls(sample_df):
    result = get_clean_sales(sample_df)
    # Debe excluir: cancelación (C489435), TEST001, y Customer ID nulo (489437)
    assert len(result) == 2
    assert not result["is_cancellation"].any()
    assert "TEST001" not in result["StockCode"].values
    assert result["Customer ID"].notna().all()


def test_build_customer_profile_columns(sample_df):
    df_valid = get_clean_sales(sample_df)
    profile = build_customer_profile(df_valid)
    expected_cols = {
        "Customer ID", "total_quantity", "total_spend",
        "n_invoices", "n_countries", "is_wholesale",
    }
    assert expected_cols.issubset(profile.columns)
    assert len(profile) == df_valid["Customer ID"].nunique()