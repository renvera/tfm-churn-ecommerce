import pandas as pd
import pytest

from src.features.snapshots import (
    build_multi_snapshot_dataset,
    filter_active_customers,
    generate_snapshot_dates,
)


def test_generate_snapshot_dates_respects_horizon():
    min_date = pd.Timestamp("2011-01-01")
    max_date = pd.Timestamp("2011-12-09")
    dates = generate_snapshot_dates(min_date, max_date, horizon_days=90)
    # Ningún snapshot puede dejar menos de 90 días de futuro disponibles
    assert dates.max() <= max_date - pd.Timedelta(days=90)


def test_generate_snapshot_dates_raises_if_horizon_too_long():
    min_date = pd.Timestamp("2011-11-01")
    max_date = pd.Timestamp("2011-12-09")
    with pytest.raises(ValueError):
        generate_snapshot_dates(min_date, max_date, horizon_days=90)


@pytest.fixture
def sample_sales():
    return pd.DataFrame(
        {
            "Customer ID": [1, 1, 2, 3],
            "Invoice": ["100", "101", "200", "300"],
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-06-01",
                    "2011-06-01",
                    "2010-01-01",  # cliente 3: compra antigua, ya inactivo
                ]
            ),
            "StockCode": ["A", "B", "C", "D"],
            "Quantity": [1, 1, 1, 1],
            "Price": [10.0, 10.0, 10.0, 10.0],
            "Country": ["United Kingdom"] * 4,
        }
    )


def test_filter_active_customers_excludes_old_purchases(sample_sales):
    active = filter_active_customers(
        sample_sales, cutoff_date=pd.Timestamp("2011-06-15"), lookback_days=365
    )
    assert 1 in active  # compró 2011-06-01, dentro de 365 días
    assert 2 in active
    assert 3 not in active  # última compra 2010-01-01, fuera de ventana


def test_multi_snapshot_dataset_has_one_row_per_active_customer_snapshot(
    sample_sales,
):
    snapshot_dates = pd.DatetimeIndex(
        [pd.Timestamp("2011-06-15"), pd.Timestamp("2011-07-15")]
    )
    dataset = build_multi_snapshot_dataset(
        sample_sales,
        snapshot_dates=snapshot_dates,
        horizon_days=30,
        lookback_days=365,
    )
    # Cliente 3 nunca debe aparecer (siempre inactivo en estos cortes)
    assert 3 not in dataset["Customer ID"].values
    # Cliente 1 y 2 deben aparecer en ambos snapshots (siguen activos)
    assert (dataset["Customer ID"] == 1).sum() == 2
    assert (dataset["Customer ID"] == 2).sum() == 2


def test_multi_snapshot_dataset_raises_if_no_active_customers():
    empty_like = pd.DataFrame(
        {
            "Customer ID": [1],
            "Invoice": ["100"],
            "InvoiceDate": pd.to_datetime(["2009-01-01"]),
            "StockCode": ["A"],
            "Quantity": [1],
            "Price": [10.0],
            "Country": ["United Kingdom"],
        }
    )
    snapshot_dates = pd.DatetimeIndex([pd.Timestamp("2011-06-15")])
    with pytest.raises(ValueError, match="Ningún snapshot"):
        build_multi_snapshot_dataset(
            empty_like, snapshot_dates=snapshot_dates, horizon_days=30
        )