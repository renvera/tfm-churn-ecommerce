import math

import pandas as pd
import pytest

from src.features.rfm import build_rfm_features


@pytest.fixture
def sample_sales():
    return pd.DataFrame(
        {
            "Customer ID": [1, 1, 1, 2, 2, 3],
            "Invoice": ["100", "101", "102", "200", "201", "300"],
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-01-15",
                    "2011-02-01",
                    "2011-01-10",
                    "2011-03-01",
                    "2011-02-15",
                ]
            ),
            "StockCode": ["A", "B", "A", "C", "D", "E"],
            "Quantity": [2, 1, 3, 5, 1, 2],
            "Price": [10.0, 5.0, 2.0, 3.0, 20.0, 7.5],
            "Country": [
                "United Kingdom",
                "United Kingdom",
                "France",
                "Germany",
                "Germany",
                "Spain",
            ],
        }
    )


def test_recency_is_days_since_last_purchase(sample_sales):
    features = build_rfm_features(
        sample_sales, cutoff_date=pd.Timestamp("2011-02-10")
    )
    customer_1 = features[features["Customer ID"] == 1].iloc[0]
    assert customer_1["recency"] == 9  # última compra observable: 2011-02-01


def test_frequency_counts_distinct_invoices(sample_sales):
    features = build_rfm_features(
        sample_sales, cutoff_date=pd.Timestamp("2011-02-10")
    )
    customer_1 = features[features["Customer ID"] == 1].iloc[0]
    assert customer_1["frequency"] == 3


def test_monetary_sums_quantity_times_unit_price(sample_sales):
    features = build_rfm_features(
        sample_sales, cutoff_date=pd.Timestamp("2011-02-10")
    )
    customer_1 = features[features["Customer ID"] == 1].iloc[0]
    expected = (2 * 10.0) + (1 * 5.0) + (3 * 2.0)
    assert customer_1["monetary"] == pytest.approx(expected)


def test_n_products_counts_distinct_stockcodes(sample_sales):
    features = build_rfm_features(
        sample_sales, cutoff_date=pd.Timestamp("2011-02-10")
    )
    customer_1 = features[features["Customer ID"] == 1].iloc[0]
    assert customer_1["n_products"] == 2  # StockCode A, B (A se repite)


def test_n_countries_counts_distinct_countries(sample_sales):
    features = build_rfm_features(
        sample_sales, cutoff_date=pd.Timestamp("2011-02-10")
    )
    customer_1 = features[features["Customer ID"] == 1].iloc[0]
    assert customer_1["n_countries"] == 2  # UK y Francia


def test_avg_days_between_purchases_with_multiple_orders(sample_sales):
    features = build_rfm_features(
        sample_sales, cutoff_date=pd.Timestamp("2011-02-10")
    )
    customer_1 = features[features["Customer ID"] == 1].iloc[0]
    # Intervalos: 01-01->01-15 (14 días), 01-15->02-01 (17 días) -> media 15.5
    assert customer_1["avg_days_between_purchases"] == pytest.approx(15.5)


def test_avg_days_between_purchases_is_nan_with_single_order(sample_sales):
    features = build_rfm_features(
        sample_sales, cutoff_date=pd.Timestamp("2011-01-20")
    )
    customer_2 = features[features["Customer ID"] == 2].iloc[0]
    assert customer_2["frequency"] == 1
    assert math.isnan(customer_2["avg_days_between_purchases"])


def test_customer_without_purchases_before_cutoff_is_excluded(sample_sales):
    features = build_rfm_features(
        sample_sales, cutoff_date=pd.Timestamp("2011-01-20")
    )
    assert 3 not in features["Customer ID"].values


def test_future_purchases_do_not_affect_features(sample_sales):
    """
    Test anti-leakage: añadir una compra posterior al corte no debe
    cambiar ninguna feature ya calculada para ese cliente.
    """
    cutoff = pd.Timestamp("2011-01-20")
    features_before = build_rfm_features(sample_sales, cutoff_date=cutoff)

    sales_with_future_purchase = pd.concat(
        [
            sample_sales,
            pd.DataFrame(
                {
                    "Customer ID": [1],
                    "Invoice": ["999"],
                    "InvoiceDate": pd.to_datetime(["2011-06-01"]),
                    "StockCode": ["Z"],
                    "Quantity": [100],
                    "Price": [999.0],
                    "Country": ["United Kingdom"],
                }
            ),
        ],
        ignore_index=True,
    )

    features_after = build_rfm_features(
        sales_with_future_purchase, cutoff_date=cutoff
    )

    pd.testing.assert_frame_equal(
        features_before.sort_values("Customer ID").reset_index(drop=True),
        features_after.sort_values("Customer ID").reset_index(drop=True),
    )


def test_missing_required_column_raises_error():
    invalid_sales = pd.DataFrame({"Customer ID": [1], "Invoice": ["100"]})
    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        build_rfm_features(invalid_sales, cutoff_date=pd.Timestamp("2011-01-01"))

def test_same_invoice_with_different_timestamps_is_one_purchase():
    """
    Regresión: una factura con líneas registradas en timestamps distintos
    (segundos de diferencia) debe contar como UNA sola compra, no como
    dos, y por tanto avg_days_between_purchases debe ser NaN.
    """
    sales = pd.DataFrame(
        {
            "Customer ID": [1, 1],
            "Invoice": ["500", "500"],
            "InvoiceDate": pd.to_datetime(
                ["2011-01-01 14:05:00", "2011-01-01 14:06:00"]
            ),
            "StockCode": ["A", "B"],
            "Quantity": [1, 2],
            "Price": [10.0, 5.0],
            "Country": ["United Kingdom", "United Kingdom"],
        }
    )
    features = build_rfm_features(
        sales, cutoff_date=pd.Timestamp("2011-02-01")
    )
    customer_1 = features[features["Customer ID"] == 1].iloc[0]
    assert customer_1["frequency"] == 1
    assert math.isnan(customer_1["avg_days_between_purchases"])