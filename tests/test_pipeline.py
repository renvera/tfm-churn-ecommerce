import pandas as pd
import pytest

from src.features.pipeline import build_dataset


@pytest.fixture
def sample_sales():
    return pd.DataFrame(
        {
            "Customer ID": [1, 1, 2, 2, 3],
            "Invoice": ["100", "101", "200", "201", "300"],
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-01-15",
                    "2011-01-10",
                    "2011-04-01",  # compra futura del cliente 2
                    "2011-02-01",
                ]
            ),
            "StockCode": ["A", "B", "C", "D", "E"],
            "Quantity": [1, 2, 1, 3, 1],
            "Price": [10.0, 5.0, 20.0, 2.0, 7.5],
            "Country": [
                "United Kingdom",
                "United Kingdom",
                "Germany",
                "Germany",
                "Spain",
            ],
        }
    )


def test_dataset_has_one_row_per_customer(sample_sales):
    dataset = build_dataset(
        sample_sales, cutoff_date=pd.Timestamp("2011-02-05"), horizon_days=90
    )
    assert len(dataset) == dataset["Customer ID"].nunique()


def test_dataset_combines_features_and_churn_columns(sample_sales):
    dataset = build_dataset(
        sample_sales, cutoff_date=pd.Timestamp("2011-02-05"), horizon_days=90
    )
    expected_cols = {
        "Customer ID",
        "recency",
        "frequency",
        "monetary",
        "n_countries",
        "n_products",
        "avg_days_between_purchases",
        "churn",
    }
    assert expected_cols.issubset(set(dataset.columns))


def test_customer_with_future_purchase_is_not_churn(sample_sales):
    dataset = build_dataset(
        sample_sales, cutoff_date=pd.Timestamp("2011-02-05"), horizon_days=90
    )
    customer_2 = dataset[dataset["Customer ID"] == 2].iloc[0]
    assert customer_2["churn"] == 0  # vuelve el 2011-04-01, dentro de 90 días


def test_customer_without_future_purchase_is_churn(sample_sales):
    dataset = build_dataset(
        sample_sales, cutoff_date=pd.Timestamp("2011-02-05"), horizon_days=90
    )
    customer_3 = dataset[dataset["Customer ID"] == 3].iloc[0]
    assert customer_3["churn"] == 1  # no vuelve tras el corte