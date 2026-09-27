import pandas as pd
import pytest

from src.labeling.churn import build_churn_labels


@pytest.fixture
def sample_sales():
    return pd.DataFrame(
        {
            "Customer ID": [1, 1, 2, 2, 3, 4],
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-02-15",
                    "2011-01-10",
                    "2011-04-01",
                    "2011-02-01",
                    "2011-03-01",
                ]
            ),
            "Invoice": ["100", "101", "200", "201", "300", "400"],
        }
    )


def test_customer_returning_within_horizon_is_not_churn(sample_sales):
    labels = build_churn_labels(
        sample_sales,
        cutoff_date=pd.Timestamp("2011-03-01"),
        horizon_days=90,
    )

    customer_1 = labels[labels["Customer ID"] == 1].iloc[0]

    assert customer_1["future_purchase_count"] == 0
    assert customer_1["churn"] == 1


def test_customer_returning_within_horizon_is_not_churn():
    sales = pd.DataFrame(
        {
            "Customer ID": [1, 1],
            "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-03-15"]),
            "Invoice": ["100", "101"],
        }
    )

    labels = build_churn_labels(
        sales,
        cutoff_date=pd.Timestamp("2011-03-01"),
        horizon_days=90,
    )

    customer_1 = labels.iloc[0]

    assert customer_1["future_purchase_count"] == 1
    assert customer_1["churn"] == 0


def test_purchase_after_horizon_is_not_counted():
    sales = pd.DataFrame(
        {
            "Customer ID": [1, 1],
            "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-07-01"]),
            "Invoice": ["100", "101"],
        }
    )

    labels = build_churn_labels(
        sales,
        cutoff_date=pd.Timestamp("2011-03-01"),
        horizon_days=90,
    )

    customer_1 = labels.iloc[0]

    assert customer_1["future_purchase_count"] == 0
    assert customer_1["churn"] == 1


def test_cutoff_purchase_is_observation_purchase():
    sales = pd.DataFrame(
        {
            "Customer ID": [1],
            "InvoiceDate": pd.to_datetime(["2011-03-01"]),
            "Invoice": ["100"],
        }
    )

    labels = build_churn_labels(
        sales,
        cutoff_date=pd.Timestamp("2011-03-01"),
        horizon_days=90,
    )

    assert len(labels) == 1
    assert labels.iloc[0]["churn"] == 1


def test_invalid_horizon_raises_error(sample_sales):
    with pytest.raises(ValueError, match="mayor que cero"):
        build_churn_labels(
            sample_sales,
            cutoff_date=pd.Timestamp("2011-03-01"),
            horizon_days=0,
        )


def test_missing_required_column_raises_error():
    invalid_sales = pd.DataFrame(
        {
            "Customer ID": [1],
            "Invoice": ["100"],
        }
    )

    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        build_churn_labels(
            invalid_sales,
            cutoff_date=pd.Timestamp("2011-03-01"),
            horizon_days=90,
        )