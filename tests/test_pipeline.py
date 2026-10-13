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


# ---------------------------------------------------------------------------
# Features de devoluciones (cancellations opcional)
# ---------------------------------------------------------------------------

CUTOFF = pd.Timestamp("2011-02-05")
RETURN_COLUMNS = {"return_amount", "monetary_net", "return_ratio"}


def _cancellations(rows):
    return pd.DataFrame(
        rows, columns=["Customer ID", "InvoiceDate", "Quantity", "Price"]
    )


def test_dataset_without_cancellations_has_no_return_columns(sample_sales):
    dataset = build_dataset(sample_sales, cutoff_date=CUTOFF, horizon_days=90)
    assert RETURN_COLUMNS.isdisjoint(dataset.columns)


def test_dataset_with_cancellations_adds_return_columns(sample_sales):
    canc = _cancellations(
        [
            [1, "2011-01-20", -2, 5.0],  # antes del corte: 10
            [2, "2011-03-01", -1, 99.0],  # después del corte: no cuenta
        ]
    )
    dataset = build_dataset(
        sample_sales, cutoff_date=CUTOFF, horizon_days=90, cancellations=canc
    )
    assert RETURN_COLUMNS.issubset(dataset.columns)
    by_customer = dataset.set_index("Customer ID")
    assert by_customer.loc[1, "return_amount"] == pytest.approx(10.0)
    assert by_customer.loc[2, "return_amount"] == pytest.approx(0.0)
    assert by_customer.loc[3, "return_amount"] == pytest.approx(0.0)


def test_cancellations_do_not_change_existing_columns(sample_sales):
    canc = _cancellations([[1, "2011-01-20", -2, 5.0]])
    plain = build_dataset(sample_sales, cutoff_date=CUTOFF, horizon_days=90)
    with_returns = build_dataset(
        sample_sales, cutoff_date=CUTOFF, horizon_days=90, cancellations=canc
    )
    pd.testing.assert_frame_equal(plain, with_returns[plain.columns])


def test_future_cancellation_does_not_change_dataset(sample_sales):
    past = _cancellations([[1, "2011-01-20", -2, 5.0]])
    with_future = _cancellations(
        [[1, "2011-01-20", -2, 5.0], [1, "2011-02-10", -50, 20.0]]
    )
    a = build_dataset(
        sample_sales, cutoff_date=CUTOFF, horizon_days=90, cancellations=past
    )
    b = build_dataset(
        sample_sales, cutoff_date=CUTOFF, horizon_days=90, cancellations=with_future
    )
    pd.testing.assert_frame_equal(a, b)