"""Tests de features de devoluciones: sin leakage y con la aritmética esperada."""

import pandas as pd
import pytest

from src.features.returns import add_return_features, get_cancellations

T = pd.Timestamp("2011-06-01")


def _dataset():
    return pd.DataFrame(
        {"Customer ID": [1, 2], "cutoff_date": [T, T], "monetary": [1000.0, 500.0]}
    )


def _canc(rows):
    return pd.DataFrame(
        rows, columns=["Customer ID", "InvoiceDate", "Quantity", "Price"]
    )


def test_net_subtracts_past_returns():
    canc = _canc([[1, "2011-05-10", -10, 5.0]])  # devuelve 50
    out = add_return_features(_dataset(), canc)
    assert out.loc[0, "return_amount"] == pytest.approx(50.0)
    assert out.loc[0, "monetary_net"] == pytest.approx(950.0)
    assert out.loc[0, "return_ratio"] == pytest.approx(0.05)
    assert out.loc[1, "monetary_net"] == pytest.approx(500.0)  # sin devoluciones


def test_future_cancellation_cannot_change_features():
    """Una cancelación posterior a T no puede alterar ninguna feature de X."""
    base = _canc([[1, "2011-05-10", -10, 5.0]])
    with_future = _canc([[1, "2011-05-10", -10, 5.0], [1, "2011-07-01", -100, 9.0]])
    pd.testing.assert_frame_equal(
        add_return_features(_dataset(), base),
        add_return_features(_dataset(), with_future),
    )


def test_cancellation_on_cutoff_day_is_included():
    out = add_return_features(_dataset(), _canc([[2, "2011-06-01", -2, 10.0]]))
    assert out.loc[1, "return_amount"] == pytest.approx(20.0)


def test_net_can_be_negative():
    out = add_return_features(_dataset(), _canc([[2, "2011-05-01", -100, 10.0]]))
    assert out.loc[1, "monetary_net"] == pytest.approx(-500.0)


def test_get_cancellations_filters_rows():
    raw = pd.DataFrame(
        {
            "Invoice": ["C100", "C101", "C102", "500"],
            "Customer ID": [1.0, None, 2.0, 3.0],
            "StockCode": ["A", "B", "TEST001", "C"],
        }
    )
    got = get_cancellations(raw)
    assert list(got["Invoice"]) == ["C100"]