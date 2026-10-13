"""Las etiquetas de churn de distintos horizontes deben estar anidadas."""

import pandas as pd
import pytest

from src.labeling.churn import build_churn_labels

CUTOFF = pd.Timestamp("2011-01-01")


@pytest.fixture
def sales():
    """Cada cliente compra antes del corte; después, a distinta distancia."""
    cutoff = CUTOFF
    rows = [
        ("A", cutoff - pd.Timedelta(days=10)),
        ("A", cutoff + pd.Timedelta(days=30)),  # vuelve a los 30 d
        ("B", cutoff - pd.Timedelta(days=10)),
        ("B", cutoff + pd.Timedelta(days=75)),  # vuelve a los 75 d
        ("C", cutoff - pd.Timedelta(days=10)),
        ("C", cutoff + pd.Timedelta(days=100)),  # vuelve a los 100 d
        ("D", cutoff - pd.Timedelta(days=10)),  # no vuelve
    ]
    return pd.DataFrame(
        {
            "Customer ID": [r[0] for r in rows],
            "InvoiceDate": [r[1] for r in rows],
            "Invoice": [str(i) for i in range(len(rows))],
        }
    )


def _churn(sales, horizon):
    labels = build_churn_labels(sales, cutoff_date=CUTOFF, horizon_days=horizon)
    return labels.set_index("Customer ID")["churn"]


def test_churn_by_horizon_matches_expected(sales):
    assert dict(_churn(sales, 60)) == {"A": 0, "B": 1, "C": 1, "D": 1}
    assert dict(_churn(sales, 90)) == {"A": 0, "B": 0, "C": 1, "D": 1}
    assert dict(_churn(sales, 120)) == {"A": 0, "B": 0, "C": 0, "D": 1}


def test_churn_is_nested_across_horizons(sales):
    """Un horizonte más largo nunca crea churn nuevo: quien vuelve dentro de
    h días también vuelve dentro de cualquier h' > h."""
    c60, c90, c120 = (_churn(sales, h) for h in (60, 90, 120))
    assert (c60 >= c90).all()
    assert (c90 >= c120).all()
