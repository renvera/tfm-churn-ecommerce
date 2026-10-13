import pandas as pd
import pytest

from src.splitting.temporal_split import summarize_split, temporal_split


@pytest.fixture
def sample_dataset():
    return pd.DataFrame(
        {
            "Customer ID": [1, 2, 1, 3, 2, 4],
            "cutoff_date": pd.to_datetime(
                [
                    "2011-01-01",
                    "2011-01-01",
                    "2011-03-01",
                    "2011-03-01",
                    "2011-06-01",
                    "2011-06-01",
                ]
            ),
            "recency": [10, 20, 5, 15, 30, 8],
            "churn": [0, 1, 0, 1, 1, 0],
        }
    )


def test_split_respects_temporal_order(sample_dataset):
    train, valid, test = temporal_split(
        sample_dataset,
        date_col="cutoff_date",
        train_end=pd.Timestamp("2011-01-01"),
        valid_end=pd.Timestamp("2011-03-01"),
    )
    assert train["cutoff_date"].max() <= pd.Timestamp("2011-01-01")
    assert valid["cutoff_date"].min() > pd.Timestamp("2011-01-01")
    assert valid["cutoff_date"].max() <= pd.Timestamp("2011-03-01")
    assert test["cutoff_date"].min() > pd.Timestamp("2011-03-01")


def test_split_covers_all_rows_without_overlap(sample_dataset):
    train, valid, test = temporal_split(
        sample_dataset,
        date_col="cutoff_date",
        train_end=pd.Timestamp("2011-01-01"),
        valid_end=pd.Timestamp("2011-03-01"),
    )
    assert len(train) + len(valid) + len(test) == len(sample_dataset)


def test_same_customer_can_appear_in_multiple_splits(sample_dataset):
    """
    Documenta el comportamiento esperado: cliente 1 y 2 aparecen en más
    de un split porque tienen varios snapshots. No es un bug.
    """
    train, valid, test = temporal_split(
        sample_dataset,
        date_col="cutoff_date",
        train_end=pd.Timestamp("2011-01-01"),
        valid_end=pd.Timestamp("2011-03-01"),
    )
    assert 1 in train["Customer ID"].values
    assert 1 in valid["Customer ID"].values
    assert 2 in train["Customer ID"].values
    assert 2 in test["Customer ID"].values


def test_invalid_boundaries_raise_error(sample_dataset):
    with pytest.raises(ValueError):
        temporal_split(
            sample_dataset,
            date_col="cutoff_date",
            train_end=pd.Timestamp("2011-06-01"),
            valid_end=pd.Timestamp("2011-01-01"),
        )


def test_summarize_split_returns_one_row_per_split(sample_dataset):
    train, valid, test = temporal_split(
        sample_dataset,
        date_col="cutoff_date",
        train_end=pd.Timestamp("2011-01-01"),
        valid_end=pd.Timestamp("2011-03-01"),
    )
    summary = summarize_split(train, valid, test)
    assert list(summary["split"]) == ["train", "valid", "test"]


# ---------------------------------------------------------------------------
# Embargo (gap_days)
# ---------------------------------------------------------------------------

HORIZON = 90


@pytest.fixture
def monthly_dataset():
    """Snapshots mensuales de 2011-01-01 a 2011-12-01, dos clientes por corte."""
    dates = pd.date_range("2011-01-01", "2011-12-01", freq="MS")
    return pd.DataFrame(
        {
            "Customer ID": [c for _ in dates for c in (1, 2)],
            "cutoff_date": [d for d in dates for _ in (1, 2)],
            "recency": 1,
            "churn": 0,
        }
    )


def _split(dataset, gap_days):
    return temporal_split(
        dataset,
        date_col="cutoff_date",
        train_end=pd.Timestamp("2011-03-01"),
        valid_end=pd.Timestamp("2011-08-01"),
        gap_days=gap_days,
    )


def test_gap_zero_keeps_previous_behaviour(monthly_dataset):
    default = temporal_split(
        monthly_dataset,
        date_col="cutoff_date",
        train_end=pd.Timestamp("2011-03-01"),
        valid_end=pd.Timestamp("2011-08-01"),
    )
    explicit = _split(monthly_dataset, gap_days=0)
    for a, b in zip(default, explicit):
        pd.testing.assert_frame_equal(a, b)
    assert sum(len(s) for s in explicit) == len(monthly_dataset)


def test_gap_drops_snapshots_inside_embargo(monthly_dataset):
    train, valid, test = _split(monthly_dataset, gap_days=HORIZON)
    # train_end + 90 d = 2011-05-30 -> valid empieza en el corte 2011-06-01
    assert train["cutoff_date"].max() == pd.Timestamp("2011-03-01")
    assert valid["cutoff_date"].min() == pd.Timestamp("2011-06-01")
    assert valid["cutoff_date"].max() == pd.Timestamp("2011-08-01")
    # valid_end + 90 d = 2011-10-30 -> test empieza en el corte 2011-11-01
    assert test["cutoff_date"].min() == pd.Timestamp("2011-11-01")
    # Se descartan los cortes de abril, mayo, septiembre y octubre (2 filas c/u)
    assert len(train) + len(valid) + len(test) == len(monthly_dataset) - 8


def test_label_windows_do_not_overlap_between_splits(monthly_dataset):
    """
    La etiqueta de un corte T mira a (T, T + HORIZON]. El primer corte del
    split siguiente debe quedar fuera de esa ventana.
    """
    train, valid, test = _split(monthly_dataset, gap_days=HORIZON)
    horizon = pd.Timedelta(days=HORIZON)
    assert valid["cutoff_date"].min() >= train["cutoff_date"].max() + horizon
    assert test["cutoff_date"].min() >= valid["cutoff_date"].max() + horizon


def test_cutoff_exactly_at_horizon_is_kept():
    """2011-01-01 + 90 d = 2011-04-01: ventanas disjuntas, no se descarta."""
    dataset = pd.DataFrame(
        {
            "Customer ID": [1, 1, 1, 1],
            "cutoff_date": pd.to_datetime(
                ["2011-01-01", "2011-03-01", "2011-04-01", "2011-07-01"]
            ),
            "churn": [0, 0, 0, 0],
        }
    )
    train, valid, test = temporal_split(
        dataset,
        date_col="cutoff_date",
        train_end=pd.Timestamp("2011-01-01"),
        valid_end=pd.Timestamp("2011-04-01"),
        gap_days=HORIZON,
    )
    assert list(train["cutoff_date"]) == [pd.Timestamp("2011-01-01")]
    assert list(valid["cutoff_date"]) == [pd.Timestamp("2011-04-01")]
    assert list(test["cutoff_date"]) == [pd.Timestamp("2011-07-01")]


def test_without_gap_label_windows_do_overlap(monthly_dataset):
    """Documenta el problema que corrige el embargo."""
    train, valid, _ = _split(monthly_dataset, gap_days=0)
    horizon = pd.Timedelta(days=HORIZON)
    assert valid["cutoff_date"].min() <= train["cutoff_date"].max() + horizon


def test_negative_gap_raises_error(monthly_dataset):
    with pytest.raises(ValueError):
        _split(monthly_dataset, gap_days=-1)


def test_gap_too_large_for_valid_window_raises_error(monthly_dataset):
    # train_end + 200 d queda después de valid_end -> valid vacío
    with pytest.raises(ValueError):
        _split(monthly_dataset, gap_days=200)