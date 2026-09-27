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