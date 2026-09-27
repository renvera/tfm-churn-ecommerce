import numpy as np
import pandas as pd
import pytest

from src.features.temporal import add_seasonal_features


@pytest.fixture
def sample_dataset():
    return pd.DataFrame(
        {
            "Customer ID": [1, 2, 3, 4],
            "cutoff_date": pd.to_datetime(
                ["2011-01-01", "2011-06-01", "2010-12-01", "2011-07-01"]
            ),
            "churn": [1, 0, 1, 0],
        }
    )


def test_month_is_extracted_correctly(sample_dataset):
    result = add_seasonal_features(sample_dataset)
    assert list(result["month"]) == [1, 6, 12, 7]


def test_is_post_holiday_flags_dec_to_mar(sample_dataset):
    result = add_seasonal_features(sample_dataset)
    assert list(result["is_post_holiday"]) == [1, 0, 1, 0]


def test_month_sin_cos_are_cyclical():
    dataset = pd.DataFrame(
        {
            "Customer ID": [1, 2],
            "cutoff_date": pd.to_datetime(["2011-01-01", "2011-12-01"]),
            "churn": [0, 0],
        }
    )
    result = add_seasonal_features(dataset)
    # Enero y diciembre deben quedar "cerca" en el espacio cíclico
    jan = result.iloc[0][["month_sin", "month_cos"]].values
    dec = result.iloc[1][["month_sin", "month_cos"]].values
    distance = np.linalg.norm(jan - dec)
    assert distance < 0.6  # mucho más cerca que enero vs junio (~2.0)


def test_original_dataframe_is_not_mutated(sample_dataset):
    original_columns = list(sample_dataset.columns)
    add_seasonal_features(sample_dataset)
    assert list(sample_dataset.columns) == original_columns