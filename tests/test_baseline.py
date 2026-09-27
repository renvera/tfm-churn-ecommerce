import numpy as np
import pandas as pd
import pytest

from src.models.baseline import build_dummy_pipeline


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(
        {
            "recency": rng.integers(0, 300, n),
            "frequency": rng.integers(1, 20, n),
            "monetary": rng.uniform(10, 5000, n),
            "n_countries": rng.integers(1, 3, n),
            "n_products": rng.integers(1, 100, n),
            "avg_days_between_purchases": np.where(
                rng.random(n) < 0.3, np.nan, rng.uniform(5, 90, n)
            ),
            "month_sin": rng.uniform(-1, 1, n),
            "month_cos": rng.uniform(-1, 1, n),
            "is_post_holiday": rng.integers(0, 2, n),
        }
    )
    y = rng.integers(0, 2, n)
    return X, y


def test_dummy_pipeline_fits_and_predicts_proba(sample_data):
    X, y = sample_data
    pipeline = build_dummy_pipeline()
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert len(proba) == len(y)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_dummy_pipeline_handles_nan_in_numeric_features(sample_data):
    X, y = sample_data
    assert X["avg_days_between_purchases"].isna().any()
    pipeline = build_dummy_pipeline()
    # No debe lanzar error pese a los nulos (SimpleImputer los resuelve)
    pipeline.fit(X, y)
    pipeline.predict(X)