import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_model import (
    build_xgboost_pipeline,
    compute_scale_pos_weight,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 300
    recency = rng.integers(0, 300, n)
    X = pd.DataFrame(
        {
            "recency": recency,
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
    prob_churn = 1 / (1 + np.exp(-(recency - 150) / 50))
    y = (rng.random(n) < prob_churn).astype(int)
    return X, y


def test_compute_scale_pos_weight():
    y = pd.Series([0, 0, 0, 1])  # 3 negativos, 1 positivo
    assert compute_scale_pos_weight(y) == pytest.approx(3.0)


def test_xgboost_pipeline_fits_and_predicts_proba(sample_data):
    X, y = sample_data
    weight = compute_scale_pos_weight(pd.Series(y))
    pipeline = build_xgboost_pipeline(n_estimators=50, scale_pos_weight=weight)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert len(proba) == len(y)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_xgboost_pipeline_handles_nan(sample_data):
    X, y = sample_data
    pipeline = build_xgboost_pipeline(n_estimators=50)
    pipeline.fit(X, y)
    pipeline.predict(X)