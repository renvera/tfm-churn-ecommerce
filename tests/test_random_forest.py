import numpy as np
import pandas as pd
import pytest

from src.models.random_forest import build_random_forest_pipeline


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


def test_rf_pipeline_fits_and_predicts_proba(sample_data):
    X, y = sample_data
    pipeline = build_random_forest_pipeline(n_estimators=50)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert len(proba) == len(y)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_rf_pipeline_handles_nan(sample_data):
    X, y = sample_data
    pipeline = build_random_forest_pipeline(n_estimators=50)
    pipeline.fit(X, y)
    pipeline.predict(X)


def test_rf_learns_recency_signal(sample_data):
    """
    Con la señal sintética (más recency -> más churn), recency debe
    aparecer entre las features con mayor importancia.
    """
    X, y = sample_data
    pipeline = build_random_forest_pipeline(n_estimators=100)
    pipeline.fit(X, y)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    recency_idx = list(feature_names).index("num__recency")

    # recency debe estar entre las 2 features más importantes
    top_2 = set(pd.Series(importances, index=range(len(importances))).nlargest(2).index)
    assert recency_idx in top_2