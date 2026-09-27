"""
src/models/logistic.py

Pipeline de Regresión Logística: primer modelo con capacidad real de
discriminación, usado como referencia interpretable antes de pasar a
modelos no lineales (Random Forest, XGBoost).
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.baseline import NUMERIC_FEATURES, build_preprocessing_pipeline


def build_logistic_pipeline(
    numeric_features: list[str] = NUMERIC_FEATURES,
    random_state: int = 42,
) -> Pipeline:
    """
    Pipeline completo: preprocesamiento + Regresión Logística con
    class_weight='balanced' (el desbalance se trata por peso de clase,
    no por resampling, como primer paso según el documento maestro).
    """
    preprocessor = build_preprocessing_pipeline(numeric_features)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )