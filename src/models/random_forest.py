"""
src/models/random_forest.py

Pipeline de Random Forest: modelo no lineal y robusto para datos
tabulares, tercer paso del orden de comparación (Dummy -> Logistic ->
Random Forest -> XGBoost).
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.models.baseline import NUMERIC_FEATURES


def build_rf_preprocessing_pipeline(
    numeric_features: list[str] = NUMERIC_FEATURES,
) -> ColumnTransformer:
    """
    Random Forest no necesita escalado (invariante a la escala de las
    features), pero sí requiere imputar los nulos de
    avg_days_between_purchases.
    """
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("num", numeric_pipe, numeric_features)])


def build_random_forest_pipeline(
    numeric_features: list[str] = NUMERIC_FEATURES,
    n_estimators: int = 300,
    max_depth: int | None = 6,
    min_samples_leaf: int = 50,
    random_state: int = 42,
) -> Pipeline:
    """
    Pipeline completo: imputación + Random Forest con class_weight
    'balanced', para mantener el mismo criterio de tratamiento del
    desbalance usado en la Regresión Logística.
    """
    preprocessor = build_rf_preprocessing_pipeline(numeric_features)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )