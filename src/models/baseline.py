"""
src/models/baseline.py

Pipeline de preprocesamiento (imputación + escalado) y modelo baseline
(DummyClassifier) como referencia mínima antes de entrenar modelos reales.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Features numéricas del dataset cliente+snapshot. Se excluyen
# explícitamente: Customer ID, cutoff_date (no predictoras) y month
# (redundante con month_sin/month_cos, y como entero induce una escala
# falsa donde diciembre y enero quedan "lejos").
NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "n_countries",
    "n_products",
    "avg_days_between_purchases",
    "month_sin",
    "month_cos",
    "is_post_holiday",
]


def build_preprocessing_pipeline(
    numeric_features: list[str] = NUMERIC_FEATURES,
) -> ColumnTransformer:
    """
    Imputa nulos (mediana) y escala las features numéricas. La mediana
    se usa en vez de la media porque monetary/recency suelen tener cola
    larga (clientes con gasto o recencia muy altos).
    """
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer([("num", numeric_pipe, numeric_features)])


def build_dummy_pipeline(
    numeric_features: list[str] = NUMERIC_FEATURES,
    strategy: str = "stratified",
    random_state: int = 42,
) -> Pipeline:
    """
    Pipeline completo: preprocesamiento + DummyClassifier.

    strategy="stratified": genera predicciones respetando la proporción
    de clases observada en train, dando un suelo probabilístico
    (ROC-AUC ~0.5, PR-AUC ~prevalencia) contra el que comparar los
    modelos reales.
    """
    preprocessor = build_preprocessing_pipeline(numeric_features)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                DummyClassifier(strategy=strategy, random_state=random_state),
            ),
        ]
    )