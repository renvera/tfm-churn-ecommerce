"""
src/models/xgboost_model.py

Pipeline de XGBoost: candidato principal de gradient boosting para
datos tabulares, último del orden de comparación (Dummy -> Logistic ->
Random Forest -> XGBoost).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.baseline import NUMERIC_FEATURES


def build_xgb_preprocessing_pipeline(
    numeric_features: list[str] = NUMERIC_FEATURES,
) -> ColumnTransformer:
    """
    XGBoost puede manejar NaN nativamente, pero mantenemos la imputación
    para que RF y XGBoost compartan el mismo preprocesamiento y la
    comparación sea consistente.
    """
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("num", numeric_pipe, numeric_features)])


def compute_scale_pos_weight(y) -> float:
    """
    scale_pos_weight = (nº negativos) / (nº positivos): el equivalente en
    XGBoost a class_weight='balanced' de scikit-learn.
    """
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return n_neg / n_pos


def build_xgboost_pipeline(
    numeric_features: list[str] = NUMERIC_FEATURES,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Pipeline completo: imputación + XGBoost.

    max_depth=4 y learning_rate=0.05 son valores conservadores desde el
    principio, dado lo visto con Random Forest: con pocas features y
    señal mayormente monótona, un modelo complejo sin regularizar
    sobreajusta fácilmente.
    """
    preprocessor = build_xgb_preprocessing_pipeline(numeric_features)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="aucpr",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )