"""
src/api/predictor.py

Carga el artefacto entrenado (modelo calibrado + metadatos) y predice.
El preprocesamiento viaja dentro del modelo, así que aquí solo se arma el
DataFrame con las columnas en el orden registrado en los metadatos.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.api.schemas import CustomerFeatures, RiskFactor
from src.config.settings import MODELS_DIR
from src.features.temporal import add_seasonal_features

DEFAULT_MODEL_PATH = MODELS_DIR / "churn_model_1.0.0.joblib"
MODEL_PATH_ENV = "CHURN_MODEL_PATH"

# Features que el cliente envía directamente; el resto se deriva de cutoff_date.
_INPUT_FEATURES = (
    "recency",
    "frequency",
    "monetary",
    "n_countries",
    "n_products",
    "avg_days_between_purchases",
)
_SEASONAL_FEATURES = ("month_sin", "month_cos", "is_post_holiday")


def resolve_model_path() -> Path:
    return Path(os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH))


class ChurnPredictor:
    def __init__(self, model, metadata: dict):
        self.model = model
        self.features: list[str] = list(metadata["features"])
        self.threshold: float = float(metadata["threshold"])
        self.version: str = str(metadata["model_version"])

        self._explainer = None  # TreeExplainer; se construye en la primera petición con explain

        unknown = set(self.features) - set(_INPUT_FEATURES) - set(_SEASONAL_FEATURES)
        if unknown:
            raise ValueError(f"El modelo usa features que la API no sabe construir: {sorted(unknown)}")

    @classmethod
    def load(cls, model_path: Path) -> "ChurnPredictor":
        meta_path = model_path.with_name(model_path.stem + "_metadata.json")
        model = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(model, metadata)

    def _build_frame(self, payload: CustomerFeatures) -> pd.DataFrame:
        cutoff = payload.cutoff_date or datetime.now(timezone.utc).date()
        row = {name: getattr(payload, name) for name in _INPUT_FEATURES}
        # None -> NaN: el imputador (mediana) del pipeline lo gestiona igual que en entrenamiento.
        if row["avg_days_between_purchases"] is None:
            row["avg_days_between_purchases"] = np.nan
        row["cutoff_date"] = pd.Timestamp(cutoff)
        frame = add_seasonal_features(pd.DataFrame([row]), date_col="cutoff_date")
        return frame[self.features]

    def predict(self, payload: CustomerFeatures) -> float:
        proba = self.model.predict_proba(self._build_frame(payload))[:, 1]
        return float(proba[0])

    def risk_level(self, probability: float) -> str:
        return "HIGH" if probability >= self.threshold else "LOW"

    def _get_explainer(self):
        """TreeExplainer del modelo BASE (imputación -> RF); Platt no altera las contribuciones."""
        if self._explainer is None:
            import shap

            base_pipeline = self.model.calibrated_classifiers_[0].estimator.estimator
            self._explainer = (
                shap.TreeExplainer(base_pipeline.named_steps["model"]),
                base_pipeline.named_steps["preprocessor"],
            )
        return self._explainer

    def explain(self, payload: CustomerFeatures, top_n: int = 3) -> list[RiskFactor]:
        """Las top_n variables con mayor |SHAP| para este cliente (clase churn=1)."""
        explainer, preprocessor = self._get_explainer()
        frame = self._build_frame(payload)
        x_trans = preprocessor.transform(frame)

        sv = explainer.shap_values(x_trans)
        # Compatibilidad entre versiones de shap: lista por clase, ndarray 3D o 2D
        if isinstance(sv, list):
            contrib = np.asarray(sv[1])[0]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            contrib = sv[0, :, 1]
        else:
            contrib = np.asarray(sv)[0]

        order = np.argsort(-np.abs(contrib))[:top_n]
        factors = []
        for i in order:
            raw_value = frame.iloc[0, i]
            factors.append(
                RiskFactor(
                    feature=self.features[i],
                    value=None if pd.isna(raw_value) else float(raw_value),
                    shap_value=float(contrib[i]),
                    direction="increases_risk" if contrib[i] > 0 else "decreases_risk",
                )
            )
        return factors