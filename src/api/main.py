"""
src/api/main.py

Capa fina de servicio: valida la entrada (Pydantic), ejecuta el modelo y
devuelve probabilidad + nivel de riesgo. No registra payloads ni identificadores.

Arranque:
    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Ruta del modelo configurable con la variable de entorno CHURN_MODEL_PATH.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.predictor import ChurnPredictor, resolve_model_path
from src.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse

logger = logging.getLogger("churn_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    path = resolve_model_path()
    try:
        app.state.predictor = ChurnPredictor.load(path)
        logger.info("Modelo cargado: versión %s", app.state.predictor.version)
    except Exception:
        # La API arranca igualmente: /health lo reporta y /predict responde 503.
        logger.exception("No se pudo cargar el modelo desde %s", path)
    yield


app = FastAPI(
    title="E-commerce Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor = app.state.predictor
    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_version=predictor.version if predictor is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures, explain: bool = False) -> PredictionResponse:
    predictor = app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not available")
    try:
        probability = predictor.predict(payload)
        top_factors = predictor.explain(payload) if explain else None
    except ValueError as exc:
        logger.warning("Invalid prediction input: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid input")
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction service unavailable")

    return PredictionResponse(
        churn_probability=probability,
        risk_level=predictor.risk_level(probability),
        threshold=predictor.threshold,
        model_version=predictor.version,
        top_factors=top_factors,
    )