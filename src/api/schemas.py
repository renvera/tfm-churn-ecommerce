"""
src/api/schemas.py

Contratos de entrada/salida de la API. La entrada son las features RFM ya
calculadas por cliente (el cálculo desde transacciones queda fuera del alcance
de la API). Las features estacionales (month_sin, month_cos, is_post_holiday)
NO se piden al cliente: se derivan de ``cutoff_date`` para evitar valores
incoherentes entre sí.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Features de un cliente a la fecha de corte T (solo información <= T)."""

    model_config = ConfigDict(
        extra="forbid",  # un campo mal escrito falla en vez de ignorarse en silencio
        json_schema_extra={
            "example": {
                "recency": 45,
                "frequency": 6,
                "monetary": 1850.40,
                "n_countries": 1,
                "n_products": 38,
                "avg_days_between_purchases": 62.5,
                "cutoff_date": "2011-06-01",
            }
        },
    )

    recency: float = Field(ge=0, description="Días desde la última compra hasta el corte.")
    frequency: float = Field(ge=1, description="Número de facturas distintas hasta el corte.")
    monetary: float = Field(
        ge=0, description="Gasto total acumulado hasta el corte (Quantity * Price, sin cancelaciones)."
    )
    n_countries: float = Field(ge=1, description="Países distintos registrados.")
    n_products: float = Field(ge=1, description="Productos distintos comprados.")
    avg_days_between_purchases: float | None = Field(
        default=None,
        ge=0,
        description="Intervalo medio entre facturas. Null si el cliente solo tiene una compra.",
    )
    cutoff_date: date | None = Field(
        default=None,
        description="Fecha de corte T (YYYY-MM-DD). Si se omite, se usa la fecha actual (UTC).",
    )


class RiskFactor(BaseModel):
    """Contribución SHAP de una variable a la predicción de este cliente."""

    feature: str
    value: float | None = Field(description="Valor de la variable usado por el modelo (null si faltaba).")
    shap_value: float = Field(
        description="Contribución al riesgo en unidades de probabilidad del modelo base (antes de calibrar)."
    )
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1, description="Probabilidad calibrada (Platt).")
    risk_level: Literal["HIGH", "LOW"] = Field(
        description="HIGH si la probabilidad alcanza el umbral operativo del modelo; LOW si no."
    )
    threshold: float = Field(description="Umbral operativo (recall >= 0.80 en validación).")
    model_version: str
    top_factors: list[RiskFactor] | None = Field(
        default=None,
        description="Variables con mayor contribución (SHAP). Solo si se pide con ?explain=true.",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None