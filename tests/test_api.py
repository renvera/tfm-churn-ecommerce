"""
tests/test_api.py

Pruebas de la API con un artefacto sintético mínimo (RF + Platt) generado en un
directorio temporal: no dependen de models/ ni del dataset real.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.predictor import MODEL_PATH_ENV
from src.evaluation.calibration import calibrate_prefit
from src.features.temporal import add_seasonal_features
from src.models.baseline import NUMERIC_FEATURES
from src.models.random_forest import build_random_forest_pipeline

THRESHOLD = 0.6
VALID_PAYLOAD = {
    "recency": 45,
    "frequency": 6,
    "monetary": 1850.4,
    "n_countries": 1,
    "n_products": 38,
    "avg_days_between_purchases": 62.5,
    "cutoff_date": "2011-06-01",
}


def _synthetic_frame(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "recency": rng.integers(0, 365, n),
            "frequency": rng.integers(1, 30, n),
            "monetary": rng.gamma(2.0, 800.0, n),
            "n_countries": rng.integers(1, 3, n),
            "n_products": rng.integers(1, 120, n),
            "avg_days_between_purchases": np.where(
                rng.random(n) < 0.2, np.nan, rng.gamma(2.0, 30.0, n)
            ),
            "cutoff_date": pd.to_datetime("2011-01-01") + pd.to_timedelta(rng.integers(0, 240, n), "D"),
        }
    )
    df = add_seasonal_features(df, date_col="cutoff_date")
    y = ((df["recency"] / 365 + rng.normal(0, 0.3, n)) > 0.5).astype(int).to_numpy()
    return df, y


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    train, y_tr = _synthetic_frame(1500, seed=1)
    valid, y_va = _synthetic_frame(800, seed=2)
    pipe = build_random_forest_pipeline(n_estimators=30, min_samples_leaf=10)
    pipe.fit(train[NUMERIC_FEATURES], y_tr)
    model = calibrate_prefit(pipe, method="sigmoid")
    model.fit(valid[NUMERIC_FEATURES], y_va)

    model_path = out / "churn_model_test.joblib"
    joblib.dump(model, model_path)
    (out / "churn_model_test_metadata.json").write_text(
        json.dumps(
            {"model_version": "test-1", "features": NUMERIC_FEATURES, "threshold": THRESHOLD}
        ),
        encoding="utf-8",
    )
    return model_path, model


@pytest.fixture
def client(artifact, monkeypatch):
    monkeypatch.setenv(MODEL_PATH_ENV, str(artifact[0]))
    with TestClient(app) as c:
        yield c


def test_health_with_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "test-1"}


def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"HIGH", "LOW"}
    assert body["threshold"] == THRESHOLD
    assert body["model_version"] == "test-1"


def test_probability_matches_pipeline_directly(client, artifact):
    """La API debe dar exactamente la misma probabilidad que el modelo usado a mano."""
    _, model = artifact
    row = pd.DataFrame([{**VALID_PAYLOAD, "cutoff_date": pd.Timestamp("2011-06-01")}])
    expected = model.predict_proba(
        add_seasonal_features(row, date_col="cutoff_date")[NUMERIC_FEATURES]
    )[:, 1][0]
    got = client.post("/predict", json=VALID_PAYLOAD).json()["churn_probability"]
    assert got == pytest.approx(expected, abs=1e-12)


def test_risk_level_is_anchored_to_threshold(client):
    body = client.post("/predict", json=VALID_PAYLOAD).json()
    expected = "HIGH" if body["churn_probability"] >= THRESHOLD else "LOW"
    assert body["risk_level"] == expected


def test_null_avg_days_between_purchases_is_accepted(client):
    payload = {**VALID_PAYLOAD, "avg_days_between_purchases": None, "frequency": 1}
    assert client.post("/predict", json=payload).status_code == 200
    payload.pop("avg_days_between_purchases")
    assert client.post("/predict", json=payload).status_code == 200


def test_cutoff_date_is_optional(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "cutoff_date"}
    assert client.post("/predict", json=payload).status_code == 200


def test_seasonality_changes_prediction_inputs(client, artifact):
    """Cambiar solo el mes del corte debe poder cambiar la probabilidad (features derivadas)."""
    p_jun = client.post("/predict", json=VALID_PAYLOAD).json()["churn_probability"]
    p_jan = client.post(
        "/predict", json={**VALID_PAYLOAD, "cutoff_date": "2011-01-15"}
    ).json()["churn_probability"]
    assert p_jun != p_jan


@pytest.mark.parametrize(
    "field,value",
    [
        ("recency", -1),
        ("frequency", 0),
        ("monetary", -10),
        ("n_products", 0),
        ("recency", "abc"),
        ("cutoff_date", "no-es-fecha"),
    ],
)
def test_invalid_values_return_422(client, field, value):
    r = client.post("/predict", json={**VALID_PAYLOAD, field: value})
    assert r.status_code == 422


def test_missing_required_field_returns_422(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "recency"}
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_fields_are_rejected(client):
    """Evita que se cuelen identificadores (p. ej. Customer ID) que la API no necesita."""
    r = client.post("/predict", json={**VALID_PAYLOAD, "customer_id": 12346})
    assert r.status_code == 422


def test_model_unavailable_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "no_existe.joblib"))
    with TestClient(app) as c:
        h = c.get("/health")
        assert h.status_code == 200
        assert h.json() == {"status": "degraded", "model_loaded": False, "model_version": None}
        r = c.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 503
        assert r.json()["detail"] == "Model not available"


# ---------------------------------------------------------------------------
# Explicaciones SHAP (?explain=true)
# ---------------------------------------------------------------------------
def test_explain_is_off_by_default(client):
    assert client.post("/predict", json=VALID_PAYLOAD).json()["top_factors"] is None


def test_explain_returns_top_three_factors(client):
    body = client.post("/predict?explain=true", json=VALID_PAYLOAD).json()
    factors = body["top_factors"]
    assert len(factors) == 3
    assert {f["feature"] for f in factors} <= set(NUMERIC_FEATURES)
    magnitudes = [abs(f["shap_value"]) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for f in factors:
        assert f["direction"] == ("increases_risk" if f["shap_value"] > 0 else "decreases_risk")


def test_explain_direction_matches_data_generating_rule(client):
    """En los datos sintéticos el riesgo sube con la recencia: debe reflejarse en SHAP."""
    payload = {**VALID_PAYLOAD, "recency": 360}
    factors = client.post("/predict?explain=true", json=payload).json()["top_factors"]
    recency = next(f for f in factors if f["feature"] == "recency")
    assert recency["direction"] == "increases_risk"


def test_explain_does_not_change_probability(client):
    plain = client.post("/predict", json=VALID_PAYLOAD).json()["churn_probability"]
    explained = client.post("/predict?explain=true", json=VALID_PAYLOAD).json()["churn_probability"]
    assert plain == explained


def test_shap_values_are_additive(artifact):
    """valor esperado + suma de SHAP == probabilidad del RF base (propiedad de SHAP)."""
    from src.api.predictor import ChurnPredictor
    from src.api.schemas import CustomerFeatures

    model_path, model = artifact
    predictor = ChurnPredictor.load(model_path)
    payload = CustomerFeatures(**VALID_PAYLOAD)

    factors = predictor.explain(payload, top_n=len(NUMERIC_FEATURES))
    explainer, _ = predictor._get_explainer()
    base_value = np.atleast_1d(explainer.expected_value)[-1]
    base_pipeline = model.calibrated_classifiers_[0].estimator.estimator
    expected = base_pipeline.predict_proba(predictor._build_frame(payload))[:, 1][0]

    assert base_value + sum(f.shap_value for f in factors) == pytest.approx(expected, abs=1e-6)