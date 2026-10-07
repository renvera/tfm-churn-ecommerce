# TFM Churn E-commerce — Bitácora 06: cierre técnico

**Fecha:** 5 de octubre de 2026

Esta bitácora consolida el tramo posterior a la Parte 5 y corrige la Parte 3, que describía XGBoost como "implementado, ejecución pendiente" cuando en realidad se creó y ejecutó después (resultados en el traspaso, sección 12).

## 1. Revisión del repositorio

Suite de tests inicial: 59 pasan. Se detectaron: ausencia de artefacto de modelo serializado, MLflow sin implementar, README y traspaso desactualizados, archivos `*.bak` y un paquete residual (`src/tfm_churn_ecommerce/`), versión de Python sin confirmar y dependencias no fijadas en el plan de despliegue.

## 2. Modelo final serializado (`scripts/train_final.py`)

Protocolo: RF en train → Platt en valid (modelo base congelado) → umbral recall >= 0,80 en valid; test solo evalúa. Genera `models/churn_model_1.0.0.joblib` y `models/churn_model_1.0.0_metadata.json`.

Resultado ejecutado en el equipo del autor: umbral **0,598547** (≈ 0,599); valid PR-AUC 0,8357 / ROC-AUC 0,7735 / recall 0,8004; test PR-AUC 0,8104 / ROC-AUC 0,7789 / precision 0,7495 / recall 0,8146 / F1 0,7807 / Brier 0,1863 / ECE 0,0563. La recarga del artefacto reproduce las predicciones (diferencia máxima 4,4e-16).

**Hallazgo:** el umbral depende de la versión de scikit-learn (1,8,0 → 0,5954; 1,9,x → 0,5985). El despliegue debe instalar desde `uv.lock`.

## 3. Monetary: bruto vs neto

Regla vigente: **bruto** (las cancelaciones se eliminan antes de calcular features). Criterio conceptual: el neto refleja mejor el valor del cliente, pero debía compararse antes de congelar la regla, como pedía el documento técnico.

Se implementó `src/features/returns.py` (devoluciones acumuladas hasta T, sin leakage, con 5 tests) y `scripts/experimento_monetary_neto.py` (3 variantes × 5 semillas, mismo protocolo). Resultado: el neto sube la PR-AUC de test de 0,8104 a 0,8114 (± 0,0007) y el ECE de 0,0584 a 0,0554: diferencia prácticamente irrelevante. Solo el 0,71 % de las filas tienen `monetary_net` <= 0 y la correlación con el bruto es 0,9928. **Decisión de trabajo:** mantener el bruto y reportar el neto como análisis de sensibilidad. Si se adoptara el neto habría que reentrenar, recalibrar, elegir un umbral nuevo y rehacer SHAP, análisis por segmento y presentación.

## 4. SHAP y análisis por segmento (resultados verificados)

- Importancia global: `monetary` ≈ `frequency` > `recency` > `n_products` > `month_cos` > `avg_days_between_purchases` > `month_sin` > `is_post_holiday` > `n_countries` (= 0).
- Wholesale (monetary >= p90 de train): con el umbral global, recall 0,333 en test frente a 0,862 en el resto. Se documenta como limitación (D6).

## 5. API con FastAPI + Pydantic

Archivos `src/api/schemas.py`, `predictor.py`, `main.py` y `tests/test_api.py` (21 tests). `GET /health` y `POST /predict` (con `?explain=true` para `top_factors` SHAP). Las features estacionales se derivan de `cutoff_date`; los campos desconocidos se rechazan; no se registran payloads. Verificado con `TestClient` y por HTTP (`uvicorn`) contra el artefacto real: la probabilidad coincide con la del modelo usado directamente, y SHAP cumple la propiedad aditiva (valor base + suma de contribuciones = probabilidad del RF base).

## 6. Docker

`Dockerfile` (instalación desde `uv.lock`, usuario sin privilegios, `HEALTHCHECK` sobre `model_loaded`) y `.dockerignore`. La sincronización desde el lockfile y la API sobre ese entorno se validaron; **el `docker build` queda por probar en local.**

## 7. Estado de tests

85 pasan (59 originales + 21 de API + 5 de devoluciones).

## 8. Pendientes

Probar el build de Docker; decidir MLflow vs `metadata.json`; confirmar la versión de Python; añadir `httpx` a `pyproject.toml` (`uv add httpx`); completar `.gitignore`; borrar `*.bak` y `src/tfm_churn_ecommerce/`; opcionalmente adelgazar la imagen moviendo jupyter/shap/xgboost a un extra `research`.
