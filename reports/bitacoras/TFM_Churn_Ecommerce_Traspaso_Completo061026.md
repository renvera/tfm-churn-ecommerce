# TFM Churn E-commerce — Documento de traspaso completo

> **Actualizado el 5-oct-2026.** El bloque técnico está cerrado: calibración, umbral, SHAP, análisis por segmento, modelo final serializado, API y Dockerfile. Las secciones 13 y 14 reflejan el estado actual; las secciones 2-12 conservan el detalle histórico.

Este documento resume TODO el estado del proyecto `tfm-churn-ecommerce` hasta el momento, para poder continuar el trabajo con otro asistente de IA sin perder contexto. No es el chat literal — es un resumen técnico completo con decisiones, código implementado, resultados obtenidos y próximos pasos.

---

## 0. Contexto del proyecto

- **Proyecto:** TFM (Trabajo Fin de Máster) — Máster Ejecutivo en Big Data & Business Analytics, EUDE Business School.
- **Título:** Predicción de churn (abandono de clientes) en e-commerce.
- **Repositorio:** `tfm-churn-ecommerce`.
- **Dataset:** UCI Online Retail II (ID 502) — 1.067.371 filas, transacciones dic-2009 a dic-2011, negocio de venta de artículos de regalo (mayormente wholesale/B2C mixto).
- **Gestión de dependencias:** `uv` + `pyproject.toml`.
- **Entorno:** Python 3.12 (nota: una bitácora anterior mencionó 3.14.3 en otra sesión; verificar cuál es el entorno activo real).
- **Documentos de referencia del proyecto** (no incluidos aquí, pero mencionados constantemente): "Documento Maestro del TFM v3.0" (metodología, literatura, criterios) y "Documento Técnico del TFM" (arquitectura, código, MLOps).
- **Existen 3 bitácoras previas en formato PDF/Markdown** (Partes 1, 2 y 3) que documentan el avance paso a paso; este documento las consolida y actualiza.

---

## 1. Estructura del repositorio (real, validada)

```
tfm-churn-ecommerce/
├── data/
│   ├── raw/                  # online_retail_II.xlsx (dataset original)
│   ├── interim/               # cachés parquet
│   └── processed/
├── notebooks/
│   └── 01_data_audit.ipynb
├── src/
│   ├── config/settings.py
│   ├── data/
│   │   ├── loader.py
│   │   └── cleaning.py
│   ├── labeling/
│   │   └── churn.py
│   ├── features/
│   │   ├── rfm.py
│   │   ├── pipeline.py
│   │   ├── snapshots.py
│   │   ├── temporal.py
│   │   └── returns.py
│   ├── splitting/              # (NOTA: no "split/" como en el plan original)
│   │   └── temporal_split.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── logistic.py
│   │   ├── random_forest.py
│   │   └── xgboost_model.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   └── threshold.py
│   ├── explainability/         # vacío: SHAP vive en scripts/analyze_shap.py y en src/api/predictor.py
│   └── api/
│       ├── schemas.py          # CustomerFeatures, PredictionResponse, RiskFactor, HealthResponse
│       ├── predictor.py        # carga joblib + metadata, predicción y SHAP
│       └── main.py             # /health y /predict (?explain=true)
├── tests/
│   ├── test_cleaning.py
│   ├── test_churn.py
│   ├── test_rfm.py
│   ├── test_pipeline.py
│   ├── test_snapshots.py
│   ├── test_temporal.py
│   ├── test_temporal_split.py
│   ├── test_baseline.py
│   ├── test_metrics.py
│   ├── test_logistic.py
│   ├── test_random_forest.py
│   ├── test_xgboost_model.py
│   ├── test_calibration.py
│   ├── test_threshold.py
│   ├── test_returns.py
│   └── test_api.py
├── scripts/
│   ├── check_structure.py
│   ├── verify_setup.py
│   ├── train_baseline_logistic.py
│   ├── train_all_models.py
│   ├── evaluate_thresholds.py
│   ├── analyze_shap.py
│   ├── analisis_wholesale.py
│   ├── control_cv_shuffle.py
│   ├── diagnostico_xgboost.py
│   ├── experimento_monetary_neto.py
│   └── train_final.py
├── models/                     # churn_model_1.0.0.joblib + _metadata.json
├── reports/
├── pyproject.toml
└── README.md
```

---

## 2. Datos: carga y limpieza (`src/data/`)

**`loader.py` — `load_online_retail(file_path, use_cache=True)`:**
- Lee las 2 hojas del Excel UCI, las concatena, cachea en parquet.
- Columnas `Invoice`, `StockCode`, `Description` casteadas a string (pyarrow rechazaba tipos mixtos).
- Resultado: 1.067.371 filas, 9 columnas (incluye `source_sheet`). Nulos: `Customer ID` → 243.007, `Description` → 4.382.
- **Columnas reales del dataset:** `['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country', 'source_sheet', 'is_cancellation']` — ojo, es `Price`, NO `UnitPrice` (nombre heredado erróneamente del dataset antiguo "Online Retail" 2010; causó un bug real, ver sección 5).

**`cleaning.py`:**
- `add_cancellation_flag(df)` — marca `is_cancellation` (prefijo `C` en `Invoice`).
- `filter_valid_customers(df)` — elimina `Customer ID` nulo.
- `remove_system_noise(df)` — elimina `StockCode == 'TEST001'`.
- `get_clean_sales(df)` — pipeline completo (las 3 reglas anteriores).
- `build_customer_profile(df_valid, wholesale_percentile=0.99)` — perfil por cliente + flag `is_wholesale`.

**Hallazgos de la auditoría:**
- Quantity negativa sin prefijo `C`: 3.457 filas, 100% con `Customer ID` nulo (se eliminan solas al filtrar por cliente válido).
- Filas válidas tras limpieza: 805.620 (antes de excluir TEST001).
- Umbral p99: Quantity=17.506, Spend=29.704,60. **70 clientes (1,19%) marcados como mayoristas** — **DECISIÓN AÚN PENDIENTE**: excluir del modelo, modelar aparte, o incluir `is_wholesale` como feature. No abordado todavía en ningún módulo de labeling/features.
- 13 clientes con más de 1 país registrado (posible ruido de dirección de envío).

---

## 3. Etiquetado de churn (`src/labeling/churn.py`)

**`build_churn_labels(df_sales, cutoff_date, horizon_days)`:**
- `churn=1` si NO hay compra válida en `(cutoff_date, cutoff_date + horizon_days]`; `churn=0` si sí la hay.

**Decisión de horizonte — H=90 días (CERRADA):**
Comparación con fecha de corte común `2011-08-11` (= fecha máxima del dataset `2011-12-09` − 120 días, para que ningún horizonte sufra censura por la derecha), sobre 5.179 clientes:

| Horizonte | % Churn | % Retención |
|---|---|---|
| 60 días | 70,73% | 29,27% |
| 90 días | 60,78% | 39,22% |
| 120 días | 53,21% | 46,79% |

**H=90 elegido** como principal (equilibrio estadístico + ventana de negocio accionable), 60 y 120 documentados como análisis de sensibilidad.

---

## 4. Features RFM (`src/features/rfm.py`)

**`build_rfm_features(df_sales, cutoff_date)`** — usa solo `InvoiceDate <= cutoff_date`. Genera:
- `recency` (días desde última compra), `frequency` (nº facturas únicas), `monetary` (`Quantity * Price`), `n_products`, `n_countries`, `avg_days_between_purchases` (NaN si solo 1 compra).

**Dos bugs reales detectados y corregidos:**
1. Columna `UnitPrice` no existe → es `Price`. Corregido en `rfm.py` (no se tocó `cleaning.py`, que ya es un contrato en uso).
2. `avg_days_between_purchases` contaba líneas de una misma factura con timestamps distintos (segundos de diferencia, por escaneo de artículos) como compras separadas. Corregido: se agrupa por `["Customer ID", "Invoice"]` (fecha mínima por factura) antes de calcular intervalos. Test de regresión añadido.

**Validación con datos reales (corte 2011-09-10):** 5.281 clientes, `monetary` sin negativos (min 1,55 / max 484.615,10), nulos en `avg_days_between_purchases` = clientes con `frequency==1` exactamente (1.577 = 1.577 tras el fix).

---

## 5. Unión features + etiqueta (`src/features/pipeline.py`)

**`build_dataset(df_sales, cutoff_date, horizon_days)`** — merge de `build_rfm_features` + `build_churn_labels`, con validación de coherencia (misma cantidad de filas tras el merge). Valida OK: 5.281 filas, distribución 56,64%/43,36% coincide con labels solas.

---

## 6. Snapshots mensuales (`src/features/snapshots.py`)

- **`generate_snapshot_dates(min_date, max_date, horizon_days, freq="MS")`** — cortes mensuales sin censura.
- **`filter_active_customers(df_sales, cutoff_date, lookback_days=365)`** — en cada snapshot, solo clientes con compra en los últimos 365 días (decisión tomada explícitamente, para no arrastrar clientes inactivos).
- **`build_multi_snapshot_dataset(...)`** — concatena todos los snapshots.

**Resultado real:** 22 snapshots (dic-2009 a sep-2011, H=90), **71.281 filas, 5.249 clientes únicos**. Primer snapshot (2009-12-01) tiene solo 1 fila (poco histórico, no es anomalía).

---

## 7. Split temporal (`src/splitting/temporal_split.py`)

**`temporal_split(dataset, date_col, train_end, valid_end)`** + **`summarize_split(...)`**. Nunca aleatorio. **Limitación documentada:** un mismo `Customer ID` puede aparecer en train, valid Y test (varios snapshots) — no es leakage (cada fila respeta su propio corte) pero sí dependencia entre observaciones.

**Split elegido — Opción A (CERRADO):**

| Split | Filas | Clientes únicos | Snapshots | Fechas | % Churn |
|---|---|---|---|---|---|
| train | 32.537 | 4.266 | 13 | 2009-12 a 2010-12 | 49,34% |
| valid | 17.076 | 4.614 | 4 | 2011-01 a 2011-04 | 64,51% |
| test | 21.668 | 4.796 | 5 | 2011-05 a 2011-09 | 58,70% |

En código: `train_end = snapshot_dates[12]`, `valid_end = snapshot_dates[16]`.

---

## 8. Hallazgo clave: deriva temporal en la tasa de churn

Al desagregar `%churn` por snapshot individual (no por split agregado):

| Periodo | % Churn |
|---|---|
| ene-2010 | 36,86% |
| jul-2010 | 51,33% |
| **dic-2010 / ene-2011 (pico)** | **66,92% / 67,08%** |
| sep-2011 (final) | 50,50% |

**Interpretación de trabajo (no confirmada estadísticamente):**
- Subida 2010: maduración de cohortes (crece la base de clientes de 955 a 4.266 filas/mes, se incorporan más clientes nuevos/ocasionales).
- Bajada 2011: efecto estacional navideño — snapshots nov-dic-2010 concentran compradores puntuales que no repiten en 90 días; esa cohorte se diluye en 2011.

**Consecuencia:** el problema NO es estacionario. Esto llevó a añadir features estacionales (sección 9).

---

## 9. Features estacionales (`src/features/temporal.py`)

**`add_seasonal_features(dataset, date_col="cutoff_date")`** — añade `month`, `month_sin`, `month_cos` (codificación cíclica), `is_post_holiday` (1 si dic-mar, 0 resto). Integrado al final de `build_dataset` en `pipeline.py`.

**Validación:** `%churn` con `is_post_holiday=0` → 53,92%; con `is_post_holiday=1` → 60,36%. Diferencia de ~6,5 puntos confirma señal real.

---

## 10. Preprocesamiento y features finales del modelo (`src/models/baseline.py`)

**`NUMERIC_FEATURES`** (usadas por todos los modelos):
```
recency, frequency, monetary, n_countries, n_products,
avg_days_between_purchases, month_sin, month_cos, is_post_holiday
```
Excluidas explícitamente: `Customer ID`, `cutoff_date`, `month` cruda (redundante con `month_sin`/`month_cos`; como entero induce escala falsa).

**`build_preprocessing_pipeline`** — `SimpleImputer(median)` + `StandardScaler` (solo para modelos lineales).

---

## 11. Métricas (`src/evaluation/metrics.py`)

**`evaluate_binary(y_true, proba, threshold=0.5)`** → PR-AUC, ROC-AUC, precision, recall, F1.

---

## 12. Resultados de los 4 modelos (orden: Dummy → Logistic → RF → XGBoost)

**Script:** `scripts/train_all_models.py` (con parche `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` al inicio, necesario porque `pythonpath` de `pyproject.toml` solo cubre pytest, no scripts sueltos).

| Modelo | Split | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| **Dummy** (stratified) | train | 0,4965 | 0,5062 | 0,4997 | 0,4991 |
| | valid | 0,6448 | 0,4995 | 0,6446 | 0,4881 |
| | test | 0,5871 | 0,5002 | 0,5871 | 0,4926 |
| **Logistic** (`class_weight="balanced"`) | train | 0,6909 | 0,7226 | 0,6351 | 0,7344 |
| | valid | 0,8271 | 0,7735 | 0,8114 | 0,7238 |
| | test | 0,7852 | 0,7702 | 0,7755 | 0,6898 |
| **Random Forest** (regularizado: `max_depth=6`, `min_samples_leaf=50`, `class_weight="balanced"`) | train | 0,7499 | 0,7720 | 0,6804 | 0,7322 |
| | valid | 0,8357 | 0,7735 | 0,8470 | 0,4735 |
| | test | 0,8104 | 0,7789 | 0,8005 | 0,5246 |
| **XGBoost** (`max_depth=4`, `learning_rate=0.05`, `n_estimators=300`, `scale_pos_weight=1,0267`) | train | 0,7797 | 0,7948 | 0,6943 | 0,7526 |
| | valid | 0,8311 | 0,7697 | 0,8451 | 0,3779 |
| | test | 0,8018 | 0,7679 | 0,7994 | 0,4860 |

**Random Forest — incidencia importante:** primer intento SIN regularizar (`n_estimators=300`, sin límite de profundidad) dio `train: PR-AUC=1.0000, ROC-AUC=1.0000` — sobreajuste severo evidente. Se regularizó con `max_depth=6, min_samples_leaf=50` y los resultados de la tabla ya son los corregidos.

**Conclusión provisional (a fecha de este documento, AÚN NO CERRADA):**
- Los 3 modelos reales están prácticamente empatados en ROC-AUC de valid/test (0,768-0,779) — diferencias de milésimas, no significativas dado que hay clientes repetidos entre splits.
- **XGBoost es el que MENOS aporta**: peor ROC-AUC en valid/test que los otros dos, y la mayor brecha train/valid (posible señal de sobreajuste leve).
- **RF y XGBoost tienen recall muy bajo a threshold=0,5** (0,38-0,52) frente a Logística (0,69-0,72). Hipótesis de trabajo: sus probabilidades predichas están calibradas para la prevalencia de train (~49%), pero valid/test tienen prevalencia real más alta (64,5% y 58,7%) por la deriva temporal — no se ha confirmado todavía.
- **Conclusión de trabajo:** la Logística parece preferible por simplicidad + interpretabilidad, dado que el RF/XGBoost no muestran mejora clara en discriminación. PERO esto no está cerrado — falta el paso de calibración/threshold para comparar de forma justa (ver sección 13).

---

## 13. Calibración, umbral y modelo final (CERRADO)

**Modelo final:** Random Forest (`max_depth=6`, `min_samples_leaf=50`, 300 árboles, `class_weight="balanced"`, `random_state=42`) ajustado solo en train + calibración Platt ajustada solo en valid (modelo base congelado con `FrozenEstimator`) + umbral operativo con recall >= 0,80 elegido solo en valid. Test solo se evalúa.

**Prevalencia por split:** train 0,4934 (n=32.537) · valid 0,6451 (n=17.076) · test 0,5870 (n=21.668) — deriva temporal que explica la mala calibración de los modelos sin recalibrar.

**Calibración en test (Brier / ECE):**

| Modelo | Sin calibrar | Platt (valid) | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Logística | 0,2034 / 0,1201 | 0,1886 / 0,0385 | 0,7852 | 0,7702 |
| Random Forest | 0,2094 / 0,1504 | **0,1863 / 0,0563** | 0,8104 | 0,7789 |
| XGBoost | 0,2275 / 0,1866 | 0,2000 / 0,1025 | 0,8008 | 0,7668 |

Platt por CV sobre train no transfiere a test (XGBoost llegó a ROC-AUC < 0,5); isotonic se probó solo como sensibilidad (RF test: Brier 0,1856, ECE 0,0566, sin mejora práctica sobre Platt).

**Umbral del RF + Platt (recall >= 0,80 en valid) = 0,599:**

| Split | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| valid | 0,7915 | 0,8004 | 0,7959 | 3738 | 2323 | 2199 | 8816 |
| test | 0,7495 | 0,8146 | 0,7807 | 5486 | 3463 | 2358 | 10361 |

**Artefacto (`scripts/train_final.py`):** guarda `models/churn_model_1.0.0.joblib` y `models/churn_model_1.0.0_metadata.json` (features en orden, umbral 0,598547, métricas, SHA-256 del Excel, versiones, commit). Aborta si el umbral no redondea a 0,599. **Importante:** con scikit-learn 1.8.0 el umbral sale 0,5954; con la versión de `uv.lock` (1.9.x) sale 0,5985. Instalar siempre desde `uv.lock`.

### SHAP (`scripts/analyze_shap.py`)
Se explica el modelo base (imputación → RF) con `TreeExplainer`; Platt solo remapea la probabilidad. Importancia global (media |SHAP|, muestra de 2.000 filas de test): `monetary` 0,0751 · `frequency` 0,0749 · `recency` 0,0602 · `n_products` 0,0309 · `month_cos` 0,0250 · `avg_days_between_purchases` 0,0197 · `month_sin` 0,0149 · `is_post_holiday` 0,0038 · `n_countries` 0,0000. Recency empuja hacia churn; frequency, monetary y n_products hacia no churn. `n_countries` tiene contribución nula (99,85 % de filas con un solo país). Los casos locales se eligen por probabilidad, no por etiqueta.

### Análisis por segmento (`scripts/analisis_wholesale.py`, decisión D6)
Wholesale = `monetary` >= p90 de train (2.736,07). Con el modelo congelado y el umbral global: en **test**, wholesale (n=4.628, prevalencia 0,248) recall 0,333 / precision 0,667 / ROC-AUC 0,7945; resto (n=17.040, prevalencia 0,679) recall 0,862 / precision 0,753 / ROC-AUC 0,7035. No se reentrena: se documenta como limitación y trabajo futuro (el tratamiento de mayoristas pendiente desde la auditoría queda resuelto así).

### Monetary bruto vs neto (`scripts/experimento_monetary_neto.py`, `src/features/returns.py`)
El modelo usa Monetary **bruto** (las cancelaciones se eliminan en `get_clean_sales`). Se probó `monetary_net` (ventas − devoluciones acumuladas hasta T, sin leakage y con test dedicado) y `return_ratio`, con el mismo protocolo y 5 semillas (42, 1, 2, 3, 4):

| Variante | Test PR-AUC | Test ROC-AUC | Brier | ECE | Umbral medio |
|---|---|---|---|---|---|
| A bruto | 0,8104 ± 0,0006 | 0,7789 ± 0,0005 | 0,1866 | 0,0584 | 0,5911 |
| B neto | 0,8114 ± 0,0007 | 0,7794 ± 0,0006 | 0,1861 | 0,0554 | 0,5932 |
| C bruto + return_ratio | 0,8099 ± 0,0003 | 0,7790 ± 0,0005 | 0,1862 | 0,0553 | 0,5947 |

Diagnóstico: 40,3 % de las filas cliente+snapshot tienen alguna devolución acumulada; `monetary_net` <= 0 solo en el 0,71 %; correlación monetary vs monetary_net 0,9928; el churn medio es menor en clientes con devoluciones (0,429 vs 0,645), probablemente porque devuelven más los clientes más activos. El neto mejora ~0,001 de PR-AUC (en las 5 semillas, pero dentro de un margen práctico irrelevante). **Estado:** se mantiene el bruto como regla final y se reporta el neto como análisis de sensibilidad; adoptar el neto obligaría a reentrenar, recalibrar, fijar un nuevo umbral y rehacer SHAP, análisis por segmento y presentación.

### API (`src/api/`)
FastAPI + Pydantic. Entrada: `recency`, `frequency`, `monetary`, `n_countries`, `n_products`, `avg_days_between_purchases` (opcional) y `cutoff_date` (opcional; de ella se derivan `month_sin`, `month_cos`, `is_post_holiday`). Salida: `churn_probability`, `risk_level` (`HIGH` si >= umbral, `LOW` si no), `threshold`, `model_version` y, con `?explain=true`, `top_factors` (3 variables con mayor |SHAP|). Errores: 422 por validación, 503 sin modelo, 400/500 sin trazas; no se registran payloads. Se leen features, umbral y versión del JSON de metadatos; ruta con `CHURN_MODEL_PATH`. Límite de alcance: la API recibe features ya calculadas por cliente.

### Docker
`Dockerfile` (python:3.12-slim, `uv sync --frozen --no-install-project`, usuario sin privilegios, `HEALTHCHECK` que exige `model_loaded: true`) y `.dockerignore`. La instalación desde `uv.lock` y la API sobre ese entorno se validaron por HTTP; el `docker build` en sí debe probarse en el equipo del usuario.

**Tests:** 85 pasan (59 originales + 16 de API + 5 de SHAP en API + 5 de devoluciones).

---

## 14. Pendientes generales

- [ ] Probar `docker build` / `docker run` en local.
- [ ] Decidir mecanismo de trazabilidad: MLflow local o mantener `metadata.json` como "mecanismo equivalente" (DoD del documento técnico).
- [ ] Confirmar la versión real de Python del entorno (3.12 según `.python-version`; una bitácora mencionó 3.14.3).
- [ ] Añadir `httpx` a las dependencias (`uv add httpx`) para `TestClient`; sincronizar `uv.lock`.
- [ ] Completar `.gitignore` (data/, .venv, __pycache__, *.bak) y decidir si se versiona el `.joblib`.
- [ ] Borrar `*.bak` y `src/tfm_churn_ecommerce/`.
- [ ] Opcional: adelgazar la imagen Docker moviendo jupyter/shap/xgboost a un extra `research`.
- [x] ~~Corregir la bitácora Parte 3~~ — XGBoost sí se implementó y ejecutó; la consolidación está en `TFM_Churn_Ecommerce_Bitacora_06_Cierre_Tecnico.md`.

---

## 15. Notas de entorno / incidencias técnicas recurrentes

- **`ModuleNotFoundError: No module named 'src'`** al ejecutar scripts sueltos con `uv run python scripts/archivo.py`: `pythonpath = ["."]` en `pyproject.toml` solo cubre pytest, no scripts. Solución adoptada en TODOS los scripts nuevos:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  ```
- Entorno Windows + PowerShell — comandos `-c "..."` con comillas anidadas complejas dan `SyntaxError` por reinterpretación de PowerShell; preferir siempre scripts en archivo `.py` sobre comandos de una línea.
- `uv` como gestor de dependencias (no pip/conda directo).

---

*Documento de traspaso actualizado tras cerrar el bloque técnico (modelo final, SHAP, API y Docker). Punto de continuación: pendientes de la sección 14.*
