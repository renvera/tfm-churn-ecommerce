# TFM Churn E-commerce — Documento de traspaso completo

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
│   │   └── temporal.py
│   ├── splitting/              # (NOTA: no "split/" como en el plan original)
│   │   └── temporal_split.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── logistic.py
│   │   ├── random_forest.py
│   │   └── xgboost_model.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration.py     # recién creado, sin ejecutar todavía
│   │   └── threshold.py       # recién creado, sin ejecutar todavía
│   ├── explainability/         # pendiente (SHAP)
│   └── api/                    # pendiente (FastAPI)
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
│   ├── test_calibration.py    # recién creado, sin ejecutar todavía
│   └── test_threshold.py      # recién creado, sin ejecutar todavía
├── scripts/
│   ├── check_structure.py
│   ├── verify_setup.py
│   ├── train_baseline_logistic.py
│   ├── train_all_models.py
│   └── evaluate_thresholds.py  # recién creado, sin ejecutar todavía
├── models/
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

## 13. EN CURSO — Calibración y selección de threshold (sección 12 del doc técnico)

**Este es el punto exacto donde se quedó el trabajo.** Se ha escrito el código pero **NO se ha ejecutado ni se tienen resultados todavía**:

**`src/evaluation/calibration.py`** (creado, sin ejecutar):
- `calibration_table(y_true, proba, n_bins=10)` — tabla de probabilidad media predicha vs tasa observada por bins.
- `expected_calibration_error(table)` — ECE.
- `brier(y_true, proba)` — Brier score.
- `plot_calibration_curves(tables, title, output_path)` — guarda PNG en `reports/`.

**`src/evaluation/threshold.py`** (creado, sin ejecutar):
- `best_threshold_f1(y_true, proba)` — threshold que maximiza F1.
- `threshold_for_min_recall(y_true, proba, min_recall)` — threshold que maximiza precision dado un recall mínimo (objetivo usado: `TARGET_RECALL = 0.80`).

**`scripts/evaluate_thresholds.py`** (creado, sin ejecutar) — hace todo el flujo:
1. Reconstruye dataset + split (idéntico a `train_all_models.py`).
2. Entrena Logistic, RF, XGBoost sobre train.
3. Calcula tabla de calibración + Brier + ECE por modelo, en valid y test. Guarda gráficos en `reports/calibration_valid.png` y `reports/calibration_test.png`.
4. Para cada modelo, calcula 3 thresholds (F1 máximo en valid, recall≥0,80 en valid, 0,5 de referencia) y evalúa cada uno en valid Y test (test solo se mira, no se usa para elegir threshold — regla de oro del split temporal).
5. Imprime matriz de confusión (TN/FP/FN/TP) para cada combinación modelo×threshold×split.

**Tests correspondientes ya creados:** `tests/test_calibration.py`, `tests/test_threshold.py` (pasan con datos sintéticos, no ejecutados sobre datos reales todavía).

### Próxima acción inmediata al retomar

```bash
uv run pytest tests/test_calibration.py tests/test_threshold.py -v
uv run python scripts/evaluate_thresholds.py
```

**Qué interpretar en la salida** (instrucciones ya dadas al usuario, pendientes de aplicar):
1. Probabilidad media predicha vs prevalencia observada en valid/test, por modelo — para confirmar o descartar la hipótesis de la sección 12.
2. Brier/ECE por modelo — menor es mejor calibrado.
3. Si el threshold elegido en valid transfiere bien a test (dado que la prevalencia cambia entre ambos por la deriva temporal).
4. Si con threshold ajustado (no 0,5) el RF/XGBoost igualan el recall de la Logística — si es así, refuerza la conclusión de preferir Logística por simplicidad.

---

## 14. Pendientes generales (no bloqueantes, pero abiertos)

- [ ] **Ejecutar `evaluate_thresholds.py` y decidir modelo final** con base en calibración/threshold, no solo PR-AUC/ROC-AUC.
- [ ] **Decidir tratamiento de clientes mayoristas** (`is_wholesale`, 70 clientes, 1,19%) — abierto desde la auditoría inicial, nunca resuelto.
- [ ] `src/explainability/` — SHAP sobre el modelo candidato final.
- [ ] Incorporar MLflow para trazabilidad de experimentos (diferido desde el inicio).
- [ ] `src/api/` — servicio FastAPI de inferencia.
- [ ] Corregir la bitácora Parte 3 — dice que XGBoost está "implementado, ejecución pendiente" cuando en realidad se creó y ejecutó en un turno posterior de la conversación original. Regenerar bitácora consolidada al cerrar este bloque.

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

*Documento generado para traspaso de contexto completo a otro asistente. Contiene todas las decisiones, código y resultados necesarios para continuar exactamente desde el punto 13 (calibración y threshold, código escrito pero sin ejecutar).*
