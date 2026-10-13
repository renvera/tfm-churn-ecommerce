# tfm-churn-ecommerce

Predicción de churn (abandono de clientes) en e-commerce con el dataset
[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI, ID 502).
Proyecto asociado al TFM *Retenzia Analytics* (Máster en Big Data & Business Analytics, EUDE).

El repositorio cubre el ciclo completo: limpieza de datos, construcción de snapshots
mensuales con etiqueta de churn a 90 días, modelos (logística, Random Forest, XGBoost),
calibración, umbral operativo, explicabilidad (SHAP) y una API FastAPI empaquetada con Docker.

> El historial de decisiones y avances del proyecto está en [`reports/bitacoras/`](reports/bitacoras/).
> El README antiguo (bitácora de los primeros pasos) se conserva allí como
> `TFM_Churn_Ecommerce_Bitacora_Avance_00_README_original.md`.

## Definición del problema

- **Unidad de análisis:** cliente en una fecha de corte (`cutoff_date`). Hay un snapshot por mes.
- **Features** (solo con información hasta el corte): recency, frequency, monetary, `n_countries`,
  `n_products`, `avg_days_between_purchases` y estacionalidad (mes en seno/coseno, post-Navidad).
- **Etiqueta:** `churn = 1` si el cliente no compra en los 90 días posteriores al corte.
  Se consideran solo clientes activos en los 365 días previos.
- **Nota:** el churn es *no contractual* (nadie cancela una suscripción); es una definición operativa
  basada en inactividad.

## Cómo ejecutarlo desde cero

Requisitos: Python 3.12 y [uv](https://docs.astral.sh/uv/). En Windows (PowerShell):
`irm https://astral.sh/uv/install.ps1 | iex`.

```bash
# 1. Dependencias (respeta uv.lock; el umbral final depende de la versión de scikit-learn)
uv sync

# 2. Datos: descargar Online Retail II (UCI, ID 502) y guardarlo como
#    data/raw/online_retail_II.xlsx  (la primera lectura tarda 2-3 min; luego usa caché parquet)

# 3. Tests
uv run pytest

# 4. Entrenar el modelo final (genera models/churn_model_1.0.0.joblib y su metadata.json)
uv run python scripts/train_final.py

# 5. Servir la API
uv run uvicorn src.api.main:app --reload
#    GET  /health
#    POST /predict            (con ?explain=true devuelve top_factors SHAP)
```

`data/raw|interim|processed` y `models/*` no se versionan (se regeneran con los pasos 2 y 4).

**Docker** (requiere haber ejecutado el paso 4, porque la imagen copia `models/`):

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Estructura

```
src/
  data/           carga y limpieza
  features/       RFM, snapshots, devoluciones (opcional), pipeline
  splitting/      split temporal (con embargo opcional) y folds walk-forward
  models/         baseline, logística, Random Forest, XGBoost
  evaluation/     métricas, calibración, umbral, bootstrap por cliente
  api/            FastAPI (schemas, predictor, main)
scripts/          entrenamiento, análisis y experimentos (ver abajo)
tests/            suite pytest
reports/          gráficos, CSV de resultados y bitácoras
```

## Protocolo de evaluación

**Protocolo principal (documentado en el TFM).** Split temporal por `cutoff_date`:
train hasta `snapshot_dates[12]`, validación hasta `snapshot_dates[16]`, test el resto.
Modelo final: Random Forest (train) → calibración Platt (validación, modelo base congelado) →
umbral con recall ≥ 0,80 elegido **solo en validación**; test solo evalúa.
`train_final.py` aborta si el umbral no redondea a 0,599.

**Análisis de robustez con embargo.** Como la etiqueta de un corte mira 90 días hacia adelante,
las ventanas de etiqueta de train y de validación se solapan si no se deja un hueco.
Un *embargo* de 90 días (`gap_days` en `temporal_split`, `SPLIT_EMBARGO_DAYS` en `settings.py`)
hace las ventanas disjuntas. El protocolo principal **no** lo aplica; los análisis de robustez sí
(horizontes, backtest rodante, solapamiento de clientes, devoluciones).

## Resultados

Modelo final (Random Forest + Platt, umbral 0,599, protocolo principal):

| | Validación | Test |
|---|---|---|
| ROC-AUC | 0,7735 | 0,7789 |
| PR-AUC | 0,8357 | 0,8104 |
| Recall | 0,8004 | 0,8146 |
| Precisión | | 0,7495 |
| F1 | | 0,7807 |
| Brier / ECE | | 0,1863 / 0,0563 |

### Efecto del embargo

Con embargo de 90 días el umbral (recall ≥ 0,80 en validación) sería 0,579 en lugar de 0,599.
La validación pasa de 17.076 a 8.608 filas, y en test el ROC-AUC baja de 0,779 a 0,773 y el
recall a 0,795. El test es otro periodo (jul-sep frente a may-sep), así que precisión y PR-AUC no
son comparables uno a uno; el ROC-AUC sí. Las conclusiones no cambian.

### Sensibilidad al horizonte (`scripts/horizon_sensitivity.py`)

ROC-AUC en test con embargo igual al horizonte (Random Forest; IC95 % bootstrap por cliente en el CSV):

| Horizonte | ROC-AUC test | Prevalencia de churn | Snapshots de test |
|---|---|---|---|
| 60 d | 0,770 | 65,4 % | 5 |
| 90 d | 0,773 | 56,2 % | 3 |
| 120 d | 0,772 | 47,7 % | 1 |

El ROC-AUC es estable; la PR-AUC cambia con la prevalencia. Con 120 d el test es un único snapshot.

### Backtest rodante (`scripts/walk_forward_backtest.py`)

11 folds con ventana expansiva y embargo de 90 d. ROC-AUC medio entre folds:
logística 0,776, Random Forest 0,778, XGBoost 0,769 (desviación 0,013-0,020). El rendimiento es
estable en el tiempo; el primer fold (solo un año de historia) es el más bajo (0,748 en RF).

### Clientes vistos y no vistos (`scripts/customer_overlap.py`)

El 80 % de las filas de test son de clientes que ya aparecían en train. En una prueba controlada con
asignación aleatoria de clientes, el ROC-AUC es 0,02-0,03 mayor para clientes vistos que para no vistos
(Random Forest: +0,025). Es una estimación del optimismo por repetir clientes entre train y test.
La comparación directa vistos/nuevos (0,794 frente a 0,724) mezcla ese efecto con que los nuevos
tienen menos historial.

### Features de devoluciones (`scripts/compare_return_features.py`)

Añadir `return_amount` y `return_ratio` (cancelaciones acumuladas hasta el corte, sin leakage) no
mejora de forma relevante. Es coherente con el experimento de `monetary_net` de la bitácora 06
(PR-AUC de test 0,8104 → 0,8114). Se mantiene el conjunto de features original; las de devoluciones
quedan como opción en el pipeline.

## Scripts

| Script | Para qué |
|---|---|
| `train_final.py` | Modelo final serializado + metadata |
| `train_baseline_logistic.py`, `train_all_models.py` | Comparación de modelos |
| `evaluate_thresholds.py`, `evaluate_top_decile.py` | Umbral y decil superior |
| `analyze_shap.py`, `analisis_wholesale.py` | Explicabilidad y segmento mayorista |
| `control_cv_shuffle.py`, `diagnostico_xgboost.py` | Controles y diagnóstico |
| `experimento_monetary_neto.py`, `compare_return_features.py` | Devoluciones / monetary neto |
| `horizon_sensitivity.py`, `walk_forward_backtest.py`, `customer_overlap.py` | Robustez (con embargo) |

Ejecutar desde la raíz: `uv run python scripts/<nombre>.py`.

## Limitaciones

- **Un solo comercio y dos años de datos.** Mayorista británico de regalos; la estacionalidad (Navidad)
  solo se observa dos veces. No hay garantía de que generalice a otros e-commerce.
- **Churn no contractual.** La etiqueta es inactividad a 90 días, no una baja real; algunos "churn"
  volverían a comprar más tarde.
- **Solape de etiquetas y de clientes.** Sin embargo, train y validación comparten ventana de
  etiqueta (efecto pequeño, ver arriba). El mismo cliente aparece en varios snapshots: el ROC-AUC del
  protocolo principal está inflado en torno a 0,02 respecto a clientes nuevos. Los IC deben calcularse
  por cliente (`cluster_bootstrap_ci`), no por fila.
- **Pocos snapshots de validación/test con embargo** (2 y 3 con 90 d): el umbral es menos estable.
- **Segmento mayorista.** Con el umbral global el recall en clientes con monetary ≥ p90 es 0,333
  frente a 0,862 en el resto.
- **Dependencia de versión.** El umbral cambia con la versión de scikit-learn (1,8,0 → 0,5954;
  1,9.x → 0,5985); instalar siempre desde `uv.lock`.
- **Pendiente:** probar `docker build` en local y decidir MLflow frente a `metadata.json`.

## Datos y licencia

Dataset: *Online Retail II*, UCI Machine Learning Repository (ID 502). Consultar su licencia y
condiciones de uso en la página del dataset; aquí no se versiona.
