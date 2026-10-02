"""
scripts/analisis_wholesale.py

Análisis de robustez por segmento (Decisión D6) con el modelo FINAL
congelado: RF ajustado en train, Platt ajustado en valid y umbral
operativo reobtenido en valid.

- Define "wholesale" como monetary >= percentil 90 de TRAIN (umbral de
  segmento aprendido sin fuga: nunca con valid/test).
- Evalúa el modelo final por segmento (wholesale vs resto) en valid y
  test: prevalencia, Brier, ROC-AUC, PR-AUC, precision/recall/F1 con el
  umbral operativo y matriz de confusión.
- Incluye el control documental de n_countries (value_counts), que
  motiva su SHAP nulo.

No reentrena el protocolo ni modifica archivos.

Uso:
    uv run python scripts/analisis_wholesale.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from src.data.loader import load_online_retail
from src.data.cleaning import get_clean_sales
from src.features.snapshots import build_multi_snapshot_dataset, generate_snapshot_dates
from src.splitting.temporal_split import temporal_split
from src.models.baseline import NUMERIC_FEATURES
from src.models.random_forest import build_random_forest_pipeline
from src.evaluation.calibration import calibrate_prefit
from src.evaluation.threshold import threshold_for_min_recall

TARGET_RECALL = 0.80
PCTL = 90

# --- Datos y split temporal (idéntico a evaluate_thresholds.py) ---
df = get_clean_sales(load_online_retail())
snapshot_dates = generate_snapshot_dates(
    df["InvoiceDate"].min(), df["InvoiceDate"].max(), horizon_days=90
)
dataset = build_multi_snapshot_dataset(
    df, snapshot_dates, horizon_days=90, lookback_days=365
)
train, valid, test = temporal_split(
    dataset,
    date_col="cutoff_date",
    train_end=snapshot_dates[12],
    valid_end=snapshot_dates[16],
)

print("=" * 70)
print("ANÁLISIS POR SEGMENTO (D6) - modelo final congelado, sin reentrenar")
print("=" * 70)

# --- Control documental: n_countries (cierra el hallazgo SHAP) ---
print("\nCONTROL n_countries (dataset completo, proporciones):")
print(dataset["n_countries"].value_counts(normalize=True).head(5).to_string())

# --- Umbral de segmento aprendido SOLO con train ---
monetary_thr = float(np.percentile(train["monetary"], PCTL))
print(f"\nDefinición de wholesale: monetary >= p{PCTL} de train = {monetary_thr:.2f}")
seg_valid = (valid["monetary"] >= monetary_thr).values
seg_test = (test["monetary"] >= monetary_thr).values

# --- Modelo final congelado: RF (train) + Platt (valid) + umbral operativo ---
pipe = build_random_forest_pipeline()
pipe.fit(train[NUMERIC_FEATURES], train["churn"])
cal = calibrate_prefit(pipe, method="sigmoid")
cal.fit(valid[NUMERIC_FEATURES], valid["churn"])

proba_valid = cal.predict_proba(valid[NUMERIC_FEATURES])[:, 1]
thr = threshold_for_min_recall(valid["churn"].values, proba_valid, TARGET_RECALL)
print(f"Umbral operativo reobtenido en valid: {thr:.3f} (esperado 0.599)")

proba_test = cal.predict_proba(test[NUMERIC_FEATURES])[:, 1]
y_valid = valid["churn"].values
y_test = test["churn"].values


def evaluar(nombre: str, y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> None:
    yy, pp = y[mask], p[mask]
    pred = (pp >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    dos_clases = len(np.unique(yy)) == 2
    auc_roc = roc_auc_score(yy, pp) if dos_clases else float("nan")
    auc_pr = average_precision_score(yy, pp) if dos_clases else float("nan")
    pct = len(yy) / len(y) * 100
    print(
        f"  {nombre:<10} n={len(yy):>6} ({pct:4.1f}%)  prevalencia={yy.mean():.4f}  "
        f"Brier={brier_score_loss(yy, pp):.4f}  ROC-AUC={auc_roc:.4f}  PR-AUC={auc_pr:.4f}"
    )
    print(
        f"  {'':<10} precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}  "
        f"|  TP={tp} FP={fp} FN={fn} TN={tn}"
    )


print("\nVALID (bloque donde se eligió el umbral):")
evaluar("resto", y_valid, proba_valid, ~seg_valid)
evaluar("wholesale", y_valid, proba_valid, seg_valid)

print("\nTEST (evaluación final):")
evaluar("resto", y_test, proba_test, ~seg_test)
evaluar("wholesale", y_test, proba_test, seg_test)

print("\nPerfil descriptivo del segmento (medianas):")
for nombre, split, mask in (("valid", valid, seg_valid), ("test", test, seg_test)):
    ws, resto = split[mask], split[~mask]
    print(
        f"  [{nombre}] wholesale: monetary={ws['monetary'].median():.0f}  "
        f"frequency={ws['frequency'].median():.0f}  recency={ws['recency'].median():.0f}  "
        f"| resto: monetary={resto['monetary'].median():.0f}  "
        f"frequency={resto['frequency'].median():.0f}  recency={resto['recency'].median():.0f}"
    )

print("\nNOTA (D6): este análisis NO reentrena ni modifica el modelo final;")
print("documenta el comportamiento por segmento como limitación/trabajo futuro.")