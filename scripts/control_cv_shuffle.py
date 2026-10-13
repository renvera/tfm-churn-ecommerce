"""
scripts/control_cv_shuffle.py

Control del diagnóstico de XGBoost (sesión 3): demostrar que la anomalía
de la CV sin barajar es estructural (bloques temporales), no un defecto
del modelo. Cuatro controles sobre los mismos datos:

  1) In-sample: XGBoost ajustado con todo train y evaluado sobre train
     (debe dar AUC alto: el modelo SÍ aprende).
  2) CV sin barajar: StratifiedKFold(shuffle=False) sobre train ordenado
     temporalmente (reproduce la anomalía; OOF esperado ~0.4385).
  3) CV barajada (control): StratifiedKFold(shuffle=True); los folds
     mezclan meses y la anomalía debe desaparecer.
  4) Referencia out-of-time: AUC del modelo full-train en valid.

Uso:
    uv run python scripts/control_cv_shuffle.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_online_retail
from src.data.cleaning import get_clean_sales
from src.features.snapshots import build_multi_snapshot_dataset, generate_snapshot_dates
from src.splitting.temporal_split import temporal_split
from src.models.baseline import NUMERIC_FEATURES
from src.models.xgboost_model import build_xgboost_pipeline, compute_scale_pos_weight

RANDOM_STATE = 42
N_SPLITS = 5

# --- Datos y split temporal (idéntico a evaluate_thresholds.py) ---
df = get_clean_sales(load_online_retail())
snapshot_dates = generate_snapshot_dates(
    df["InvoiceDate"].min(), df["InvoiceDate"].max(), horizon_days=90
)
dataset = build_multi_snapshot_dataset(
    df, snapshot_dates, horizon_days=90, lookback_days=365
)
train, valid, _ = temporal_split(
    dataset,
    date_col="cutoff_date",
    train_end=snapshot_dates[12],
    valid_end=snapshot_dates[16],
)
X_train, y_train = train[NUMERIC_FEATURES], train["churn"].values
X_valid, y_valid = valid[NUMERIC_FEATURES], valid["churn"].values

spw = compute_scale_pos_weight(train["churn"])


def cv_auc(cv: StratifiedKFold) -> tuple[np.ndarray, float]:
    """Ajusta un XGBoost fresco por fold; devuelve AUC por fold y OOF."""
    oof = np.zeros(len(y_train))
    fold_aucs = []
    for tr_idx, te_idx in cv.split(X_train, y_train):
        pipe = build_xgboost_pipeline(scale_pos_weight=spw)
        pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        p = pipe.predict_proba(X_train.iloc[te_idx])[:, 1]
        oof[te_idx] = p
        fold_aucs.append(roc_auc_score(y_train[te_idx], p))
    return np.array(fold_aucs), roc_auc_score(y_train, oof)


print("=" * 70)
print("CONTROL CV - XGBOOST: ¿la anomalía era temporal (bloques), no del modelo?")
print("=" * 70)
print(f"  train n={len(train)}  valid n={len(valid)}  scale_pos_weight={spw:.4f}")
print(f"  train ordenado por cutoff_date: {train['cutoff_date'].is_monotonic_increasing}")

# 1) In-sample: el modelo aprende con todo train
print("\nAjustando modelo full-train (in-sample)...")
pipe_full = build_xgboost_pipeline(scale_pos_weight=spw)
pipe_full.fit(X_train, y_train)
auc_in = roc_auc_score(y_train, pipe_full.predict_proba(X_train)[:, 1])

# 2) CV sin barajar (reproduce la anomalía de la sesión 3)
print("CV sin barajar (5 folds en bloque temporal, tarda unos minutos)...")
aucs_no, oof_no = cv_auc(StratifiedKFold(n_splits=N_SPLITS, shuffle=False))

# 3) CV barajada (control: los folds mezclan meses)
print("CV barajada (5 folds mezclando meses)...")
aucs_sh, oof_sh = cv_auc(
    StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
)

# 4) Referencia out-of-time (mismo pipe_full ya ajustado con train)
auc_oot = roc_auc_score(y_valid, pipe_full.predict_proba(X_valid)[:, 1])

print("\nRESULTADOS (ROC-AUC):")
print(f"  1) In-sample (train completo) : {auc_in:.4f}")
print(f"  2) CV sin barajar             : folds={['%.4f' % a for a in aucs_no]}  OOF={oof_no:.4f}")
print(f"  3) CV barajada (control)      : folds={['%.4f' % a for a in aucs_sh]}  OOF={oof_sh:.4f}")
print(f"  4) Ref. out-of-time (valid)   : {auc_oot:.4f}")

print("\nLECTURA:")
print("  - (1) alto => el modelo aprende: la anomalía no es de capacidad de ajuste.")
print("  - (2) invertido (~0.44) con folds en bloque temporal => reproduce la anomalía.")
print("  - (3) normalizado, cercano a (4) => al mezclar meses desaparece la inversión:")
print("    el confusor era la estructura temporal de los folds, no la calibración.")