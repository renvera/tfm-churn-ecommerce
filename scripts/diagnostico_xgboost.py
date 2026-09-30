"""
scripts/diagnostico_xgboost.py

Diagnóstico de la anomalía de la sección 2: XGBoost con Platt-CV en train
da ROC-AUC ~0.27 (ranking invertido). Comprobamos si las predicciones
out-of-fold del propio XGBoost (sin calibrar) ya salen invertidas.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score

from src.data.loader import load_online_retail
from src.data.cleaning import get_clean_sales
from src.features.snapshots import (
    build_multi_snapshot_dataset,
    generate_snapshot_dates,
)
from src.splitting.temporal_split import temporal_split
from src.models.baseline import NUMERIC_FEATURES
from src.models.xgboost_model import (
    build_xgboost_pipeline,
    compute_scale_pos_weight,
)

# Mismo pipeline de datos que evaluate_thresholds.py
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

pipe_xgb = build_xgboost_pipeline(
    scale_pos_weight=compute_scale_pos_weight(train["churn"])
)

# AUC por fold (cada fold reentrena el pipeline desde cero)
aucs = cross_val_score(
    pipe_xgb, train[NUMERIC_FEATURES], train["churn"], cv=5, scoring="roc_auc"
)
print("AUC por fold (5-fold CV en train):", np.round(aucs, 4))

# AUC de las predicciones out-of-fold agrupadas (lo que ve el calibrador)
p_oof = cross_val_predict(
    pipe_xgb, train[NUMERIC_FEATURES], train["churn"], cv=5,
    method="predict_proba",
)[:, 1]
print("AUC out-of-fold agrupado:", round(roc_auc_score(train["churn"], p_oof), 4))