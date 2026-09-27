"""
scripts/03_train_baseline_logistic.py

Entrena y compara el baseline (DummyClassifier) contra la Regresión
Logística, sobre el dataset completo de snapshots con el split temporal
ya decidido (Opción A).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd

from src.data.loader import load_online_retail
from src.data.cleaning import get_clean_sales
from src.features.snapshots import generate_snapshot_dates, build_multi_snapshot_dataset
from src.splitting.temporal_split import temporal_split
from src.models.baseline import build_dummy_pipeline, NUMERIC_FEATURES
from src.models.logistic import build_logistic_pipeline
from src.evaluation.metrics import evaluate_binary

df = get_clean_sales(load_online_retail())
min_date = df["InvoiceDate"].min()
max_date = df["InvoiceDate"].max()

snapshot_dates = generate_snapshot_dates(min_date, max_date, horizon_days=90)
dataset = build_multi_snapshot_dataset(
    df, snapshot_dates, horizon_days=90, lookback_days=365
)

train_end = snapshot_dates[12]
valid_end = snapshot_dates[16]
train, valid, test = temporal_split(
    dataset, date_col="cutoff_date", train_end=train_end, valid_end=valid_end
)

models = {
    "dummy": build_dummy_pipeline(),
    "logistic": build_logistic_pipeline(),
}

for model_name, pipeline in models.items():
    pipeline.fit(train[NUMERIC_FEATURES], train["churn"])
    print(f"=== {model_name} ===")
    for split_name, split in [("train", train), ("valid", valid), ("test", test)]:
        proba = pipeline.predict_proba(split[NUMERIC_FEATURES])[:, 1]
        metrics = evaluate_binary(split["churn"].values, proba)
        print(
            f"  {split_name}: "
            f"pr_auc={metrics['pr_auc']:.4f}  "
            f"roc_auc={metrics['roc_auc']:.4f}  "
            f"recall={metrics['recall']:.4f}  "
            f"precision={metrics['precision']:.4f}"
        )
    print()