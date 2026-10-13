"""
scripts/walk_forward_backtest.py

Backtest rodante (walk-forward) con embargo: en lugar de un único split
train/valid/test, para cada snapshot de test posible se entrena con todo lo
anterior (ventana expansiva) dejando un embargo de HORIZON_DAYS entre el
último corte de train y el de test.

Cada snapshot se evalúa una sola vez (periodos de test disjuntos), con el
mayor historial de entrenamiento disponible en ese momento. Sirve para ver
el rendimiento por periodo y su variabilidad (incluida la estacionalidad
navideña), en vez de depender de 2-3 snapshots concretos.

Limitaciones a tener en cuenta al leer los resultados:
- En los primeros folds el modelo solo ha visto un año o menos de datos, y
  las features estacionales (mes) no cubren un ciclo anual completo.
- La prevalencia de churn cambia entre periodos; ROC-AUC es la métrica
  comparable, PR-AUC se lee junto a la prevalencia.

Salida: tabla por fold, resumen por modelo y reports/walk_forward_backtest.csv

Uso:
    uv run python scripts/walk_forward_backtest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config.settings import REPORTS_DIR
from src.data.cleaning import get_clean_sales
from src.data.loader import load_online_retail
from src.features.snapshots import build_multi_snapshot_dataset, generate_snapshot_dates
from src.models.baseline import NUMERIC_FEATURES
from src.models.logistic import build_logistic_pipeline
from src.models.random_forest import build_random_forest_pipeline
from src.models.xgboost_model import build_xgboost_pipeline, compute_scale_pos_weight
from src.splitting.walk_forward import walk_forward_folds

HORIZON_DAYS = 90
LOOKBACK_DAYS = 365
MIN_TRAIN_SNAPSHOTS = 9


def build_models(y_train):
    weight = compute_scale_pos_weight(y_train)
    return {
        "logistic": build_logistic_pipeline(),
        "random_forest": build_random_forest_pipeline(),
        "xgboost": build_xgboost_pipeline(scale_pos_weight=weight),
    }


df = get_clean_sales(load_online_retail())
dates = generate_snapshot_dates(
    df["InvoiceDate"].min(), df["InvoiceDate"].max(), horizon_days=HORIZON_DAYS
)
dataset = build_multi_snapshot_dataset(
    df, dates, horizon_days=HORIZON_DAYS, lookback_days=LOOKBACK_DAYS
)

folds = walk_forward_folds(
    dataset["cutoff_date"].unique(),
    embargo_days=HORIZON_DAYS,
    min_train_snapshots=MIN_TRAIN_SNAPSHOTS,
)
print(
    f"{len(folds)} folds; embargo {HORIZON_DAYS} d; "
    f"mínimo {MIN_TRAIN_SNAPSHOTS} snapshots de train\n"
)

rows = []
for train_end, test_date in folds:
    train = dataset[dataset["cutoff_date"] <= train_end]
    test = dataset[dataset["cutoff_date"] == test_date]
    assert test_date - train_end >= pd.Timedelta(days=HORIZON_DAYS)  # sin solape de etiquetas
    y_test = test["churn"].to_numpy()

    line = (
        f"test {test_date.date()} (train hasta {train_end.date()}, "
        f"{train['cutoff_date'].nunique()} snaps/{len(train)} filas; "
        f"test {len(test)} filas, churn {y_test.mean() * 100:.1f}%)"
    )
    metrics = []
    for name, pipe in build_models(train["churn"]).items():
        pipe.fit(train[NUMERIC_FEATURES], train["churn"])
        p = pipe.predict_proba(test[NUMERIC_FEATURES])[:, 1]
        roc, pr = roc_auc_score(y_test, p), average_precision_score(y_test, p)
        metrics.append(f"{name} {roc:.3f}")
        rows.append(
            {
                "test_date": test_date,
                "train_end": train_end,
                "train_snapshots": train["cutoff_date"].nunique(),
                "train_rows": len(train),
                "test_rows": len(test),
                "test_prevalence": y_test.mean(),
                "model": name,
                "roc_auc": roc,
                "pr_auc": pr,
            }
        )
    print(f"{line}\n    ROC-AUC: " + " | ".join(metrics))

res = pd.DataFrame(rows)

print("\nResumen entre folds (ROC-AUC):")
summary = res.groupby("model")["roc_auc"].agg(["mean", "std", "min", "max"])
print(summary.round(4).to_string())

pivot = res.pivot(index="test_date", columns="model", values="roc_auc")
best = pivot.idxmax(axis=1).value_counts()
print("\nModelo con mejor ROC-AUC por fold:", dict(best))

out_path = Path(REPORTS_DIR) / "walk_forward_backtest.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(out_path, index=False)
print(f"\nGuardado en {out_path}")
