import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import confusion_matrix

from src.data.loader import load_online_retail
from src.data.cleaning import get_clean_sales
from src.features.snapshots import generate_snapshot_dates, build_multi_snapshot_dataset
from src.splitting.temporal_split import temporal_split
from src.models.baseline import NUMERIC_FEATURES
from src.models.logistic import build_logistic_pipeline
from src.models.random_forest import build_random_forest_pipeline
from src.models.xgboost_model import build_xgboost_pipeline, compute_scale_pos_weight
from src.evaluation.metrics import evaluate_binary
from src.evaluation.calibration import (
    brier,
    calibration_table,
    expected_calibration_error,
    plot_calibration_curves,
)
from src.evaluation.threshold import best_threshold_f1, threshold_for_min_recall

TARGET_RECALL = 0.80

# --- Datos y split (idéntico a train_all_models.py) ---
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

# --- Entrenamiento (solo con train) ---
models = {
    "logistic": build_logistic_pipeline(),
    "random_forest": build_random_forest_pipeline(),
    "xgboost": build_xgboost_pipeline(
        scale_pos_weight=compute_scale_pos_weight(train["churn"])
    ),
}
proba = {"valid": {}, "test": {}}
for name, pipeline in models.items():
    pipeline.fit(train[NUMERIC_FEATURES], train["churn"])
    proba["valid"][name] = pipeline.predict_proba(valid[NUMERIC_FEATURES])[:, 1]
    proba["test"][name] = pipeline.predict_proba(test[NUMERIC_FEATURES])[:, 1]

splits = {"valid": valid, "test": test}

# --- 1) Diagnóstico de calibración ---
print("=" * 70)
print("CALIBRACIÓN (sin recalibrar)")
print("=" * 70)
for split_name, split in splits.items():
    y = split["churn"].values
    print(f"\n[{split_name}] prevalencia observada de churn: {y.mean():.4f}")
    tables = {}
    for name in models:
        p = proba[split_name][name]
        table = calibration_table(y, p, n_bins=10)
        tables[name] = table
        print(
            f"  {name:<14} prob. media predicha={p.mean():.4f}  "
            f"Brier={brier(y, p):.4f}  ECE={expected_calibration_error(table):.4f}"
        )
    plot_calibration_curves(
        tables,
        title=f"Curvas de calibración ({split_name})",
        output_path=ROOT / "reports" / f"calibration_{split_name}.png",
    )

# --- 2) Threshold elegido en valid, evaluado UNA vez en test ---
print("\n" + "=" * 70)
print(f"THRESHOLDS (elegidos en valid; objetivo alternativo: recall >= {TARGET_RECALL})")
print("=" * 70)
y_valid = valid["churn"].values
y_test = test["churn"].values

for name in models:
    thresholds = {
        "F1 max": best_threshold_f1(y_valid, proba["valid"][name]),
        f"recall>={TARGET_RECALL}": threshold_for_min_recall(
            y_valid, proba["valid"][name], TARGET_RECALL
        ),
        "0.5 (ref.)": 0.5,
    }
    print(f"\n=== {name} ===")
    for label, thr in thresholds.items():
        for split_name, y in (("valid", y_valid), ("test", y_test)):
            p = proba[split_name][name]
            m = evaluate_binary(y, p, threshold=thr)
            tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int)).ravel()
            print(
                f"  {label:<12} thr={thr:.3f} | {split_name:<5} "
                f"precision={m['precision']:.4f} recall={m['recall']:.4f} "
                f"f1={m['f1']:.4f} | TN={tn} FP={fp} FN={fn} TP={tp}"
            ) 