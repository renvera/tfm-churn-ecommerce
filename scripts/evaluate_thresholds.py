import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.data.loader import load_online_retail
from src.data.cleaning import get_clean_sales
from src.features.snapshots import build_multi_snapshot_dataset, generate_snapshot_dates
from src.splitting.temporal_split import temporal_split
from src.models.baseline import NUMERIC_FEATURES
from src.models.logistic import build_logistic_pipeline
from src.models.random_forest import build_random_forest_pipeline
from src.models.xgboost_model import build_xgboost_pipeline, compute_scale_pos_weight
from src.evaluation.metrics import evaluate_binary
from src.evaluation.calibration import (
    brier,
    build_calibrated_pipeline,
    calibrate_prefit,
    calibration_table,
    expected_calibration_error,
    plot_calibration_curves,
)
from src.evaluation.threshold import best_threshold_f1, threshold_for_min_recall

TARGET_RECALL = 0.80

# --- Datos y split temporal (idéntico a train_all_models.py) ---
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
splits = {"valid": valid, "test": test}
y_valid = valid["churn"].values
y_test = test["churn"].values

print("=" * 70)
print("PREVALENCIA POR SPLIT (diagnóstico de deriva temporal)")
print("=" * 70)
for name, split in (("train", train), ("valid", valid), ("test", test)):
    print(f"  {name:<6} n={len(split):>6}  prevalencia churn={split['churn'].mean():.4f}")


def diagnosticar(proba: dict, plot_name: str) -> dict:
    """Brier/ECE/AUCs por modelo y split; guarda curvas de calibración."""
    resumen = {}
    for split_name, split in splits.items():
        y = split["churn"].values
        print(f"\n[{split_name}] prevalencia observada de churn: {y.mean():.4f}")
        tables = {}
        for name, p in proba[split_name].items():
            table = calibration_table(y, p, n_bins=10)
            tables[name] = table
            resumen[(name, split_name)] = {
                "brier": brier(y, p),
                "ece": expected_calibration_error(table),
                "roc_auc": roc_auc_score(y, p),
                "pr_auc": average_precision_score(y, p),
            }
            r = resumen[(name, split_name)]
            print(
                f"  {name:<14} prob.media={p.mean():.4f}  Brier={r['brier']:.4f}  "
                f"ECE={r['ece']:.4f}  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}"
            )
        plot_calibration_curves(
            tables,
            title=f"Curvas de calibración ({plot_name}, {split_name})",
            output_path=ROOT / "reports" / f"calibration_{plot_name}_{split_name}.png",
        )
    return resumen


def evaluar_thresholds(proba: dict, etiqueta: str) -> None:
    """Umbrales elegidos en valid y evaluados en valid y test."""
    print("\n" + "=" * 70)
    print(f"THRESHOLDS {etiqueta} (elegidos en valid; objetivo: recall >= {TARGET_RECALL})")
    print("=" * 70)
    for name in proba["valid"]:
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


# --- Modelos base ajustados solo con train ---
models = {
    "logistic": build_logistic_pipeline(),
    "random_forest": build_random_forest_pipeline(),
    "xgboost": build_xgboost_pipeline(
        scale_pos_weight=compute_scale_pos_weight(train["churn"])
    ),
}
proba_raw = {"valid": {}, "test": {}}
for name, pipeline in models.items():
    pipeline.fit(train[NUMERIC_FEATURES], train["churn"])
    proba_raw["valid"][name] = pipeline.predict_proba(valid[NUMERIC_FEATURES])[:, 1]
    proba_raw["test"][name] = pipeline.predict_proba(test[NUMERIC_FEATURES])[:, 1]

print("\n" + "=" * 70)
print("1) SIN RECALIBRAR (referencia)")
print("=" * 70)
diag_raw = diagnosticar(proba_raw, "raw")
evaluar_thresholds(proba_raw, "SIN RECALIBRAR")

# --- Referencia: Platt con CV sobre train (no corrige la deriva temporal) ---
calibrated_traincv = {
    name: build_calibrated_pipeline(pipe, method="sigmoid", cv=5)
    for name, pipe in models.items()
}
proba_traincv = {"valid": {}, "test": {}}
for name, cal in calibrated_traincv.items():
    cal.fit(train[NUMERIC_FEATURES], train["churn"])
    proba_traincv["valid"][name] = cal.predict_proba(valid[NUMERIC_FEATURES])[:, 1]
    proba_traincv["test"][name] = cal.predict_proba(test[NUMERIC_FEATURES])[:, 1]

print("\n" + "=" * 70)
print("2) PLATT CON CV EN TRAIN (referencia: el mapa se aprende con")
print("   predicciones out-of-fold de train y no transfiere a test)")
print("=" * 70)
diagnosticar(proba_traincv, "sigmoid_traincv")

# --- Recalibración sobre valid con el modelo base congelado ---
calibrated_valid = {
    name: calibrate_prefit(pipe, method="sigmoid") for name, pipe in models.items()
}
proba_platt_valid = {"valid": {}, "test": {}}
for name, cal in calibrated_valid.items():
    cal.fit(valid[NUMERIC_FEATURES], valid["churn"])
    proba_platt_valid["valid"][name] = cal.predict_proba(valid[NUMERIC_FEATURES])[:, 1]
    proba_platt_valid["test"][name] = cal.predict_proba(test[NUMERIC_FEATURES])[:, 1]

print("\n" + "=" * 70)
print("3) PLATT AJUSTADO EN VALID (modelo base congelado)")
print("=" * 70)
diag_platt_valid = diagnosticar(proba_platt_valid, "platt")
evaluar_thresholds(proba_platt_valid, "PLATT EN VALID")

# --- Sensibilidad: isotonic ajustado en valid (solo calibración) ---
print("\n" + "=" * 70)
print("4) SENSIBILIDAD: ISOTONIC AJUSTADO EN VALID (solo Brier/ECE)")
print("=" * 70)
for name, pipe in models.items():
    cal = calibrate_prefit(pipe, method="isotonic")
    cal.fit(valid[NUMERIC_FEATURES], valid["churn"])
    for split_name, split in splits.items():
        y = split["churn"].values
        p = cal.predict_proba(split[NUMERIC_FEATURES])[:, 1]
        table = calibration_table(y, p, n_bins=10)
        print(
            f"  {name:<14} [{split_name}] Brier={brier(y, p):.4f}  "
            f"ECE={expected_calibration_error(table):.4f}"
        )

# --- Resumen para el TFM ---
print("\n" + "=" * 70)
print("RESUMEN EN TEST")
print("=" * 70)
for name in models:
    r0 = diag_raw[(name, "test")]
    r1 = diag_platt_valid[(name, "test")]
    print(f"  {name}")
    print(
        f"    sin calibrar : Brier={r0['brier']:.4f}  ECE={r0['ece']:.4f}  "
        f"ROC-AUC={r0['roc_auc']:.4f}  PR-AUC={r0['pr_auc']:.4f}"
    )
    print(
        f"    Platt(valid) : Brier={r1['brier']:.4f}  ECE={r1['ece']:.4f}  "
        f"ROC-AUC={r1['roc_auc']:.4f}  PR-AUC={r1['pr_auc']:.4f}"
    )