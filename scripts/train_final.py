"""
scripts/train_final.py

Entrena y guarda el modelo FINAL del TFM como artefacto reutilizable por la API:

    Random Forest (ajustado SOLO con train)
    + calibración Platt/sigmoid (ajustada SOLO con valid, modelo base congelado)
    + umbral operativo (recall >= 0.80, elegido SOLO en valid)

Mismo protocolo que evaluate_thresholds.py / analyze_shap.py: test no se usa
para entrenar, calibrar ni elegir el umbral; solo se evalúa y se registra.

Genera en models/ (o en --output-dir):
    churn_model_<version>.joblib         modelo calibrado (incluye el preprocesamiento)
    churn_model_<version>_metadata.json  features, umbral, versiones, hash de datos, métricas

Si el umbral reobtenido no coincide con el esperado (0.599), el script FALLA
(código de salida 1) y no escribe nada: así se detecta cualquier desviación
respecto al modelo documentado en el TFM.

Uso:
    uv run python scripts/train_final.py
    uv run python scripts/train_final.py --output-dir /ruta/prueba   # no toca models/
    uv run python scripts/train_final.py --skip-threshold-check      # solo diagnóstico
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import brier_score_loss, confusion_matrix

from src.config.settings import MODELS_DIR, RAW_DATASET_FILE
from src.data.cleaning import get_clean_sales
from src.data.loader import load_online_retail
from src.evaluation.calibration import (
    calibrate_prefit,
    calibration_table,
    expected_calibration_error,
)
from src.evaluation.metrics import evaluate_binary
from src.evaluation.threshold import threshold_for_min_recall
from src.features.snapshots import build_multi_snapshot_dataset, generate_snapshot_dates
from src.features.temporal import POST_HOLIDAY_MONTHS
from src.models.baseline import NUMERIC_FEATURES
from src.models.random_forest import build_random_forest_pipeline
from src.splitting.temporal_split import temporal_split

# --- Parámetros del protocolo (no cambiar sin reabrir el TFM) ---
MODEL_VERSION = "1.0.0"
HORIZON_DAYS = 90
LOOKBACK_DAYS = 365
TRAIN_END_IDX = 12
VALID_END_IDX = 16
TARGET_RECALL = 0.80
RANDOM_STATE = 42
EXPECTED_THRESHOLD = 0.599
THRESHOLD_DECIMALS = 3  # el 0.599 documentado está redondeado a 3 decimales


def sha256_file(path: Path) -> str | None:
    """Hash del archivo de datos original (trazabilidad); None si no existe."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    """Commit actual y si hay cambios sin commitear; None si no es un repo git."""

    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def split_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "unique_customers": int(df["Customer ID"].nunique()),
        "churn_prevalence": round(float(df["churn"].mean()), 6),
        "first_cutoff": df["cutoff_date"].min().strftime("%Y-%m-%d"),
        "last_cutoff": df["cutoff_date"].max().strftime("%Y-%m-%d"),
    }


def evaluate_block(y: np.ndarray, proba: np.ndarray, thr: float) -> dict:
    """Métricas discriminativas, de calibración y matriz de confusión en el umbral."""
    m = evaluate_binary(y, proba, threshold=thr)
    table = calibration_table(y, proba, n_bins=10)
    tn, fp, fn, tp = confusion_matrix(y, (proba >= thr).astype(int), labels=[0, 1]).ravel()
    return {
        "pr_auc": round(float(m["pr_auc"]), 6),
        "roc_auc": round(float(m["roc_auc"]), 6),
        "precision": round(float(m["precision"]), 6),
        "recall": round(float(m["recall"]), 6),
        "f1": round(float(m["f1"]), 6),
        "brier": round(float(brier_score_loss(y, proba)), 6),
        "ece": round(expected_calibration_error(table), 6),
        "mean_predicted_proba": round(float(proba.mean()), 6),
        "observed_prevalence": round(float(y.mean()), 6),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument(
        "--skip-threshold-check",
        action="store_true",
        help="No abortar si el umbral difiere de 0.599 (solo para diagnóstico).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"ENTRENAMIENTO DEL MODELO FINAL v{MODEL_VERSION}")
    print("=" * 70)

    # 1) Datos y split temporal (idéntico a evaluate_thresholds.py)
    df = get_clean_sales(load_online_retail())
    snapshot_dates = generate_snapshot_dates(
        df["InvoiceDate"].min(), df["InvoiceDate"].max(), horizon_days=HORIZON_DAYS
    )
    dataset = build_multi_snapshot_dataset(
        df, snapshot_dates, horizon_days=HORIZON_DAYS, lookback_days=LOOKBACK_DAYS
    )
    train_end = snapshot_dates[TRAIN_END_IDX]
    valid_end = snapshot_dates[VALID_END_IDX]
    train, valid, test = temporal_split(
        dataset, date_col="cutoff_date", train_end=train_end, valid_end=valid_end
    )
    splits = {"train": train, "valid": valid, "test": test}
    for name, s in splits.items():
        print(f"  {name:<6} n={len(s):>6}  prevalencia churn={s['churn'].mean():.4f}")

    # 2) RF solo con train -> Platt solo con valid (modelo base congelado)
    pipe = build_random_forest_pipeline(random_state=RANDOM_STATE)
    pipe.fit(train[NUMERIC_FEATURES], train["churn"])

    model = calibrate_prefit(pipe, method="sigmoid")
    model.fit(valid[NUMERIC_FEATURES], valid["churn"])

    # 3) Umbral operativo elegido SOLO en valid
    proba = {
        name: model.predict_proba(s[NUMERIC_FEATURES])[:, 1] for name, s in splits.items()
    }
    thr = threshold_for_min_recall(valid["churn"].values, proba["valid"], TARGET_RECALL)
    print(f"\nUmbral operativo (recall >= {TARGET_RECALL} en valid): {thr:.6f}")

    if round(thr, THRESHOLD_DECIMALS) != EXPECTED_THRESHOLD:
        msg = (
            f"El umbral reobtenido ({thr:.6f}) no coincide con el documentado "
            f"({EXPECTED_THRESHOLD}). Revisa versiones de librerías, datos o código."
        )
        if args.skip_threshold_check:
            print(f"AVISO: {msg}")
        else:
            print(f"ERROR: {msg}\nNo se ha guardado ningún artefacto.", file=sys.stderr)
            return 1
    else:
        print(f"  OK: coincide con el umbral documentado ({EXPECTED_THRESHOLD}).")

    # 4) Métricas (test solo se evalúa, nunca se usa para decidir)
    metrics = {
        name: evaluate_block(s["churn"].values, proba[name], thr)
        for name, s in splits.items()
        if name != "train"
    }
    for name, m in metrics.items():
        cm = m["confusion_matrix"]
        print(
            f"  [{name}] PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
            f"precision={m['precision']:.4f} recall={m['recall']:.4f} F1={m['f1']:.4f} "
            f"Brier={m['brier']:.4f} ECE={m['ece']:.4f} | "
            f"TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}"
        )

    # 5) Guardado: modelo + metadatos
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"churn_model_{MODEL_VERSION}.joblib"
    meta_path = args.output_dir / f"churn_model_{MODEL_VERSION}_metadata.json"

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "RandomForestClassifier + Platt (sigmoid) calibration",
        "protocol": {
            "base_model_fit": "train",
            "calibration_fit": "valid (base model frozen)",
            "threshold_selection": "valid",
            "test_usage": "evaluation only",
        },
        "features": list(NUMERIC_FEATURES),
        "seasonal_features_derived_from": "cutoff_date",
        "post_holiday_months": sorted(POST_HOLIDAY_MONTHS),
        "threshold": float(thr),
        "threshold_rule": f"max precision with recall >= {TARGET_RECALL} on valid",
        "target_recall": TARGET_RECALL,
        "churn_horizon_days": HORIZON_DAYS,
        "lookback_days": LOOKBACK_DAYS,
        "random_state": RANDOM_STATE,
        "split": {
            "train_end": train_end.strftime("%Y-%m-%d"),
            "valid_end": valid_end.strftime("%Y-%m-%d"),
            "train": split_summary(train),
            "valid": split_summary(valid),
            "test": split_summary(test),
        },
        "metrics": metrics,
        "data": {
            "file": RAW_DATASET_FILE.name,
            "sha256": sha256_file(RAW_DATASET_FILE),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "git": git_info(),
    }
    if metadata["data"]["sha256"] is None:
        print(f"AVISO: no se encontró {RAW_DATASET_FILE}; el hash de datos queda en null.")

    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6) Verificación de ida y vuelta: el artefacto recargado debe predecir igual
    reloaded = joblib.load(model_path)
    sample = test[NUMERIC_FEATURES].head(500)
    diff = float(
        np.abs(
            reloaded.predict_proba(sample)[:, 1] - model.predict_proba(sample)[:, 1]
        ).max()
    )
    if diff > 1e-12:
        print(f"ERROR: el modelo recargado difiere del original (max diff={diff}).", file=sys.stderr)
        return 1
    print(f"\nVerificación de recarga: max|diff| = {diff:.1e}")
    print(f"Modelo:    {model_path}")
    print(f"Metadatos: {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())