"""
scripts/experimento_monetary_neto.py

Compara Monetary bruto vs neto con el MISMO protocolo del modelo final:
RF (train) + Platt (valid) + umbral con recall >= 0.80 en valid; test solo evalúa.

Variantes:
  A_bruto        : features actuales (monetary = ventas)
  B_neto         : monetary_net (ventas - devoluciones) en lugar de monetary
  C_bruto+ratio  : features actuales + return_ratio (devuelto / vendido)

Como un solo RF con semilla 42 puede confundir ruido con mejora, cada variante
se repite con varias semillas y se reporta media y desviación.

Uso:
    uv run python scripts/experimento_monetary_neto.py
    uv run python scripts/experimento_monetary_neto.py --seeds 42 1 2 3 4
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data.cleaning import get_clean_sales
from src.data.loader import load_online_retail
from src.evaluation.calibration import (
    calibrate_prefit,
    calibration_table,
    expected_calibration_error,
)
from src.evaluation.metrics import evaluate_binary
from src.evaluation.threshold import threshold_for_min_recall
from src.features.returns import add_return_features, get_cancellations
from src.features.snapshots import build_multi_snapshot_dataset, generate_snapshot_dates
from src.models.baseline import NUMERIC_FEATURES
from src.models.random_forest import build_random_forest_pipeline
from src.splitting.temporal_split import temporal_split

HORIZON_DAYS, LOOKBACK_DAYS = 90, 365
TRAIN_END_IDX, VALID_END_IDX = 12, 16
TARGET_RECALL = 0.80

VARIANTS = {
    "A_bruto": NUMERIC_FEATURES,
    "B_neto": [("monetary_net" if f == "monetary" else f) for f in NUMERIC_FEATURES],
    "C_bruto+ratio": NUMERIC_FEATURES + ["return_ratio"],
}


def run_variant(train, valid, test, features, seed):
    pipe = build_random_forest_pipeline(numeric_features=features, random_state=seed)
    pipe.fit(train[features], train["churn"])
    model = calibrate_prefit(pipe, method="sigmoid")
    model.fit(valid[features], valid["churn"])

    p_valid = model.predict_proba(valid[features])[:, 1]
    p_test = model.predict_proba(test[features])[:, 1]
    thr = threshold_for_min_recall(valid["churn"].values, p_valid, TARGET_RECALL)

    m = evaluate_binary(test["churn"].values, p_test, threshold=thr)
    return {
        "threshold": thr,
        "valid_pr_auc": evaluate_binary(valid["churn"].values, p_valid, thr)["pr_auc"],
        "test_pr_auc": m["pr_auc"],
        "test_roc_auc": m["roc_auc"],
        "test_precision": m["precision"],
        "test_recall": m["recall"],
        "test_brier": brier_score_loss(test["churn"].values, p_test),
        "test_ece": expected_calibration_error(calibration_table(test["churn"].values, p_test)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "monetary_neto_vs_bruto.csv")
    args = ap.parse_args()

    raw = load_online_retail()
    sales = get_clean_sales(raw)
    cancellations = get_cancellations(raw)

    dates = generate_snapshot_dates(
        sales["InvoiceDate"].min(), sales["InvoiceDate"].max(), horizon_days=HORIZON_DAYS
    )
    dataset = build_multi_snapshot_dataset(
        sales, dates, horizon_days=HORIZON_DAYS, lookback_days=LOOKBACK_DAYS
    )
    dataset = add_return_features(dataset, cancellations)

    print("DIAGNÓSTICO DE DEVOLUCIONES (todas las filas cliente+snapshot)")
    print(f"  filas: {len(dataset)}")
    print(f"  con alguna devolución acumulada: {(dataset['return_amount'] > 0).mean():.2%}")
    print(f"  monetary_net <= 0:               {(dataset['monetary_net'] <= 0).mean():.2%}")
    print(f"  return_ratio medio (con devol.): "
          f"{dataset.loc[dataset['return_amount'] > 0, 'return_ratio'].mean():.3f}")
    print(f"  correlación monetary vs monetary_net: "
          f"{dataset['monetary'].corr(dataset['monetary_net']):.4f}")
    print(f"  churn medio con/sin devolución: "
          f"{dataset.loc[dataset['return_amount'] > 0, 'churn'].mean():.3f} / "
          f"{dataset.loc[dataset['return_amount'] == 0, 'churn'].mean():.3f}")

    train, valid, test = temporal_split(
        dataset, date_col="cutoff_date", train_end=dates[TRAIN_END_IDX], valid_end=dates[VALID_END_IDX]
    )

    rows = []
    for name, feats in VARIANTS.items():
        for seed in args.seeds:
            res = run_variant(train, valid, test, feats, seed)
            rows.append({"variant": name, "seed": seed, **res})
            print(f"  {name:<14} seed={seed:<3} test PR-AUC={res['test_pr_auc']:.4f} "
                  f"ROC-AUC={res['test_roc_auc']:.4f} thr={res['threshold']:.4f}")

    results = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    cols = ["threshold", "valid_pr_auc", "test_pr_auc", "test_roc_auc",
            "test_precision", "test_recall", "test_brier", "test_ece"]
    summary = results.groupby("variant")[cols].agg(["mean", "std"]).round(4)
    print("\nRESUMEN (media y desviación entre semillas)")
    print(summary.T.to_string())
    print(f"\nGuardado: {args.out}")


if __name__ == "__main__":
    main()