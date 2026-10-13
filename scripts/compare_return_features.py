"""
scripts/compare_return_features.py

Ablación: modelos con las features base (NUMERIC_FEATURES) frente a base +
devoluciones (FEATURES_WITH_RETURNS), con el mismo split temporal y el mismo
embargo. Muestra métricas en valid/test y un intervalo bootstrap por cliente
de la diferencia (con devoluciones - sin devoluciones) en test, para ver si
la mejora supera el ruido muestral.

Uso:
    uv run python scripts/compare_return_features.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config.settings import SPLIT_EMBARGO_DAYS
from src.data.cleaning import get_clean_sales
from src.data.loader import load_online_retail
from src.features.returns import get_cancellations
from src.features.snapshots import build_multi_snapshot_dataset, generate_snapshot_dates
from src.models.baseline import FEATURES_WITH_RETURNS, NUMERIC_FEATURES
from src.models.logistic import build_logistic_pipeline
from src.models.random_forest import build_random_forest_pipeline
from src.models.xgboost_model import build_xgboost_pipeline, compute_scale_pos_weight
from src.splitting.temporal_split import summarize_split, temporal_split

HORIZON_DAYS = 90
N_BOOT = 300
SEED = 42

raw = load_online_retail()
df = get_clean_sales(raw)
cancellations = get_cancellations(raw)

snapshot_dates = generate_snapshot_dates(
    df["InvoiceDate"].min(), df["InvoiceDate"].max(), horizon_days=HORIZON_DAYS
)
dataset = build_multi_snapshot_dataset(
    df,
    snapshot_dates,
    horizon_days=HORIZON_DAYS,
    lookback_days=365,
    cancellations=cancellations,
)
train, valid, test = temporal_split(
    dataset,
    date_col="cutoff_date",
    train_end=snapshot_dates[12],
    valid_end=snapshot_dates[16],
    gap_days=SPLIT_EMBARGO_DAYS,
)
print(summarize_split(train, valid, test).to_string(index=False), "\n")

xgb_weight = compute_scale_pos_weight(train["churn"])
BUILDERS = {
    "logistic": lambda feats: build_logistic_pipeline(numeric_features=feats),
    "random_forest": lambda feats: build_random_forest_pipeline(numeric_features=feats),
    "xgboost": lambda feats: build_xgboost_pipeline(
        numeric_features=feats, scale_pos_weight=xgb_weight
    ),
}


def cluster_bootstrap_diff(y, p_base, p_new, customers, n_boot=N_BOOT, seed=SEED):
    """
    Bootstrap por cliente (un mismo cliente aparece en varios snapshots, así
    que se remuestrean clientes, no filas). Devuelve (media, IC95 bajo, IC95
    alto) de la diferencia en ROC-AUC y PR-AUC: nuevo - base.
    """
    rng = np.random.default_rng(seed)
    _, codes = np.unique(customers, return_inverse=True)
    codes = codes.ravel()
    n_cust = int(codes.max()) + 1
    d_roc, d_pr = [], []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_cust, n_cust), minlength=n_cust)
        w = counts[codes]
        if w.sum() == 0:
            continue
        d_roc.append(
            roc_auc_score(y, p_new, sample_weight=w) - roc_auc_score(y, p_base, sample_weight=w)
        )
        d_pr.append(
            average_precision_score(y, p_new, sample_weight=w)
            - average_precision_score(y, p_base, sample_weight=w)
        )
    out = []
    for d in (d_roc, d_pr):
        lo, hi = np.percentile(d, [2.5, 97.5])
        out.append((float(np.mean(d)), float(lo), float(hi)))
    return out


for name, build in BUILDERS.items():
    probs = {}
    print(f"=== {name} ===")
    for label, feats in (("base", NUMERIC_FEATURES), ("+devoluciones", FEATURES_WITH_RETURNS)):
        pipe = build(feats)
        pipe.fit(train[feats], train["churn"])
        row = []
        for split_name, split in (("valid", valid), ("test", test)):
            p = pipe.predict_proba(split[feats])[:, 1]
            probs[(label, split_name)] = p
            row.append(
                f"{split_name}: roc_auc={roc_auc_score(split['churn'], p):.4f} "
                f"pr_auc={average_precision_score(split['churn'], p):.4f}"
            )
        print(f"  {label:<14} " + " | ".join(row))

    (roc_m, roc_lo, roc_hi), (pr_m, pr_lo, pr_hi) = cluster_bootstrap_diff(
        test["churn"].to_numpy(),
        probs[("base", "test")],
        probs[("+devoluciones", "test")],
        test["Customer ID"].to_numpy(),
    )
    print(
        f"  Δ test (nuevo-base), IC95% bootstrap por cliente: "
        f"ROC-AUC {roc_m:+.4f} [{roc_lo:+.4f}, {roc_hi:+.4f}] | "
        f"PR-AUC {pr_m:+.4f} [{pr_lo:+.4f}, {pr_hi:+.4f}]\n"
    )
