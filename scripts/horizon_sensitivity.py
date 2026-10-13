"""
scripts/horizon_sensitivity.py

Sensibilidad del horizonte de churn (CHURN_HORIZONS: 60, 90 y 120 días).

Para cada horizonte h se reconstruye el dataset de snapshots con su propia
etiqueta (sin compra en los h días posteriores al corte) y se entrenan los
mismos modelos con las mismas features base (NUMERIC_FEATURES).

Para que los horizontes sean comparables:
- Las fechas de corte de train y valid son las mismas en calendario para los
  tres (las del horizonte de referencia de 90 días: snapshot 12 y 16).
- El embargo entre splits es igual a h (sin solape de ventanas de etiqueta).
- Los snapshots disponibles dependen de h (el último corte deja h días de
  futuro observable), así que valid y test tienen tamaños distintos por
  horizonte; la tabla los muestra.

Notas de lectura:
- ROC-AUC no depende de la prevalencia, así que es la métrica comparable
  entre horizontes. PR-AUC sí depende de ella: se compara con la prevalencia
  de test (suelo de un clasificador sin información).
- Intervalos: bootstrap por cliente (un mismo cliente aparece en varios
  snapshots), solo sobre el conjunto de test.

Salida: tabla por pantalla y reports/horizon_sensitivity.csv

Uso:
    uv run python scripts/horizon_sensitivity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config.settings import CHURN_HORIZONS, REPORTS_DIR
from src.data.cleaning import get_clean_sales
from src.data.loader import load_online_retail
from src.features.snapshots import build_multi_snapshot_dataset, generate_snapshot_dates
from src.models.baseline import NUMERIC_FEATURES
from src.models.logistic import build_logistic_pipeline
from src.models.random_forest import build_random_forest_pipeline
from src.models.xgboost_model import build_xgboost_pipeline, compute_scale_pos_weight
from src.splitting.temporal_split import temporal_split

REFERENCE_HORIZON = 90
SPLIT_SNAPSHOT_IDX = (12, 16)  # mismos índices que el resto de scripts
LOOKBACK_DAYS = 365
N_BOOT = 200
SEED = 42


def cluster_bootstrap_ci(y, p, customers, n_boot=N_BOOT, seed=SEED):
    """IC95% (percentiles) de ROC-AUC y PR-AUC remuestreando clientes."""
    rng = np.random.default_rng(seed)
    _, codes = np.unique(customers, return_inverse=True)
    codes = codes.ravel()
    n_cust = int(codes.max()) + 1
    roc, pr = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_cust, n_cust), minlength=n_cust)[codes]
        roc.append(roc_auc_score(y, p, sample_weight=w))
        pr.append(average_precision_score(y, p, sample_weight=w))
    return np.percentile(roc, [2.5, 97.5]), np.percentile(pr, [2.5, 97.5])


def build_models(y_train):
    weight = compute_scale_pos_weight(y_train)
    return {
        "logistic": build_logistic_pipeline(),
        "random_forest": build_random_forest_pipeline(),
        "xgboost": build_xgboost_pipeline(scale_pos_weight=weight),
    }


df = get_clean_sales(load_online_retail())
min_date, max_date = df["InvoiceDate"].min(), df["InvoiceDate"].max()

ref_dates = generate_snapshot_dates(min_date, max_date, horizon_days=REFERENCE_HORIZON)
train_end = ref_dates[SPLIT_SNAPSHOT_IDX[0]]
valid_end = ref_dates[SPLIT_SNAPSHOT_IDX[1]]
print(f"Cortes de split (fijos en calendario): train_end={train_end}, valid_end={valid_end}\n")

datasets = {}
for h in CHURN_HORIZONS:
    dates = generate_snapshot_dates(min_date, max_date, horizon_days=h)
    datasets[h] = build_multi_snapshot_dataset(
        df, dates, horizon_days=h, lookback_days=LOOKBACK_DAYS
    )

# --- A) Prevalencia de churn sobre los mismos cortes para todos los horizontes
common_last = max_date - pd.Timedelta(days=max(CHURN_HORIZONS))
print(f"A) %churn en cortes comunes (cutoff <= {common_last.date()}):")
for h in CHURN_HORIZONS:
    d = datasets[h]
    d = d[d["cutoff_date"] <= common_last]
    print(f"   h={h:>3}: {d['churn'].mean() * 100:5.1f}%  ({len(d)} filas)")
print()

# --- B) Modelos por horizonte
rows = []
for h in CHURN_HORIZONS:
    train, valid, test = temporal_split(
        datasets[h], "cutoff_date", train_end, valid_end, gap_days=h
    )
    if min(len(train), len(valid), len(test)) == 0:
        print(f"h={h}: algún split vacío con estas fechas; se omite.\n")
        continue
    print(
        f"B) h={h}: train {len(train)} filas/{train['cutoff_date'].nunique()} snaps "
        f"(churn {train['churn'].mean() * 100:.1f}%) | valid {len(valid)}/"
        f"{valid['cutoff_date'].nunique()} (churn {valid['churn'].mean() * 100:.1f}%) | "
        f"test {len(test)}/{test['cutoff_date'].nunique()} "
        f"(churn {test['churn'].mean() * 100:.1f}%)"
    )
    for name, pipe in build_models(train["churn"]).items():
        pipe.fit(train[NUMERIC_FEATURES], train["churn"])
        p_valid = pipe.predict_proba(valid[NUMERIC_FEATURES])[:, 1]
        p_test = pipe.predict_proba(test[NUMERIC_FEATURES])[:, 1]
        y_test = test["churn"].to_numpy()
        roc_ci, pr_ci = cluster_bootstrap_ci(
            y_test, p_test, test["Customer ID"].to_numpy()
        )
        rows.append(
            {
                "horizon_days": h,
                "model": name,
                "valid_snapshots": valid["cutoff_date"].nunique(),
                "test_snapshots": test["cutoff_date"].nunique(),
                "test_rows": len(test),
                "test_prevalence": y_test.mean(),
                "valid_roc_auc": roc_auc_score(valid["churn"], p_valid),
                "test_roc_auc": roc_auc_score(y_test, p_test),
                "test_roc_auc_lo": roc_ci[0],
                "test_roc_auc_hi": roc_ci[1],
                "valid_pr_auc": average_precision_score(valid["churn"], p_valid),
                "test_pr_auc": average_precision_score(y_test, p_test),
                "test_pr_auc_lo": pr_ci[0],
                "test_pr_auc_hi": pr_ci[1],
            }
        )
    print()

res = pd.DataFrame(rows)
out_path = Path(REPORTS_DIR) / "horizon_sensitivity.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(out_path, index=False)

print("C) Resultados en test (IC95% bootstrap por cliente):")
for _, r in res.iterrows():
    print(
        f"   h={int(r.horizon_days):>3} {r.model:<14} "
        f"ROC-AUC {r.test_roc_auc:.4f} [{r.test_roc_auc_lo:.4f}, {r.test_roc_auc_hi:.4f}] | "
        f"PR-AUC {r.test_pr_auc:.4f} [{r.test_pr_auc_lo:.4f}, {r.test_pr_auc_hi:.4f}] "
        f"(prevalencia {r.test_prevalence:.4f})"
    )
print(f"\nGuardado en {out_path}")
