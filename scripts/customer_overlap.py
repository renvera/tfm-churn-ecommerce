"""
scripts/customer_overlap.py

Efecto de que un mismo cliente aparezca en train y en test.

En el split temporal un cliente tiene filas en varios snapshots, así que
muchos clientes de valid/test ya estaban en train. Esto no es leakage
temporal, pero las observaciones del mismo cliente no son independientes.
El script lo cuantifica de tres formas:

1) Solapamiento: % de filas y clientes de valid/test vistos en train.
2) Clientes vistos vs nuevos en test (escenario real: reentrenar con el
   historial y puntuar tanto clientes existentes como recién llegados).
   Los nuevos tienen menos historial, así que la comparación no aísla el
   efecto del solapamiento por sí sola.
3) Prueba controlada: los clientes se asignan al azar a dos mitades (A, B);
   se entrena solo con las filas de train de A y se evalúa en test sobre
   clientes de A (vistos) y de B (nunca vistos). Al ser una asignación
   aleatoria, ambos grupos son comparables y la diferencia mide el
   optimismo atribuible a que el cliente estuviera en train. Se repite con
   varias semillas.

Salida: tablas por pantalla y reports/customer_overlap.csv

Uso:
    uv run python scripts/customer_overlap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config.settings import REPORTS_DIR, SPLIT_EMBARGO_DAYS
from src.data.cleaning import get_clean_sales
from src.data.loader import load_online_retail
from src.evaluation.bootstrap import cluster_bootstrap_ci
from src.features.snapshots import build_multi_snapshot_dataset, generate_snapshot_dates
from src.models.baseline import NUMERIC_FEATURES
from src.models.logistic import build_logistic_pipeline
from src.models.random_forest import build_random_forest_pipeline
from src.models.xgboost_model import build_xgboost_pipeline, compute_scale_pos_weight
from src.splitting.temporal_split import temporal_split

HORIZON_DAYS = 90
LOOKBACK_DAYS = 365
N_SEEDS = 10


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
train, valid, test = temporal_split(
    dataset,
    "cutoff_date",
    dates[12],
    dates[16],
    gap_days=SPLIT_EMBARGO_DAYS,
)
y_test = test["churn"].to_numpy()
cust_test = test["Customer ID"].to_numpy()
seen_in_train = set(train["Customer ID"])
records = []

# --- 1) Solapamiento
print("1) Solapamiento con train")
for name, part in (("valid", valid), ("test", test)):
    seen = part["Customer ID"].isin(seen_in_train)
    n_cust = part["Customer ID"].nunique()
    n_seen = part.loc[seen, "Customer ID"].nunique()
    print(
        f"   {name}: {seen.mean() * 100:.1f}% de las filas son de clientes vistos en train; "
        f"{n_seen}/{n_cust} clientes únicos ({n_seen / n_cust * 100:.1f}%)"
    )
    records.append(
        {
            "analysis": "overlap",
            "group": name,
            "n_rows": len(part),
            "share_seen_rows": seen.mean(),
            "n_customers": n_cust,
            "n_customers_seen": n_seen,
        }
    )
print(
    f"   Snapshots por cliente en train: media {train.groupby('Customer ID').size().mean():.1f}, "
    f"mediana {train.groupby('Customer ID').size().median():.0f}\n"
)

# --- 2) Vistos vs nuevos en test
print("2) Test: clientes vistos en train vs clientes nuevos (IC95% bootstrap por cliente)")
is_seen = test["Customer ID"].isin(seen_in_train).to_numpy()
fitted = {}
for name, pipe in build_models(train["churn"]).items():
    pipe.fit(train[NUMERIC_FEATURES], train["churn"])
    fitted[name] = pipe.predict_proba(test[NUMERIC_FEATURES])[:, 1]

for name, p in fitted.items():
    for label, mask in (("todos", np.ones_like(is_seen)), ("vistos", is_seen), ("nuevos", ~is_seen)):
        ci = cluster_bootstrap_ci(y_test[mask], p[mask], cust_test[mask])
        roc, pr = roc_auc_score(y_test[mask], p[mask]), average_precision_score(y_test[mask], p[mask])
        print(
            f"   {name:<14} {label:<7} n={mask.sum():>6}  "
            f"ROC-AUC {roc:.4f} [{ci['roc_auc'][0]:.4f}, {ci['roc_auc'][1]:.4f}]  "
            f"PR-AUC {pr:.4f}  (churn {y_test[mask].mean() * 100:.1f}%)"
        )
        records.append(
            {
                "analysis": "seen_vs_new",
                "model": name,
                "group": label,
                "n_rows": int(mask.sum()),
                "prevalence": y_test[mask].mean(),
                "roc_auc": roc,
                "roc_auc_lo": ci["roc_auc"][0],
                "roc_auc_hi": ci["roc_auc"][1],
                "pr_auc": pr,
                "pr_auc_lo": ci["pr_auc"][0],
                "pr_auc_hi": ci["pr_auc"][1],
            }
        )
print()

# --- 3) Prueba controlada con asignación aleatoria de clientes
print(f"3) Prueba controlada (clientes asignados al azar a A/B, {N_SEEDS} semillas)")
all_customers = np.array(sorted(set(train["Customer ID"]) | set(test["Customer ID"])))
diffs = {m: [] for m in ("logistic", "random_forest", "xgboost")}
for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    group_a = set(rng.choice(all_customers, size=len(all_customers) // 2, replace=False))
    train_a = train[train["Customer ID"].isin(group_a)]
    in_a = test["Customer ID"].isin(group_a).to_numpy()
    seen = in_a & test["Customer ID"].isin(set(train_a["Customer ID"])).to_numpy()
    unseen = ~in_a
    for name, pipe in build_models(train_a["churn"]).items():
        pipe.fit(train_a[NUMERIC_FEATURES], train_a["churn"])
        p = pipe.predict_proba(test[NUMERIC_FEATURES])[:, 1]
        diffs[name].append(
            (roc_auc_score(y_test[seen], p[seen]), roc_auc_score(y_test[unseen], p[unseen]))
        )
for name, v in diffs.items():
    v = np.array(v)
    d = v[:, 0] - v[:, 1]
    print(
        f"   {name:<14} ROC-AUC vistos {v[:, 0].mean():.4f} | no vistos {v[:, 1].mean():.4f} | "
        f"diferencia media {d.mean():+.4f} (min {d.min():+.4f}, max {d.max():+.4f})"
    )
    records.append(
        {
            "analysis": "controlled_random_split",
            "model": name,
            "group": "vistos_menos_no_vistos",
            "roc_auc": d.mean(),
            "roc_auc_lo": d.min(),
            "roc_auc_hi": d.max(),
            "n_rows": N_SEEDS,
        }
    )

out_path = Path(REPORTS_DIR) / "customer_overlap.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(out_path, index=False)
print(f"\nGuardado en {out_path}")
