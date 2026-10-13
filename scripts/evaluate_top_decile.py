import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.loader import load_online_retail
from src.data.cleaning import get_clean_sales
from src.features.snapshots import build_multi_snapshot_dataset, generate_snapshot_dates
from src.splitting.temporal_split import temporal_split
from src.models.baseline import NUMERIC_FEATURES
from src.models.random_forest import build_random_forest_pipeline
from src.evaluation.calibration import calibrate_prefit

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

rf = build_random_forest_pipeline()
rf.fit(train[NUMERIC_FEATURES], train["churn"])
cal = calibrate_prefit(rf, method="sigmoid")
cal.fit(valid[NUMERIC_FEATURES], valid["churn"])


def top_decile(split, nombre):
    y = split["churn"].values
    p = cal.predict_proba(split[NUMERIC_FEATURES])[:, 1]
    base = y.mean()
    k = int(np.ceil(0.10 * len(y)))
    idx = np.argsort(-p, kind="stable")[:k]
    prec = y[idx].mean()
    print(
        f"[{nombre}] n={len(y)}  tasa base={base:.4f}  "
        f"n decil superior={k}  precision decil superior={prec:.4f}  "
        f"lift={prec / base:.4f}  lift max teorico={1 / base:.4f}"
    )


top_decile(valid, "valid")
top_decile(test, "test")
