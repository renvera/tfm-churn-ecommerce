"""
scripts/analyze_shap.py

Interpretabilidad del modelo final (Random Forest calibrado con Platt
ajustado en valid) mediante SHAP.

- Reconstruye EXACTAMENTE el mismo dataset y split temporal que
  scripts/evaluate_thresholds.py (mismas funciones y parámetros).
- Entrena el Random Forest SOLO con train.
- Calibra con Platt sobre valid (modelo base congelado) y reobtiene el
  umbral operativo (objetivo recall >= 0.80 elegido en valid).
- Explica el modelo BASE (imputación -> Random Forest) con TreeExplainer:
  el calibrador Platt solo transforma la probabilidad, no altera el
  ranking ni las contribuciones por variable.
- Explica clientes del split test; la selección de los casos locales se
  hace por probabilidad del modelo, nunca por la etiqueta real.

Uso:
    uv run python scripts/analyze_shap.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap
import sklearn

from src.data.loader import load_online_retail
from src.data.cleaning import get_clean_sales
from src.features.snapshots import build_multi_snapshot_dataset, generate_snapshot_dates
from src.splitting.temporal_split import temporal_split
from src.models.baseline import NUMERIC_FEATURES
from src.models.random_forest import build_random_forest_pipeline
from src.evaluation.calibration import calibrate_prefit
from src.evaluation.threshold import threshold_for_min_recall

TARGET_RECALL = 0.80
N_SHAP_SAMPLE = 2000
RANDOM_STATE = 42

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# 1) Datos y split temporal (idéntico a evaluate_thresholds.py)
# --------------------------------------------------------------------------
print("=" * 70)
print("ANÁLISIS SHAP - RANDOM FOREST CALIBRADO (modelo final)")
print("=" * 70)
print(f"  shap={shap.__version__}  scikit-learn={sklearn.__version__}")

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
for name, split in (("train", train), ("valid", valid), ("test", test)):
    print(
        f"  {name:<6} n={len(split):>6}  "
        f"prevalencia churn={split['churn'].mean():.4f}"
    )


# --------------------------------------------------------------------------
# 2) Modelo final: RF en train + Platt en valid + umbral operativo
# --------------------------------------------------------------------------
pipe = build_random_forest_pipeline()
pipe.fit(train[NUMERIC_FEATURES], train["churn"])

cal = calibrate_prefit(pipe, method="sigmoid")
cal.fit(valid[NUMERIC_FEATURES], valid["churn"])

proba_valid = cal.predict_proba(valid[NUMERIC_FEATURES])[:, 1]
thr = threshold_for_min_recall(valid["churn"].values, proba_valid, TARGET_RECALL)
print(f"\nUmbral operativo reobtenido en valid (recall>={TARGET_RECALL}): {thr:.3f}")
print("  (debe coincidir con el 0.599 de evaluate_thresholds.py)")


# --------------------------------------------------------------------------
# 3) Valores SHAP del modelo base sobre una muestra de test
# --------------------------------------------------------------------------
pre = pipe.named_steps["preprocessor"]
model = pipe.named_steps["model"]

test_sample = test.sample(n=min(N_SHAP_SAMPLE, len(test)), random_state=RANDOM_STATE)
X_raw = test_sample[NUMERIC_FEATURES]
X_trans = pre.transform(X_raw)  # ndarray; columnas = NUMERIC_FEATURES en orden

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_trans)

# Compatibilidad entre versiones de shap: lista por clase, ndarray 3D o 2D
if isinstance(sv, list):
    sv_pos, ev = sv[1], explainer.expected_value
    ev_pos = ev[1] if np.ndim(ev) > 0 else ev
elif isinstance(sv, np.ndarray) and sv.ndim == 3:
    sv_pos, ev = sv[:, :, 1], explainer.expected_value
    ev_pos = ev[1] if np.ndim(ev) > 0 else ev
else:
    sv_pos, ev_pos = sv, explainer.expected_value

# Verificación aditiva: base + suma de SHAP == probabilidad del RF base
proba_raw_sample = pipe.predict_proba(X_raw)[:, 1]
err = np.abs(ev_pos + sv_pos.sum(axis=1) - proba_raw_sample).max()
print(f"\nVerificación aditiva SHAP: max|base + ΣSHAP − proba_RF| = {err:.6f}")


# --------------------------------------------------------------------------
# 4) Importancia global (bar + beeswarm) y tabla en consola
# --------------------------------------------------------------------------
shap.summary_plot(
    sv_pos, X_trans, feature_names=NUMERIC_FEATURES,
    plot_type="bar", show=False,
)
plt.savefig(REPORTS / "shap_global_importance.png", bbox_inches="tight", dpi=150)
plt.close("all")

shap.summary_plot(sv_pos, X_trans, feature_names=NUMERIC_FEATURES, show=False)
plt.savefig(REPORTS / "shap_beeswarm.png", bbox_inches="tight", dpi=150)
plt.close("all")

mean_abs = np.abs(sv_pos).mean(axis=0)
mean_signed = sv_pos.mean(axis=0)
orden = np.argsort(mean_abs)[::-1]
print("\nIMPORTANCIA GLOBAL (media |SHAP|, muestra de test):")
for idx in orden:
    f = NUMERIC_FEATURES[idx]
    direccion = "hacia churn" if mean_signed[idx] > 0 else "hacia no churn"
    print(
        f"  {f:<28} media|SHAP|={mean_abs[idx]:.4f}  "
        f"SHAP medio={mean_signed[idx]:+.4f} ({direccion})"
    )


# --------------------------------------------------------------------------
# 5) Explicaciones locales (waterfall): casos alto y bajo riesgo de test
# --------------------------------------------------------------------------
proba_cal_sample = cal.predict_proba(X_raw)[:, 1]
i_high = int(np.argmax(proba_cal_sample))
i_low = int(np.argmin(proba_cal_sample))


def explicar_local(i: int, salida: str) -> None:
    expl = shap.Explanation(
        values=sv_pos[i],
        base_values=ev_pos,
        data=X_trans[i],
        feature_names=NUMERIC_FEATURES,
    )
    shap.plots.waterfall(expl, max_display=9, show=False)
    plt.savefig(REPORTS / salida, bbox_inches="tight", dpi=150)
    plt.close("all")


def resumen_caso(i: int, titulo: str) -> None:
    row = test_sample.iloc[i]
    print(f"\nCASO {titulo} (fila {i} de la muestra de test):")
    cid = row["Customer ID"] if "Customer ID" in test_sample.columns else "N/A"
    print(f"  Customer ID={cid}  cutoff_date={row['cutoff_date']}")
    for f in NUMERIC_FEATURES:
        print(f"    {f:<28} = {row[f]:.4f}")
    p_raw, p_cal = proba_raw_sample[i], proba_cal_sample[i]
    decision = "PRIORITARIO (churn)" if p_cal >= thr else "no prioritario"
    print(f"  probabilidad RF (sin calibrar)  = {p_raw:.4f}")
    print(f"  probabilidad calibrada (Platt)  = {p_cal:.4f}")
    print(f"  umbral operativo                = {thr:.3f}")
    print(f"  decisión                        = {decision}")
    print(f"  etiqueta real (solo contexto)   = {int(row['churn'])}")


explicar_local(i_high, "shap_local_high_risk.png")
explicar_local(i_low, "shap_local_low_risk.png")
resumen_caso(i_high, "DE RIESGO ALTO (mayor prob. calibrada)")
resumen_caso(i_low, "DE RIESGO BAJO (menor prob. calibrada)")

print(
    "\nNota metodológica: los casos se seleccionaron por probabilidad del "
    "modelo, no por la etiqueta real; la etiqueta se muestra solo como "
    "contexto posterior. SHAP explica el RF base; Platt solo remapea la "
    "probabilidad y no cambia el ranking."
)
print(f"\nGráficos guardados en: {REPORTS}")
print("  shap_global_importance.png, shap_beeswarm.png,")
print("  shap_local_high_risk.png, shap_local_low_risk.png")