"""
src/evaluation/calibration.py

Diagnóstico de calibración: ¿una probabilidad de 0.80 corresponde a
~80% de churn real? Tabla por bins de probabilidad, Brier score y ECE.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo guarda a fichero
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """
    Agrupa las predicciones en bins de igual tamaño (cuantiles) y compara
    la probabilidad media predicha con la tasa observada de churn.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    table = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), mean_pred=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index(drop=True)
    )
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """ECE: diferencia absoluta media entre predicho y observado, ponderada por bin."""
    total = table["n"].sum()
    gap = (table["mean_pred"] - table["observed_rate"]).abs()
    return float((table["n"] / total * gap).sum())


def brier(y_true, proba) -> float:
    """Brier score: error cuadrático medio de las probabilidades (menor es mejor)."""
    return float(brier_score_loss(y_true, proba))


def plot_calibration_curves(
    tables: dict[str, pd.DataFrame], title: str, output_path: str | Path
) -> None:
    """Guarda las curvas de calibración de varios modelos en un PNG."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Calibración perfecta")
    for name, table in tables.items():
        ax.plot(table["mean_pred"], table["observed_rate"], marker="o", label=name)
    ax.set_xlabel("Probabilidad media predicha")
    ax.set_ylabel("Tasa observada de churn")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)