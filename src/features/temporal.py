"""
src/features/temporal.py

Features estacionales derivadas de la fecha de corte (cutoff_date), para
ayudar al modelo a capturar la deriva temporal detectada en el %churn
(pico en dic-2010/ene-2011, ligado a la campaña navideña; ver análisis
exploratorio por snapshot).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Meses donde el análisis exploratorio mostró el pico de churn
# (snapshots dic-2010 a abr-2011: %churn entre 62% y 67%), consistente
# con clientes que compraron solo por Navidad y no repiten en 90 días.
POST_HOLIDAY_MONTHS = {12, 1, 2, 3}


def add_seasonal_features(
    dataset: pd.DataFrame,
    date_col: str = "cutoff_date",
) -> pd.DataFrame:
    """
    Añade columnas estacionales basadas en el mes de cutoff_date:

    - month: mes calendario (1-12).
    - month_sin / month_cos: codificación cíclica del mes.
    - is_post_holiday: 1 si el snapshot cae en dic/ene/feb/mar
      (ventana donde se observó el pico de churn), 0 en caso contrario.

    No modifica el DataFrame original.
    """
    df = dataset.copy()
    month = df[date_col].dt.month

    df["month"] = month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_post_holiday"] = month.isin(POST_HOLIDAY_MONTHS).astype(int)

    return df