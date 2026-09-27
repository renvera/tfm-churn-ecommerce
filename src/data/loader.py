"""
src/data/loader.py
Carga del dataset Online Retail II (UCI) desde el Excel con dos hojas
(Year 2009-2010 y Year 2010-2011), unificándolas en un único DataFrame.
Cachea el resultado en Parquet para acelerar cargas posteriores.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config.settings import (
    RAW_DATASET_FILE,
    CACHE_FILE,
    INTERIM_DIR,
    SHEET_NAMES,
    EXPECTED_COLUMNS,
)

logger = logging.getLogger(__name__)


def _load_from_excel(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo en {file_path}. "
            "Verifica que hayas colocado el .xlsx en data/raw/."
        )

    frames = []
    for sheet in SHEET_NAMES:
        logger.info("Cargando hoja: %s", sheet)
        df_sheet = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")

        missing_cols = EXPECTED_COLUMNS - set(df_sheet.columns)
        if missing_cols:
            raise ValueError(
                f"La hoja '{sheet}' no tiene las columnas esperadas. "
                f"Faltan: {missing_cols}. Columnas encontradas: {list(df_sheet.columns)}"
            )

        # Forzar tipos consistentes: varias columnas de texto (Invoice,
        # StockCode, Description) mezclan números y strings dentro de la
        # misma columna (ej. 'C489449' en cancelaciones, descripciones que
        # son solo un número), lo que rompe la conversión a Parquet si se
        # dejan como object/mixed. Se castean a string, preservando los
        # nulos reales (NaN) para no perder esa información.
        for col in ("Invoice", "StockCode", "Description"):
            df_sheet[col] = df_sheet[col].apply(
                lambda x: x if pd.isna(x) else str(x)
            )

        df_sheet["source_sheet"] = sheet
        frames.append(df_sheet)

    return pd.concat(frames, ignore_index=True)


def load_online_retail(file_path: Path = RAW_DATASET_FILE, use_cache: bool = True) -> pd.DataFrame:
    """
    Carga y concatena las dos hojas del dataset Online Retail II.
    Usa una caché en Parquet (data/interim/) para evitar releer el Excel
    en cada ejecución.

    Parameters
    ----------
    file_path : Path
        Ruta al archivo .xlsx original.
    use_cache : bool
        Si True (por defecto), usa/crea la caché Parquet.
        Si False, fuerza la relectura desde Excel.

    Returns
    -------
    pd.DataFrame
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    if use_cache and CACHE_FILE.exists():
        logger.info("Cargando desde caché Parquet: %s", CACHE_FILE)
        return pd.read_parquet(CACHE_FILE)

    df = _load_from_excel(file_path)
    logger.info("Dataset unificado: %d filas, %d columnas", len(df), df.shape[1])

    if use_cache:
        df.to_parquet(CACHE_FILE, index=False)
        logger.info("Cache guardada en: %s", CACHE_FILE)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    data = load_online_retail()
    print(data.head())
    print(f"\nTotal filas: {len(data)}")
    print(f"Rango de fechas: {data['InvoiceDate'].min()} → {data['InvoiceDate'].max()}")
    print(f"\nValores nulos por columna:\n{data.isna().sum()}")