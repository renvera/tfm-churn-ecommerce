"""
src/config/settings.py
Configuración centralizada del proyecto: rutas y constantes globales.
"""

from pathlib import Path

# Raíz del proyecto (2 niveles arriba de este archivo: src/config/settings.py)
ROOT_DIR = Path(__file__).resolve().parents[2]

# Directorios de datos
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Archivos
RAW_DATASET_FILE = RAW_DIR / "online_retail_II.xlsx"
CACHE_FILE = INTERIM_DIR / "online_retail_ii_raw.parquet"

# Otros directorios
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Constantes del dataset
SHEET_NAMES = ["Year 2009-2010", "Year 2010-2011"]

EXPECTED_COLUMNS = {
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country",
}

# Horizontes de churn a evaluar (en días)
CHURN_HORIZONS = [60, 90, 120]