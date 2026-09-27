"""
scripts/verify_setup.py
Verifica que el entorno base (Paso 0) esté correctamente configurado.
"""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PACKAGES = [
    "pandas", "numpy", "sklearn", "xgboost", "matplotlib",
    "shap", "fastapi", "pydantic", "uvicorn", "pytest", "openpyxl", "pyarrow",
]


def check_python_version():
    print(f"Python: {sys.version}")
    major, minor = sys.version_info[:2]
    if not (major == 3 and 11 <= minor <= 13):
        print("  AVISO: se recomienda Python 3.11-3.13")
    else:
        print("  OK")


def check_packages():
    print("\nPaquetes:")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"  OK    {pkg}")
        except ImportError:
            print(f"  FALTA {pkg}")
            missing.append(pkg)
    return missing


def check_dataset():
    print("\nDataset:")
    path = ROOT / "data" / "raw" / "online_retail_II.xlsx"
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  OK    {path} ({size_mb:.1f} MB)")
    else:
        print(f"  FALTA {path}")


def main():
    print("=" * 50)
    print("VERIFICACIÓN DEL ENTORNO")
    print("=" * 50)

    check_python_version()
    missing = check_packages()
    check_dataset()

    print("\n" + "=" * 50)
    if missing:
        print(f"RESULTADO: Faltan {len(missing)} paquetes. Ejecuta 'uv sync'.")
    else:
        print("RESULTADO: Todo correcto. El entorno está bien configurado.")
    print("=" * 50)


if __name__ == "__main__":
    main()