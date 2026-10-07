#!/usr/bin/env python3
"""
cleanup_repo.py — Limpieza del repositorio tfm-churn-ecommerce.

Pendientes que resuelve:
  - Borrar archivos *.bak
  - Borrar paquete residual src/tfm_churn_ecommerce/
  - Confirmar la versión de Python del entorno

Uso:
    uv run python scripts/cleanup_repo.py          # modo dry-run (muestra qué borraría)
    uv run python scripts/cleanup_repo.py --apply  # borra de verdad

IMPORTANTE: ejecutar desde la raíz del repositorio.
"""

import sys
import shutil
import platform
from pathlib import Path

DRY_RUN = "--apply" not in sys.argv
ROOT = Path(__file__).resolve().parent.parent  # raíz del repo


def log(msg: str) -> None:
    prefix = "[DRY-RUN] " if DRY_RUN else "[APPLY]   "
    print(prefix + msg)


def check_python_version() -> None:
    """Confirma la versión real de Python del entorno activo."""
    v = platform.python_version()
    impl = platform.python_implementation()
    print(f"\n{'='*60}")
    print(f"Python activo: {impl} {v}")
    print(f"Ejecutable:    {sys.executable}")
    print(f"{'='*60}\n")

    major, minor, _ = v.split(".")
    if int(major) != 3 or int(minor) != 12:
        print(f"  ⚠  ATENCIÓN: se esperaba Python 3.12 (del .python-version).")
        print(f"     Versión activa: {v}")
    else:
        print(f"  ✓  Python 3.12 confirmado.\n")


def remove_bak_files() -> None:
    """Elimina todos los archivos *.bak del repositorio."""
    bak_files = list(ROOT.rglob("*.bak"))
    if not bak_files:
        print("  ✓  No se encontraron archivos *.bak.\n")
        return

    print(f"  Archivos *.bak encontrados ({len(bak_files)}):")
    for f in bak_files:
        rel = f.relative_to(ROOT)
        log(f"  rm {rel}")
        if not DRY_RUN:
            f.unlink()
    print()


def remove_residual_package() -> None:
    """Elimina src/tfm_churn_ecommerce/ si existe."""
    target = ROOT / "src" / "tfm_churn_ecommerce"
    if not target.exists():
        print("  ✓  src/tfm_churn_ecommerce/ no existe (ya limpio).\n")
        return

    log(f"  rmtree src/tfm_churn_ecommerce/")
    if not DRY_RUN:
        shutil.rmtree(target)
    print()


def main() -> None:
    if DRY_RUN:
        print("\nMODO DRY-RUN: no se borra nada. Añade --apply para ejecutar.")
    else:
        print("\nMODO APPLY: se eliminan los archivos.")

    check_python_version()

    print("── Archivos *.bak ──────────────────────────────────────")
    remove_bak_files()

    print("── Paquete residual ────────────────────────────────────")
    remove_residual_package()

    if DRY_RUN:
        print("Dry-run completado. Ejecuta con --apply para borrar.\n")
    else:
        print("Limpieza completada.\n")


if __name__ == "__main__":
    main()
