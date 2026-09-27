"""
scripts/check_structure.py
Verifica que la estructura de carpetas y archivos del proyecto esté completa.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "data/raw", "data/interim", "data/processed",
    "notebooks",
    "src/config", "src/data", "src/labeling", "src/features", "src/split",
    "src/models", "src/evaluation", "src/explainability", "src/api",
    "tests", "scripts", "models", "reports",
]

EXPECTED_INIT_FILES = [
    "src/__init__.py",
    "src/config/__init__.py", "src/data/__init__.py", "src/labeling/__init__.py",
    "src/features/__init__.py", "src/split/__init__.py", "src/models/__init__.py",
    "src/evaluation/__init__.py", "src/explainability/__init__.py", "src/api/__init__.py",
]

EXPECTED_GITKEEP = [
    "data/raw/.gitkeep", "data/interim/.gitkeep", "data/processed/.gitkeep", "models/.gitkeep",
]

EXPECTED_ROOT_FILES = [
    "pyproject.toml", ".gitignore", "README.md",
]


def check(paths, label):
    print(f"\n--- {label} ---")
    ok, missing = [], []
    for p in paths:
        full = ROOT / p
        if full.exists():
            ok.append(p)
        else:
            missing.append(p)
    for p in ok:
        print(f"  OK    {p}")
    for p in missing:
        print(f"  FALTA {p}")
    return missing


def main():
    print(f"Verificando estructura en: {ROOT}\n")

    all_missing = []
    all_missing += check(EXPECTED_DIRS, "Carpetas")
    all_missing += check(EXPECTED_INIT_FILES, "Archivos __init__.py")
    all_missing += check(EXPECTED_GITKEEP, "Archivos .gitkeep")
    all_missing += check(EXPECTED_ROOT_FILES, "Archivos raíz")

    # Revisar si hay carpetas/archivos inesperados dentro de src/
    print("\n--- Contenido real de src/ ---")
    src_path = ROOT / "src"
    if src_path.exists():
        for item in sorted(src_path.rglob("*")):
            print(f"  {item.relative_to(ROOT)}")

    print("\n" + "=" * 50)
    if all_missing:
        print(f"RESULTADO: Faltan {len(all_missing)} elementos.")
    else:
        print("RESULTADO: Estructura completa. Todo correcto.")
    print("=" * 50)


if __name__ == "__main__":
    main()