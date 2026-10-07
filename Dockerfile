# ============================================================
# Dockerfile — tfm-churn-ecommerce (imagen optimizada)
# Python 3.12-slim · uv · FastAPI · usuario sin privilegios
#
# Cambios respecto a versión anterior:
#   - jupyter, shap y xgboost movidos a extra [research]
#     (no se instalan en la imagen de producción)
#   - HEALTHCHECK explícito que exige model_loaded: true
#   - Capa de dependencias separada de la copia del código
#     (mejor cache en rebuilds)
# ============================================================

FROM python:3.12-slim AS base

# Instalar uv (gestor de dependencias)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ── Capa 1: dependencias (se cachea si pyproject.toml/uv.lock no cambian) ──
COPY pyproject.toml uv.lock ./

# --no-install-project: instala solo las dependencias, no el paquete src/
# --no-group dev: excluye httpx y pytest (solo producción)
# --no-extra research: excluye jupyter, shap, xgboost de la imagen
RUN uv sync --frozen --no-install-project --no-group dev

# ── Capa 2: modelo serializado ───────────────────────────────
COPY models/ ./models/

# ── Capa 3: código fuente ───────────────────────────────────
COPY src/ ./src/

# ── Usuario sin privilegios ─────────────────────────────────
RUN useradd --no-create-home --shell /bin/false apiuser
USER apiuser

# ── Variables de entorno ────────────────────────────────────
ENV CHURN_MODEL_PATH=/app/models/churn_model_1.0.0.joblib
ENV PYTHONPATH=/app
ENV UV_CACHE_DIR=/tmp/uv-cache

EXPOSE 8000

# ── Health check ────────────────────────────────────────────
# Comprueba que el modelo está cargado (model_loaded: true en /health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "\
import urllib.request, json, sys; \
r = urllib.request.urlopen('http://localhost:8000/health', timeout=8); \
d = json.loads(r.read()); \
sys.exit(0 if d.get('model_loaded') else 1)"

# ── Arranque ────────────────────────────────────────────────
CMD ["uv", "run", "--no-sync", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# ============================================================
# Nota: el extra [research] debe declararse en pyproject.toml:
#
# [project.optional-dependencies]
# research = [
#     "jupyter>=1.0",
#     "shap>=0.45",
#     "xgboost>=2.0",
# ]
#
# Para el entorno de desarrollo local (con todas las deps):
#   uv sync --extra research
#
# Para la imagen Docker (sin research):
#   uv sync --frozen --no-install-project --no-group dev
# ============================================================
