FROM python:3.13-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_CACHE_DIR=/tmp/uv-cache

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable --no-install-project

COPY tsmixer_m5 ./tsmixer_m5
RUN uv sync --frozen --no-dev --no-editable

RUN find /app/.venv -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /app/.venv/lib/python*/site-packages/torch/test \
           /app/.venv/lib/python*/site-packages/torch/include \
           /app/.venv/lib/python*/site-packages/torch/share \
           /app/.venv/lib/python*/site-packages/numpy/core/include 2>/dev/null || true

FROM python:3.13-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    DEVICE=cpu \
    USE_AMP=true \
    MAX_BATCH_SIZE=1000 \
    MODEL_ARTIFACT_LOCAL_DIR=/app/artifact \
    DATA_SNAPSHOT_DIR=/app/data/m5_sample

WORKDIR /app

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

COPY --from=builder /app/.venv /app/.venv
COPY healthcheck.py /app/healthcheck.py

COPY tsmixer_m5 /app/tsmixer_m5
COPY best_wrmsse_seed_42.pth /app/artifact/best_wrmsse_seed_42.pth
COPY data/m5_sample /app/data/m5_sample

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["tsmixer_m5.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]