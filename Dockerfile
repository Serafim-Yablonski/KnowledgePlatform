# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Install all runtime dependencies into a virtual environment.
# Run `uv lock` locally before building to ensure uv.lock is up-to-date.
FROM python:3.14-slim AS builder

WORKDIR /app

# Same uv version as CI (.github/workflows/ci.yml).
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/uv

COPY pyproject.toml uv.lock ./
# Ship the dependencies' bytecode: the runtime user can't write __pycache__ into the
# root-owned venv, so without it every process — each PDF parse child included —
# compiles the dependencies it imports again (measured: ~1.5 s of a worker's start).
ENV UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# Minimal image: no build tools, no pip, no gcc.
FROM python:3.14-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/alembic.ini

ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN addgroup --system app && adduser --system --ingroup app app
# UPLOAD_DIR's default. A new named volume mounted here starts as a copy of this
# directory, ownership included; without it the volume is root's and uploads fail.
RUN mkdir -p /data/uploads && chown app:app /data/uploads
USER app

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
