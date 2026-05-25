# 015. Docker Multi-Stage Build: Builder + Runtime Separation
Status: Accepted | Date: 2026-05-25

## Context
The production Docker image needs uv for dependency installation, but uv, pip, and gcc should not be present in the runtime image — they increase attack surface and image size. Python 3.14 on `python:3.14-slim` has limited prebuilt wheel availability for some C extensions, so the build environment and runtime environment must be kept separate and predictable.

## Decision
Two-stage `Dockerfile`: a `builder` stage installs uv and runs `uv sync --frozen --no-dev` to populate `/app/.venv`; a `runtime` stage copies only the `.venv` from the builder, plus `src/`, `alembic/`, and `alembic.ini`. The runtime stage has no pip, no uv, no gcc, and no build tools. `PATH` is set to `/app/.venv/bin:$PATH` so the venv's Python and all entry points resolve without activation. `CMD` is `uvicorn src.main:app --host 0.0.0.0 --port 8000`.

## Consequences
The runtime image contains only the application code and its resolved dependencies — roughly 300 MB for the full stack including asyncpg, psycopg3, and the ML dependencies. No build tools means no `pip install` surface in production. `uv sync --frozen` ensures the lockfile is the source of truth; if `pyproject.toml` and `uv.lock` diverge the build fails explicitly.

The main operational constraint: `uv.lock` must be committed and kept up to date. Running `uv lock` locally before `docker build` is a prerequisite. Rejected alternative: single-stage image with pip. This embeds the entire build toolchain in the runtime image, doubles the layer count for dependency changes, and makes it harder to reason about what is actually running in production.
