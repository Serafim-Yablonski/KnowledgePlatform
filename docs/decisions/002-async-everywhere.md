# 002. Async Driver Strategy: asyncpg + psycopg3, Never psycopg2

Status: Accepted | Date: 2026-05-04

## Context

The platform uses PostgreSQL 18 for all persistence. Three subsystems need database access: the FastAPI application layer (SQLAlchemy ORM queries in route handlers and repositories), the Alembic migration runner, and LangGraph's `AsyncPostgresSaver` checkpointer. Each has different requirements around async/sync operation and driver compatibility, and Python will raise an import-time conflict if both asyncpg and psycopg2 are loaded in the same process.

## Decision

**asyncpg** is the exclusive driver for all application code. SQLAlchemy's asyncio extension runs over asyncpg (`postgresql+asyncpg://`), giving fully non-blocking I/O for every ORM query. The module-level `engine` in `src/core/database.py` owns the connection pool; `get_db()` in `src/core/dependencies.py` is the single entry point for route handlers.

**psycopg3** (`psycopg[binary]`) serves two purposes: (1) Alembic migrations — `alembic/env.py` calls `asyncio.run()` with an asyncpg connection, but test harnesses pass the asyncpg URL directly via `cfg.set_main_option` so no driver conversion is needed; (2) LangGraph's `AsyncPostgresSaver` checkpointer, which ships its own psycopg3-based async pool and cannot be swapped for asyncpg without forking LangGraph.

**psycopg2 is explicitly excluded.** asyncpg and psycopg2 conflict at the C-extension level when both are loaded in the same process. Any transitive dependency that pulls in psycopg2 must be excluded in `pyproject.toml`. psycopg3 (the `psycopg` package) coexists cleanly with asyncpg because it is a wholly separate package.

## Consequences

Two drivers are installed (`asyncpg`, `psycopg[binary]`), each owning a distinct scope: asyncpg for the hot path, psycopg3 for LangGraph checkpointing. Alembic migrations also run over asyncpg via `asyncio.run()` in `env.py`. Any new dependency that pulls in psycopg2 must be audited and excluded. Rejected alternatives: using psycopg3 for the full application (loses the mature asyncpg driver ecosystem and requires different SQLAlchemy dialect strings throughout); using only asyncpg everywhere including LangGraph (impossible without forking LangGraph's checkpointer).
