# 006. Celery Workers Use a Separate Synchronous Database Engine
Status: Accepted | Date: 2026-05-14

## Context
The FastAPI application uses `asyncpg` via SQLAlchemy's `create_async_engine`. Celery workers are launched as separate OS processes and execute task functions synchronously. We need database access inside Celery tasks.

## Decision
Celery workers use a dedicated synchronous SQLAlchemy engine backed by `psycopg3` (`postgresql+psycopg://`). This engine is defined in `src/workers/database.py` and is entirely separate from the `asyncpg` engine in `src/core/database.py`.

`asyncpg` requires an active `asyncio` event loop. Celery worker processes have no event loop by default, so calling `asyncpg` APIs in a task raises `RuntimeError: no running event loop`. Wrapping tasks in `asyncio.run()` is possible but creates a new event loop per task call, which bypasses the connection pool entirely — each invocation opens a new TCP connection to PostgreSQL. Under load this exhausts the server's connection limit.

A separate sync engine avoids these problems. `psycopg3` is already a project dependency (required by LangGraph's `AsyncPostgresSaver`), so no new package is introduced.

## Consequences
Two engine objects exist in the process space when FastAPI and a Celery worker run together (e.g. in development with `task_always_eager`). Each has its own connection pool (5 + 10 async vs. 5 + 10 sync), which doubles PostgreSQL connections in that scenario. In production, workers are separate OS processes so pool isolation is the correct behavior. The `asyncpg` engine is explicitly rejected for worker use to prevent the event-loop and pool-exhaustion failure modes described above.
