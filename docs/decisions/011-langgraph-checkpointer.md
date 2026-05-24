# 011. LangGraph Checkpointer: AsyncPostgresSaver over SQLite, psycopg3 over asyncpg
Status: Accepted | Date: 2026-05-21

## Context
The research workflow uses LangGraph for multi-step graph orchestration with human-in-the-loop interrupts. Graph state must survive process restarts (crash recovery) and be readable from any worker process (multi-worker deployments). The checkpointer is the persistence layer for this state.

## Decision
Use `AsyncPostgresSaver` (from `langgraph-checkpoint-postgres`) backed by the same PostgreSQL 18 instance as the rest of the application, with its own dedicated psycopg3 (`psycopg[binary]`) async connection pool. The checkpointer is initialised once in the FastAPI lifespan and reused across all requests.

For synthesis streaming, tokens are published to Redis: each chunk is appended to a Redis list (for late-subscriber replay) and published on a pub/sub channel. The SSE endpoint replays the list then subscribes for live delivery. This works across multiple Uvicorn workers, unlike an in-process asyncio.Queue.

## Consequences
`AsyncPostgresSaver` requires psycopg3, not asyncpg. LangGraph's checkpoint protocol uses psycopg3's cursor and connection API directly — asyncpg's binary protocol is incompatible. This means the application runs two PostgreSQL driver stacks: asyncpg for all application queries (SQLAlchemy async sessions) and psycopg3 exclusively for the LangGraph checkpointer. Both coexist without conflict; psycopg2 is never used (see ADR-005).

SQLite was rejected: it is single-process, not safe for concurrent writes, and cannot be shared across Uvicorn workers or pods. An in-memory `MemorySaver` was rejected for the same reason and is used only in tests. Redis-only persistence was considered but rejected because Redis is volatile — a restart would lose all in-flight research state and make crash recovery impossible without the PostgreSQL checkpoint.
