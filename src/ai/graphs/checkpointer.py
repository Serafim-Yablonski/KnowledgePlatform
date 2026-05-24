from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.core.config import get_settings

_checkpointer: AsyncPostgresSaver | None = None


def get_psycopg_connstr() -> str:
    url = get_settings().SYNC_DATABASE_URL
    # Strip SQLAlchemy driver prefix — psycopg3 needs a plain postgres:// URI.
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix) :]
    return url


@asynccontextmanager
async def setup_checkpointer() -> AsyncGenerator[AsyncPostgresSaver]:
    global _checkpointer
    connstr = get_psycopg_connstr()
    async with AsyncPostgresSaver.from_conn_string(connstr) as saver:
        await saver.setup()
        _checkpointer = saver
        try:
            yield saver
        finally:
            _checkpointer = None


def get_checkpointer() -> AsyncPostgresSaver:
    if _checkpointer is None:
        raise RuntimeError(
            "Checkpointer not initialised — call setup_checkpointer() in lifespan"
        )
    return _checkpointer
