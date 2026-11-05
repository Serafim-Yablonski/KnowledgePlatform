"""``python -m src.workers {extract,embed}``: one pipeline worker process (ADR-027).

Each phase is its own service: a stuck or memory-hungry extraction can't starve
embedding, the memory cap applies to extraction alone, and only the embed worker
holds the embedding provider's key.

``python -m src.workers reindex``: owe embedding again to every document with
extracted text, for the embed workers to pick up (ADR-030), then exit.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack

import httpx
import logfire
import structlog
from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.ai.embeddings import EmbeddingService
from src.core.config import settings
from src.core.logging import setup_logging
from src.core.observability import configure_logfire, instrument_libraries
from src.workers.queue import reindex
from src.workers.runner import Work, embed_work, extract_work, serve

logger = structlog.get_logger(__name__)


async def _extract(
    sessions: async_sessionmaker[AsyncSession], _: AsyncExitStack
) -> Work:
    return extract_work(sessions, settings)


async def _embed(
    sessions: async_sessionmaker[AsyncSession], stack: AsyncExitStack
) -> Work:
    # One client for the process: its connection pool outlives every attempt.
    http = await stack.enter_async_context(httpx.AsyncClient(timeout=30.0))
    embedder = EmbeddingService.from_settings(settings, http)
    return embed_work(sessions, embedder, settings)


PHASES = {"extract": _extract, "embed": _embed}


async def reindex_all(database_url: str) -> int:
    """One transaction under one span: every re-embedding continues its trace."""
    engine = create_async_engine(database_url, poolclass=NullPool)
    try:
        with logfire.span("reindex"):
            async with async_sessionmaker(engine).begin() as session:
                owed = await reindex(session, dict(logfire.get_context()) or None)
        logger.info("documents owed embedding again", count=owed)
        return owed
    finally:
        await engine.dispose()


def configure_process(phase: str) -> None:
    """This worker process's logging and tracing; the entry point's alone, never an
    import's (the API imports the queue module)."""
    setup_logging(environment=settings.ENVIRONMENT, log_level=settings.LOG_LEVEL)
    # Nothing reaches a worker with a trace context of its own: the attempt
    # continues the one our API stored in the row (ADR-027).
    configure_logfire(
        token=settings.LOGFIRE_TOKEN.get_secret_value() or None,
        environment=settings.ENVIRONMENT,
        service_name=f"{settings.SERVICE_NAME}-worker-{phase}",
        trust_incoming_trace_context=False,
    )
    instrument_libraries()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.workers")
    parser.add_argument("command", choices=[*sorted(PHASES), "reindex"])
    command = parser.parse_args(argv).command

    configure_process(command)
    if command == "reindex":
        asyncio.run(reindex_all(settings.ASYNC_DATABASE_URL.get_secret_value()))
    else:
        asyncio.run(
            serve(PHASES[command], settings.ASYNC_DATABASE_URL.get_secret_value())
        )


if __name__ == "__main__":
    main()
