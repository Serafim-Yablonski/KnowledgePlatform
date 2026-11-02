"""``python -m src.workers {extract,embed}``: one pipeline worker process (ADR-027).

Each phase is its own service: a stuck or memory-hungry extraction can't starve
embedding, the memory cap applies to extraction alone, and only the embed worker
holds the embedding provider's key.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.ai.embeddings import EmbeddingService
from src.core.config import settings
from src.core.logging import setup_logging
from src.core.observability import configure_logfire, instrument_libraries
from src.workers.runner import Work, embed_work, extract_work, serve


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


def configure_process(phase: str) -> None:
    """This worker process's logging and tracing; the entry point's alone, never an
    import's (the API imports the queue module)."""
    setup_logging(environment=settings.ENVIRONMENT, log_level=settings.LOG_LEVEL)
    # Nothing reaches a worker with a trace context of its own: the attempt
    # continues the one our API stored in the row (ADR-027).
    configure_logfire(
        token=settings.LOGFIRE_TOKEN or None,
        environment=settings.ENVIRONMENT,
        service_name=f"{settings.SERVICE_NAME}-worker-{phase}",
        trust_incoming_trace_context=False,
    )
    instrument_libraries()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.workers")
    parser.add_argument("phase", choices=sorted(PHASES))
    phase = parser.parse_args(argv).phase

    configure_process(phase)
    asyncio.run(serve(PHASES[phase], settings.ASYNC_DATABASE_URL))


if __name__ == "__main__":
    main()
