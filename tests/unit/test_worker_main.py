"""``python -m src.workers``: which phase a process runs, and what it builds."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

import pytest

from src.workers import __main__ as cli
from src.workers.queue import EMBEDDING, EXTRACTION


def test_an_unknown_phase_is_refused() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["index"])
    assert exc_info.value.code == 2


async def test_the_extract_worker_builds_no_embedding_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The embedding key stays out of the process that parses untrusted files."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the extract worker built an embedding client")

    monkeypatch.setattr(cli.EmbeddingService, "from_settings", refuse)
    async with AsyncExitStack() as stack:
        work = await cli.PHASES["extract"](None, stack)  # type: ignore[arg-type]

    assert work.phase is EXTRACTION


async def test_the_embed_worker_shares_one_client_across_attempts() -> None:
    async with AsyncExitStack() as stack:
        work = await cli.PHASES["embed"](None, stack)  # type: ignore[arg-type]
        embedder = work.job.args[1]  # type: ignore[attr-defined]
        client = embedder._http_client
        assert client is not None and not client.is_closed

    assert work.phase is EMBEDDING
    # Closed with the process's stack, after the worker has wound down.
    assert client.is_closed
