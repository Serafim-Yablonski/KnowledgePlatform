"""What only the shipped stack can show (P7d; tests/compose/conftest.py builds it):
the image indexes uploads end to end. R-a's failures are accepted on the host
(tests/acceptance); the restart policy is pinned by tests/unit/test_compose.py."""

from __future__ import annotations

import subprocess
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from tests.compose.conftest import Stack
from tests.pdfs import pdf_with_text
from tests.support.acceptance import Api, eventually, settle_deadline_s

pytestmark = pytest.mark.compose


@pytest.fixture
async def api(stack: Stack) -> AsyncIterator[Api]:
    async with httpx.AsyncClient(base_url=stack.api_url, timeout=30) as client:
        api = Api(client)
        await api.sign_in()
        yield api


async def _all_indexed(api: Api, docs: list[uuid.UUID], stack: Stack) -> None:
    """Waits until every document is settled (indexed or failed), then requires all
    indexed: a failure ends the wait at once and names its reason. The logs are
    read when the check fails, not when it starts."""

    async def settled() -> list[dict[str, Any]] | None:
        bodies = [await api.document(doc) for doc in docs]
        if any(b["status"] not in ("indexed", "failed") for b in bodies):
            return None
        return bodies

    bodies = await eventually(
        settled,
        timeout=settle_deadline_s(),
        what=lambda: f"{len(docs)} documents to settle\n{stack.logs()}",
        # Nothing here is timed: polling slower spares the stack's CPU, which the
        # runner shares, and the failure report the captured request logs.
        interval=0.5,
    )
    failed = {
        b["original_filename"]: b["failure_reason"]
        for b in bodies
        if b["status"] != "indexed"
    }
    assert not failed, f"documents failed: {failed}\n{stack.logs()}"


def _restarts(stack: Stack, service: str) -> int:
    return int(
        subprocess.run(
            ["docker", "inspect", "-f", "{{.RestartCount}}", stack.container(service)],
            env=stack.env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )


async def test_the_image_indexes_uploads_end_to_end(api: Api, stack: Stack) -> None:
    """Migrations, the API, the shared uploads volume (fresh, so writable by the
    image's user: F-P6-1), both workers and the embed worker's provider settings.
    The PDF is small: it proves the image's parse child runs, not how fast; timing
    under the short PIPELINE_ENV limits is tests/acceptance's, on the host."""
    pdf = pdf_with_text("Parsed by the shipped stack.")
    docs = [
        await api.upload("notes.txt", b"Indexed by the shipped stack."),
        await api.upload("notes.pdf", pdf, "application/pdf"),
    ]

    await _all_indexed(api, docs, stack)
    # Nothing restarted: the restart policy would hide a worker that crashed.
    restarts = {s: _restarts(stack, s) for s in ("worker-extract", "worker-embed")}
    assert restarts == {"worker-extract": 0, "worker-embed": 0}, stack.logs()
