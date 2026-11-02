"""What only the shipped stack can show (P7d; tests/compose/conftest.py builds it):
the image indexes uploads end to end. R-a's failures are accepted on the host
(tests/acceptance); the restart policy is pinned by tests/unit/test_compose.py."""

from __future__ import annotations

import subprocess
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest

from tests.acceptance.test_pipeline import slow_pdf
from tests.compose.conftest import Stack
from tests.support.acceptance import Api, eventually, settle_deadline_s

pytestmark = pytest.mark.compose


@pytest.fixture
async def api(stack: Stack) -> AsyncIterator[Api]:
    async with httpx.AsyncClient(base_url=stack.api_url, timeout=30) as client:
        api = Api(client)
        await api.sign_in()
        yield api


async def _status_is(
    api: Api, docs: list[uuid.UUID], status: str, stack: Stack
) -> None:
    async def reached() -> bool | None:
        for doc in docs:
            if (await api.document(doc))["status"] != status:
                return None
        return True

    await eventually(
        reached,
        timeout=settle_deadline_s(),
        what=f"{len(docs)} documents {status}\n{stack.logs()}",
    )


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
    image's user: F-P6-1), both workers and the embed worker's provider settings."""
    docs = [
        await api.upload("notes.txt", b"Indexed by the shipped stack."),
        await api.upload("slow.pdf", slow_pdf(), "application/pdf"),
    ]

    await _status_is(api, docs, "indexed", stack)
    # Nothing restarted: the restart policy would hide a worker that crashed.
    restarts = {s: _restarts(stack, s) for s in ("worker-extract", "worker-embed")}
    assert restarts == {"worker-extract": 0, "worker-embed": 0}, stack.logs()
