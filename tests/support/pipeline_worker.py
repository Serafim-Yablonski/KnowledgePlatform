"""A pipeline worker process with a stand-in job, for tests/integration/
test_worker_process.py: the real loop, signals and database writes (runner.serve),
with a job whose timing the test controls.

    python -m tests.support.pipeline_worker {block,crash,finish}

Each run of the job leaves a file in KP_WORKER_TEST_DIR, so a test counts runs across
processes. block: runs until cancelled. crash: SIGKILLs its own process (the OOM
killer, a lost host). finish: returns at once.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import signal
import sys
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.config import settings
from src.workers.runner import Work, extract_work, serve

RUNS = Path(os.environ["KP_WORKER_TEST_DIR"])


def _ran(document_id: uuid.UUID) -> None:
    (RUNS / f"run-{document_id}-{uuid.uuid4().hex}").touch()


async def block(document_id: uuid.UUID) -> None:
    _ran(document_id)
    await asyncio.sleep(3600)


async def crash(document_id: uuid.UUID) -> None:
    _ran(document_id)
    os.kill(os.getpid(), signal.SIGKILL)


async def finish(document_id: uuid.UUID) -> None:
    _ran(document_id)


JOBS = {"block": block, "crash": crash, "finish": finish}


async def _work(sessions: async_sessionmaker[AsyncSession], _: AsyncExitStack) -> Work:
    return dataclasses.replace(extract_work(sessions, settings), job=JOBS[sys.argv[1]])


if __name__ == "__main__":
    asyncio.run(serve(_work, settings.ASYNC_DATABASE_URL.get_secret_value()))
