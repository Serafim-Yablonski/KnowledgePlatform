"""A task that runs until its worker stops it, for the shutdown test (pipeline step 8b).

A real worker loads it with ``--include``. Its tasks report through files in the
directory named by KP_SHUTDOWN_TEST_DIR: ``started`` once one runs, ``finished`` when a
short one returns, ``on_failure`` if Celery calls that hook, which for our pipeline
tasks would write FAILED.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from celery import Task

from src.workers.celery_app import celery_app


def _report(name: str, text: str = "") -> None:
    Path(os.environ["KP_SHUTDOWN_TEST_DIR"], name).write_text(text)


class _ReportsFailure(Task):
    def on_failure(self, exc: Exception, *args: Any, **kwargs: Any) -> None:
        _report("on_failure", repr(exc))


@celery_app.task(base=_ReportsFailure, name="tests.support.slow_task.run_until_stopped")
def run_until_stopped() -> None:
    _report("started")
    time.sleep(600)


@celery_app.task(base=_ReportsFailure, name="tests.support.slow_task.finish_soon")
def finish_soon(seconds: float) -> None:
    _report("started")
    time.sleep(seconds)
    _report("finished")
