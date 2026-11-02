"""A stopped worker hands its running task back to the queue at once (step 8b).

`docker compose stop` sends SIGTERM, waits `stop_grace_period`, then SIGKILLs. Celery's
SIGTERM is a warm shutdown that waits for running tasks (minutes, ours), so the kill
came first and the message waited an hour in `unacked` for the visibility timeout.
With REMAP_SIGTERM=SIGQUIT the worker waits `worker_soft_shutdown_timeout`, cancels
what still runs and restores its messages before exiting.

A real worker on a real Redis, with the worker environment and grace period read from
docker-compose.yml. Its soft timeout is shortened to keep the tests fast;
tests/unit/test_compose.py pins that the real one fits inside the grace period.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from subprocess import Popen

import pytest
from kombu import Message

from tests.support import redis as compose_redis_support
from tests.support.compose import service

ROOT = Path(__file__).resolve().parents[2]
QUEUE = "shutdown-test"
DOCKER_DEFAULT_GRACE_S = 10


def grace_period_s(name: str) -> float:
    """A compose duration such as "30s" or "1m"; Docker's default when unset."""
    raw = service(name).get("stop_grace_period")
    if raw is None:
        return DOCKER_DEFAULT_GRACE_S
    match = re.fullmatch(r"(\d+)(s|m)", str(raw))
    assert match, f"unsupported duration {raw!r}"
    return int(match[1]) * (60 if match[2] == "m" else 1)


def _wait_for(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.1)


SOFT_SHUTDOWN_S = 3

WORKERS = ["celery-worker-extract", "celery-worker-embed"]


@contextmanager
def _worker(compose_service: str, tmp_path: Path) -> Iterator[tuple[str, Popen[bytes]]]:
    """A real worker with `compose_service`'s REMAP_SIGTERM, consuming QUEUE on a Redis
    started as compose starts it; yields the broker URL and the process."""
    from tests.conftest import _ensure_docker_host

    _ensure_docker_host()
    environment = service(compose_service).get("environment", {})
    with compose_redis_support.compose_redis() as redis_container:
        broker = compose_redis_support.url(redis_container)
        env = {
            **os.environ,
            "LOGFIRE_TOKEN": "",  # never export from a test
            **{k: str(v) for k, v in environment.items() if k == "REMAP_SIGTERM"},
            "CELERY_BROKER_URL": broker,
            "CELERY_WORKER_SOFT_SHUTDOWN_TIMEOUT": str(SOFT_SHUTDOWN_S),
            "KP_SHUTDOWN_TEST_DIR": str(tmp_path),
        }
        with (tmp_path / "worker.log").open("w") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "celery", "-A", "src.workers.celery_app",
                 "worker", "-Q", QUEUE, "-c", "1", "--include",
                 "tests.support.slow_task", "--without-mingle", "--without-gossip",
                 "-l", "warning"],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            )  # fmt: skip
            try:
                yield broker, proc
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()


def _stop_like_compose(proc: Popen[bytes], compose_service: str) -> None:
    """SIGTERM, then fail where compose would SIGKILL."""
    proc.send_signal(signal.SIGTERM)
    grace = grace_period_s(compose_service)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"worker still running after the {grace} s grace period: "
            "compose would SIGKILL it with the task unacknowledged"
        )


def _queued(broker: str) -> Message | None:
    from src.workers.celery_app import celery_app

    with celery_app.connection_for_read(broker) as conn:
        message: Message | None = conn.default_channel.basic_get(QUEUE, no_ack=True)
    return message


@pytest.mark.parametrize("compose_service", WORKERS)
def test_a_stopped_worker_puts_its_running_task_back(
    compose_service: str, tmp_path: Path
) -> None:
    from src.workers.celery_app import celery_app
    from tests.support.slow_task import run_until_stopped

    with _worker(compose_service, tmp_path) as (broker, proc):
        with celery_app.connection_for_write(broker) as conn:
            run_until_stopped.apply_async(queue=QUEUE, connection=conn)
        _wait_for(tmp_path / "started", timeout=60)

        _stop_like_compose(proc, compose_service)

        message = _queued(broker)
        assert message is not None, "the running task wasn't put back"
        assert message.headers["task"] == run_until_stopped.name
        # A cancel is no failure: our tasks' on_failure would write FAILED.
        assert not (tmp_path / "on_failure").exists()


def test_a_task_that_ends_within_the_soft_timeout_finishes(tmp_path: Path) -> None:
    """The soft timeout is time to finish, not only a delay before the cancel: a short
    task completes and is acknowledged, so it isn't run twice."""
    from src.workers.celery_app import celery_app
    from tests.support.slow_task import finish_soon

    compose_service = WORKERS[0]
    with _worker(compose_service, tmp_path) as (broker, proc):
        with celery_app.connection_for_write(broker) as conn:
            finish_soon.apply_async(
                args=[SOFT_SHUTDOWN_S / 2], queue=QUEUE, connection=conn
            )
        _wait_for(tmp_path / "started", timeout=60)

        _stop_like_compose(proc, compose_service)

        assert (tmp_path / "finished").exists()
        assert _queued(broker) is None
