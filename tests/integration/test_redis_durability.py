"""The broker's Redis keeps every message it acknowledged across a kill (step 8a).

Redis is started with the image and command docker-compose.yml gives the `redis`
service, so dropping persistence from compose fails these tests. A kill of the Redis
process is the case that matters: the default RDB snapshots lose every write since the
last snapshot (up to an hour at our write rate), AOF loses nothing it acknowledged.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import pytest
from testcontainers.core.container import DockerContainer

from tests.support import redis as compose_redis_support


@pytest.fixture
def compose_redis() -> Generator[DockerContainer]:
    from tests.conftest import _ensure_docker_host

    _ensure_docker_host()
    with compose_redis_support.compose_redis() as container:
        yield container


def test_a_published_task_survives_a_kill_of_redis(
    compose_redis: DockerContainer,
) -> None:
    from src.workers.celery_app import QUEUE_EXTRACT, celery_app
    from src.workers.tasks.extract_text import extract_text

    document_id = str(uuid.uuid4())
    with celery_app.connection_for_write(compose_redis_support.url(compose_redis)) as c:
        extract_text.apply_async(args=[document_id], connection=c)

    # Killed right after the publish returned: no snapshot can have run since.
    wrapped = compose_redis.get_wrapped_container()
    wrapped.kill(signal="SIGKILL")
    wrapped.start()
    compose_redis_support.wait_until_ready(compose_redis)

    with celery_app.connection_for_read(compose_redis_support.url(compose_redis)) as c:
        message = c.default_channel.basic_get(QUEUE_EXTRACT, no_ack=True)
    assert message is not None, "the queued task was lost with the Redis process"
    assert message.headers["task"] == extract_text.name
    args, _kwargs, _embed = message.decode()
    assert args == [document_id]


def test_redis_never_evicts(compose_redis: DockerContainer) -> None:
    """Queued tasks have no TTL; any eviction policy could drop them silently. The
    cache needs none: every key it writes carries its own TTL (step 8, S8-13)."""
    client = compose_redis_support.client(compose_redis)
    assert client.config_get("maxmemory-policy") == {"maxmemory-policy": "noeviction"}
    assert client.config_get("maxmemory") == {"maxmemory": "0"}


def test_health_checks_the_broker(
    compose_redis: DockerContainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/health reports the broker apart from the cache: CELERY_BROKER_URL and
    REDIS_URL may name different servers, and uploads need the broker."""
    from fastapi.testclient import TestClient

    from src.main import app
    from src.workers.celery_app import celery_app

    broker_url = compose_redis_support.url(compose_redis)
    monkeypatch.setitem(celery_app.conf, "broker_url", broker_url)
    checks = TestClient(app).get("/health").json()["checks"]
    assert checks["broker"] == "ok"
    assert "cache" in checks  # no lifespan here: the cache client isn't set up

    compose_redis.get_wrapped_container().stop()
    checks = TestClient(app).get("/health").json()["checks"]
    assert checks["broker"] == "error"
