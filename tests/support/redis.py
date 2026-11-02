"""A real Redis started exactly as docker-compose.yml starts the `redis` service."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import redis
from testcontainers.core.container import DockerContainer

from tests.support.compose import service
from tests.support.docker import wait_for_port_mappings

PORT = 6379


def url(container: DockerContainer, db: int = 0) -> str:
    """Re-read on every call: a restarted container gets a new host port."""
    host, port = container.get_container_host_ip(), container.get_exposed_port(PORT)
    return f"redis://{host}:{port}/{db}"


def client(container: DockerContainer) -> redis.Redis:
    return redis.Redis.from_url(url(container), decode_responses=True)


def wait_until_ready(container: DockerContainer, timeout: float = 15.0) -> None:
    """PING succeeds only once Redis has loaded its data (it answers LOADING before)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if client(container).ping():
                return
        except redis.RedisError:
            if time.monotonic() >= deadline:
                raise
        time.sleep(0.1)


@contextmanager
def compose_redis() -> Iterator[DockerContainer]:
    spec = service("redis")
    container = DockerContainer(spec["image"]).with_exposed_ports(PORT)
    if "command" in spec:
        container.with_command(spec["command"])
    # Kept active throughout: a restart publishes a new port mapping asynchronously.
    with wait_for_port_mappings():
        container.start()
        try:
            wait_until_ready(container)
            yield container
        finally:
            container.stop()
