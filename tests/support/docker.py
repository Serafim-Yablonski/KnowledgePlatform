"""Work around a testcontainers race with Docker Desktop's port publishing."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from testcontainers.core.docker_client import DockerClient

_POLL_INTERVAL_S = 0.05


@contextmanager
def wait_for_port_mappings(timeout: float = 10.0) -> Iterator[None]:
    """Make testcontainers wait for a container's host-port mapping to appear.

    Docker Desktop publishes port mappings asynchronously, 60-140 ms after start in
    our measurements, but testcontainers (4.14.2; unchanged in 4.15.0) reads the
    mapping once and raises ConnectionError when it is still empty. Ryuk's
    "Started!" log comes too early to cover the gap, so ~1 in 4 full test runs
    failed before any test ran. Recheck on upgrade: delete this once
    DockerClient.port waits by itself.
    """
    original = DockerClient.port

    def port(self: DockerClient, container_id: str, port: int) -> str:
        deadline = time.monotonic() + timeout
        while True:
            try:
                return original(self, container_id, port)
            except ConnectionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(_POLL_INTERVAL_S)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DockerClient, "port", port)
        yield
