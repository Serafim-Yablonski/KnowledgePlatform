"""Unit tests for the testcontainers port-mapping wait (no Docker)."""

from __future__ import annotations

import pytest
from testcontainers.core.docker_client import DockerClient

from tests.support.docker import wait_for_port_mappings


class _FakeLookup:
    """Stands in for DockerClient.port: 'not available' a few times, then a port."""

    def __init__(self, misses: int) -> None:
        self.misses = misses
        self.calls = 0

    def __call__(self, _client: object, container_id: str, port: int) -> str:
        self.calls += 1
        if self.calls <= self.misses:
            raise ConnectionError(f"Port mapping for {container_id}:{port} unavailable")
        return "49153"


def test_waits_until_mapping_is_published(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = _FakeLookup(misses=2)
    monkeypatch.setattr(DockerClient, "port", lookup)

    with wait_for_port_mappings(timeout=5.0):
        host_port = DockerClient.port(object(), "ryuk", 8080)  # type: ignore[arg-type]

    assert host_port == "49153"
    assert lookup.calls == 3


def test_reraises_once_deadline_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = _FakeLookup(misses=10_000)
    monkeypatch.setattr(DockerClient, "port", lookup)

    with (
        wait_for_port_mappings(timeout=0.2),
        pytest.raises(ConnectionError, match="unavailable"),
    ):
        DockerClient.port(object(), "ryuk", 8080)  # type: ignore[arg-type]

    assert lookup.calls > 1


def test_restores_original_lookup_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = _FakeLookup(misses=0)
    monkeypatch.setattr(DockerClient, "port", lookup)

    with wait_for_port_mappings():
        pass

    assert DockerClient.port is lookup
