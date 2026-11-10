"""Unit tests for the composition root's wiring (no DB, Redis or network)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.composition import Services


def _services() -> Services:
    return Services(uow=MagicMock(), redis=MagicMock(), http_client=MagicMock())


def test_services_are_memoized_per_instance() -> None:
    services = _services()
    assert services.search is services.search
    assert services.ai._search_service is services.search
    assert services.ai._document_service is services.document
    assert services.research._search is services.search


def test_embedding_client_is_shared_and_uses_injected_http_client() -> None:
    http_client = MagicMock()
    services = Services(uow=MagicMock(), redis=MagicMock(), http_client=http_client)
    assert services.search._embedding is services.embeddings
    assert services.embeddings._http_client is http_client
