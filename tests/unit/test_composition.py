"""Unit tests for the composition root's wiring (no DB, Redis or network)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.composition import Services
from src.repositories.api_key_cached import CachedApiKeyRepository
from src.repositories.workspace_cached import CachedWorkspaceRepository


def _services() -> Services:
    return Services(session=MagicMock(), redis=MagicMock(), http_client=MagicMock())


def test_workspace_service_uses_cached_repository() -> None:
    # Guards the drift that motivated the root: one copy once skipped the cache.
    assert isinstance(_services().workspace._repo, CachedWorkspaceRepository)


def test_api_key_service_uses_cached_repository() -> None:
    assert isinstance(_services().api_key._repo, CachedApiKeyRepository)


def test_services_are_memoized_per_instance() -> None:
    services = _services()
    assert services.search is services.search
    assert services.ai._search_service is services.search
    assert services.ai._document_service is services.document
    assert services.research._search is services.search


def test_embedding_client_is_shared_and_uses_injected_http_client() -> None:
    http_client = MagicMock()
    services = Services(session=MagicMock(), redis=MagicMock(), http_client=http_client)
    assert services.search._embedding is services.embeddings
    assert services.embeddings._http_client is http_client


def test_separate_units_of_work_do_not_share_services() -> None:
    assert _services().workspace is not _services().workspace
