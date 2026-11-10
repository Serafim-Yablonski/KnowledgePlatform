"""Composition root: the one place that builds repositories and services.

Adapters (REST, MCP) and scripts get finished services from here and never construct
repositories themselves. ``Services`` takes its resources as arguments and never
opens them, so its caller owns their lifetime: in the API, the lifespan builds one
per process and hands it to the adapters as lifespan state (``src.main``). It holds
no session: each service method opens its own unit of work from the factory
(ADR-033). The pipeline workers
need none of these: their jobs run their own queries on the engine each worker
process opens, and the embed worker builds its ``EmbeddingService`` with the same
``from_settings`` (ADR-027).
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import httpx
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncEngine

from src.ai.embeddings import EmbeddingService
from src.core.cache import ResponseCache
from src.core.config import get_settings, settings
from src.repositories.file_store import LocalFileStore
from src.repositories.protocols import UnitOfWorkFactory
from src.repositories.unit_of_work import unit_of_work_factory
from src.services.ai import AIService
from src.services.api_key import ApiKeyService
from src.services.auth import AuthService
from src.services.document import DocumentService
from src.services.research import ResearchService
from src.services.search import SearchService
from src.services.workspace import WorkspaceService


class Services:
    """All services, on one unit-of-work factory.

    Built lazily: a caller that only needs ``workspace`` never builds the embedding
    client. Holds no session, so concurrent callers can share one instance.
    """

    def __init__(
        self,
        uow: UnitOfWorkFactory,
        redis: aioredis.Redis,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._uow = uow
        self._redis = redis
        self._http = http_client

    @cached_property
    def embeddings(self) -> EmbeddingService:
        return EmbeddingService.from_settings(get_settings(), self._http)

    @cached_property
    def auth(self) -> AuthService:
        return AuthService(self._uow)

    @cached_property
    def api_key(self) -> ApiKeyService:
        return ApiKeyService(self._uow, ResponseCache(self._redis))

    @cached_property
    def workspace(self) -> WorkspaceService:
        return WorkspaceService(self._uow, files=self.files)

    @cached_property
    def files(self) -> LocalFileStore:
        # The singleton, not get_settings(): that builds a fresh Settings per call.
        return LocalFileStore(Path(settings.UPLOAD_DIR))

    @cached_property
    def document(self) -> DocumentService:
        return DocumentService(self._uow, files=self.files)

    @cached_property
    def search(self) -> SearchService:
        return SearchService(self._uow, embedding_service=self.embeddings)

    @cached_property
    def ai(self) -> AIService:
        return AIService(search_service=self.search, document_service=self.document)

    @cached_property
    def research(self) -> ResearchService:
        return ResearchService(search_service=self.search, redis_client=self._redis)


def build_services(
    db: AsyncEngine, redis: aioredis.Redis, http_client: httpx.AsyncClient
) -> Services:
    return Services(unit_of_work_factory(db), redis, http_client)
