"""Composition root: the one place that builds repositories and services.

Adapters (REST, MCP), scripts and workers get finished services from here and never
construct repositories themselves. ``Services`` takes its resources as arguments and
never opens them, so each caller owns their lifetime: FastAPI per request,
``open_services()`` per MCP call, a Celery task with its own engine and clients.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import cached_property

import httpx
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.embeddings import EmbeddingService
from src.core.cache import ResponseCache
from src.core.config import get_settings
from src.core.database import get_session
from src.core.http import get_async_http_client
from src.core.redis import get_async_redis_client
from src.repositories.api_key import SQLAlchemyApiKeyRepository
from src.repositories.api_key_cached import CachedApiKeyRepository
from src.repositories.document import SQLAlchemyDocumentRepository
from src.repositories.search import SQLAlchemySearchRepository
from src.repositories.user import SQLAlchemyUserRepository
from src.repositories.workspace import SQLAlchemyWorkspaceRepository
from src.repositories.workspace_cached import CachedWorkspaceRepository
from src.services.ai import AIService
from src.services.api_key import ApiKeyService
from src.services.auth import AuthService
from src.services.document import DocumentService
from src.services.research import ResearchService
from src.services.search import SearchService
from src.services.workspace import WorkspaceService


class Services:
    """All services for one unit of work, sharing one DB session.

    Built lazily: a request that only needs ``workspace`` never builds the embedding
    client. Not shareable across concurrent requests — the session isn't.
    """

    def __init__(
        self,
        session: AsyncSession,
        redis: aioredis.Redis,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._session = session
        self._redis = redis
        self._http = http_client

    @cached_property
    def _cache(self) -> ResponseCache:
        return ResponseCache(self._redis)

    @cached_property
    def embeddings(self) -> EmbeddingService:
        cfg = get_settings()
        return EmbeddingService(
            api_key=cfg.EMBEDDING_API_KEY or cfg.GOOGLE_API_KEY or "",
            model=cfg.EMBEDDING_MODEL,
            dimensions=cfg.EMBEDDING_DIMENSIONS,
            http_client=self._http,
        )

    @cached_property
    def auth(self) -> AuthService:
        return AuthService(SQLAlchemyUserRepository(self._session))

    @cached_property
    def api_key(self) -> ApiKeyService:
        inner = SQLAlchemyApiKeyRepository(self._session)
        return ApiKeyService(CachedApiKeyRepository(inner, self._cache))

    @cached_property
    def workspace(self) -> WorkspaceService:
        inner = SQLAlchemyWorkspaceRepository(self._session)
        return WorkspaceService(
            workspace_repo=CachedWorkspaceRepository(inner, self._cache),
            user_repo=SQLAlchemyUserRepository(self._session),
        )

    @cached_property
    def document(self) -> DocumentService:
        return DocumentService(
            repo=SQLAlchemyDocumentRepository(self._session),
            session=self._session,
        )

    @cached_property
    def search(self) -> SearchService:
        return SearchService(
            search_repo=SQLAlchemySearchRepository(self._session),
            embedding_service=self.embeddings,
        )

    @cached_property
    def ai(self) -> AIService:
        return AIService(search_service=self.search, document_service=self.document)

    @cached_property
    def research(self) -> ResearchService:
        return ResearchService(search_service=self.search, redis_client=self._redis)


@asynccontextmanager
async def open_services() -> AsyncGenerator[Services]:
    """One unit of work over the API process's pools, for callers outside FastAPI."""
    async with get_session() as session:
        yield Services(session, get_async_redis_client(), get_async_http_client())
