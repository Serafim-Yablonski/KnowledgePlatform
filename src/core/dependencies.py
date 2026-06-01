from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_session
from src.core.exceptions import UnauthorizedError
from src.core.redis import get_redis
from src.domain.roles import WorkspaceRole
from src.models.user import User
from src.models.workspace import Workspace
from src.services.document import DocumentService
from src.services.search import SearchService

if TYPE_CHECKING:
    from src.services.ai import AIService
    from src.services.api_key import ApiKeyService
    from src.services.auth import AuthService
    from src.services.research import ResearchService
    from src.services.workspace import WorkspaceService

_bearer = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with get_session() as session:
        yield session


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None:
        raise UnauthorizedError("Missing authentication token")

    from src.repositories.user import SQLAlchemyUserRepository
    from src.services.auth import AuthService

    repo = SQLAlchemyUserRepository(session)
    service = AuthService(repo)
    return await service.get_current_user(credentials.credentials)


def get_document_service(
    session: AsyncSession = Depends(get_db),
) -> DocumentService:
    from src.repositories.document import SQLAlchemyDocumentRepository

    return DocumentService(
        repo=SQLAlchemyDocumentRepository(session),
        session=session,
    )


def get_ai_service(
    session: AsyncSession = Depends(get_db),
) -> AIService:
    from src.ai.embeddings import EmbeddingService
    from src.core.config import get_settings
    from src.repositories.document import SQLAlchemyDocumentRepository
    from src.repositories.search import SQLAlchemySearchRepository
    from src.services.ai import AIService

    cfg = get_settings()
    embedding_svc = EmbeddingService(
        api_key=cfg.EMBEDDING_API_KEY or cfg.GOOGLE_API_KEY or "",
        model=cfg.EMBEDDING_MODEL,
        dimensions=cfg.EMBEDDING_DIMENSIONS,
    )
    return AIService(
        search_service=SearchService(
            search_repo=SQLAlchemySearchRepository(session),
            embedding_service=embedding_svc,
        ),
        document_service=DocumentService(
            repo=SQLAlchemyDocumentRepository(session),
            session=session,
        ),
    )


def get_search_service(
    session: AsyncSession = Depends(get_db),
) -> SearchService:
    from src.ai.embeddings import EmbeddingService
    from src.core.config import get_settings
    from src.repositories.search import SQLAlchemySearchRepository

    cfg = get_settings()
    embedding_svc = EmbeddingService(
        api_key=cfg.EMBEDDING_API_KEY or cfg.GOOGLE_API_KEY or "",
        model=cfg.EMBEDDING_MODEL,
        dimensions=cfg.EMBEDDING_DIMENSIONS,
    )
    return SearchService(
        search_repo=SQLAlchemySearchRepository(session),
        embedding_service=embedding_svc,
    )


def get_auth_service(session: AsyncSession = Depends(get_db)) -> AuthService:
    from src.repositories.user import SQLAlchemyUserRepository
    from src.services.auth import AuthService

    return AuthService(SQLAlchemyUserRepository(session))


def get_api_key_service(
    session: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> ApiKeyService:
    from src.core.cache import ResponseCache
    from src.repositories.api_key import SQLAlchemyApiKeyRepository
    from src.repositories.api_key_cached import CachedApiKeyRepository
    from src.services.api_key import ApiKeyService

    cache = ResponseCache(redis)
    inner = SQLAlchemyApiKeyRepository(session)
    return ApiKeyService(CachedApiKeyRepository(inner, cache))


def get_workspace_service(
    session: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> WorkspaceService:
    from src.core.cache import ResponseCache
    from src.repositories.user import SQLAlchemyUserRepository
    from src.repositories.workspace import SQLAlchemyWorkspaceRepository
    from src.repositories.workspace_cached import CachedWorkspaceRepository
    from src.services.workspace import WorkspaceService

    cache = ResponseCache(redis)
    inner_repo = SQLAlchemyWorkspaceRepository(session)
    return WorkspaceService(
        workspace_repo=CachedWorkspaceRepository(inner_repo, cache),
        user_repo=SQLAlchemyUserRepository(session),
    )


async def get_current_workspace(
    workspace_id: uuid.UUID,
    request: Request,
    user: User = Depends(get_current_user),
    service: WorkspaceService = Depends(get_workspace_service),
) -> Workspace:
    workspace, membership = await service.get_workspace_for_user(workspace_id, user.id)
    request.state.workspace_role = membership.role
    return workspace


def get_workspace_role(
    request: Request,
    _workspace: Workspace = Depends(get_current_workspace),
) -> WorkspaceRole:
    return request.state.workspace_role  # type: ignore[no-any-return]


def get_research_service(
    session: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> ResearchService:
    from src.ai.embeddings import EmbeddingService  # noqa: PLC0415
    from src.core.config import get_settings  # noqa: PLC0415
    from src.repositories.search import SQLAlchemySearchRepository  # noqa: PLC0415
    from src.services.research import ResearchService  # noqa: PLC0415

    cfg = get_settings()
    embedding_svc = EmbeddingService(
        api_key=cfg.EMBEDDING_API_KEY or cfg.GOOGLE_API_KEY or "",
        model=cfg.EMBEDDING_MODEL,
        dimensions=cfg.EMBEDDING_DIMENSIONS,
    )
    search_svc = SearchService(
        search_repo=SQLAlchemySearchRepository(session),
        embedding_service=embedding_svc,
    )
    return ResearchService(search_service=search_svc, redis_client=redis)
