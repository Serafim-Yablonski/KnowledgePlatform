from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_session
from src.core.exceptions import ForbiddenError
from src.core.redis import get_redis
from src.models.user import User
from src.models.workspace import Workspace
from src.services.document import DocumentService
from src.services.search import SearchService

if TYPE_CHECKING:
    from src.services.ai import AIService
    from src.services.research import ResearchService

_bearer = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with get_session() as session:
        yield session


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None:
        raise ForbiddenError("Missing authentication token")

    from src.repositories.user import SQLAlchemyUserRepository
    from src.services.auth import AuthService

    repo = SQLAlchemyUserRepository(session)
    service = AuthService(repo)
    return await service.get_current_user(credentials.credentials)


def get_document_service(session: AsyncSession = Depends(get_db)) -> DocumentService:
    from src.repositories.document import SQLAlchemyDocumentRepository

    return DocumentService(repo=SQLAlchemyDocumentRepository(session), session=session)


async def get_current_workspace(
    workspace_id: uuid.UUID,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> Workspace:
    from src.repositories.workspace import SQLAlchemyWorkspaceRepository

    repo = SQLAlchemyWorkspaceRepository(session)
    membership = await repo.get_membership(workspace_id, user.id)
    if membership is None:
        raise ForbiddenError("Not a member of this workspace")
    request.state.workspace_role = membership.role
    workspace = await repo.get_by_id(workspace_id)
    if workspace is None:
        raise ForbiddenError("Not a member of this workspace")
    return workspace


def get_ai_service(
    session: AsyncSession = Depends(get_db),
    redis: Any = Depends(get_redis),
) -> AIService:
    from src.ai.embeddings import EmbeddingService
    from src.core.cache import ResponseCache
    from src.core.config import get_settings
    from src.repositories.document import SQLAlchemyDocumentRepository
    from src.repositories.search import SQLAlchemySearchRepository
    from src.services.ai import AIService

    cfg = get_settings()
    embedding_svc = EmbeddingService(
        api_key=cfg.EMBEDDING_API_KEY or cfg.GOOGLE_API_KEY or "",
        model=cfg.EMBEDDING_MODEL,
        dimensions=cfg.EMBEDDING_DIMENSIONS,
        redis_client=redis,
    )
    return AIService(
        search_service=SearchService(
            search_repo=SQLAlchemySearchRepository(session),
            embedding_service=embedding_svc,
            cache=ResponseCache(redis, key_prefix="nexus:ai_search"),
        ),
        document_service=DocumentService(
            repo=SQLAlchemyDocumentRepository(session),
            session=session,
        ),
    )


def get_search_service(
    session: AsyncSession = Depends(get_db),
    redis: Any = Depends(get_redis),
) -> SearchService:
    from src.ai.embeddings import EmbeddingService
    from src.core.cache import ResponseCache
    from src.core.config import get_settings
    from src.repositories.search import SQLAlchemySearchRepository

    cfg = get_settings()
    embedding_svc = EmbeddingService(
        api_key=cfg.EMBEDDING_API_KEY or cfg.GOOGLE_API_KEY or "",
        model=cfg.EMBEDDING_MODEL,
        dimensions=cfg.EMBEDDING_DIMENSIONS,
        redis_client=redis,
    )
    return SearchService(
        search_repo=SQLAlchemySearchRepository(session),
        embedding_service=embedding_svc,
        cache=ResponseCache(redis),
    )


def get_auth_service(session: AsyncSession = Depends(get_db)) -> Any:
    from src.repositories.user import SQLAlchemyUserRepository
    from src.services.auth import AuthService

    return AuthService(SQLAlchemyUserRepository(session))


def get_api_key_service(session: AsyncSession = Depends(get_db)) -> Any:
    from src.repositories.api_key import SQLAlchemyApiKeyRepository
    from src.services.api_key import ApiKeyService

    return ApiKeyService(SQLAlchemyApiKeyRepository(session))


def get_workspace_service(session: AsyncSession = Depends(get_db)) -> Any:
    from src.repositories.user import SQLAlchemyUserRepository
    from src.repositories.workspace import SQLAlchemyWorkspaceRepository
    from src.services.workspace import WorkspaceService

    return WorkspaceService(
        workspace_repo=SQLAlchemyWorkspaceRepository(session),
        user_repo=SQLAlchemyUserRepository(session),
    )


def get_research_service(
    session: AsyncSession = Depends(get_db),
    redis: Any = Depends(get_redis),
) -> ResearchService:
    from src.ai.embeddings import EmbeddingService  # noqa: PLC0415
    from src.core.cache import ResponseCache  # noqa: PLC0415
    from src.core.config import get_settings  # noqa: PLC0415
    from src.repositories.search import SQLAlchemySearchRepository  # noqa: PLC0415
    from src.services.research import ResearchService  # noqa: PLC0415

    cfg = get_settings()
    embedding_svc = EmbeddingService(
        api_key=cfg.EMBEDDING_API_KEY or cfg.GOOGLE_API_KEY or "",
        model=cfg.EMBEDDING_MODEL,
        dimensions=cfg.EMBEDDING_DIMENSIONS,
        redis_client=redis,
    )
    search_svc = SearchService(
        search_repo=SQLAlchemySearchRepository(session),
        embedding_service=embedding_svc,
        cache=ResponseCache(redis, key_prefix="nexus:research_search"),
    )
    return ResearchService(search_service=search_svc, redis_client=redis)
