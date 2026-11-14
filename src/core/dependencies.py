"""FastAPI glue: the process's services and clients, and the request's user and
workspace (wiring: src.composition; built in the lifespan, src.main)."""

from __future__ import annotations

import uuid

import redis.asyncio as aioredis
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.composition import Services
from src.core.exceptions import UnauthorizedError
from src.domain.roles import WorkspaceRole
from src.models.user import User
from src.models.workspace import Workspace
from src.services.ai import AIService
from src.services.api_key import ApiKeyService
from src.services.auth import AuthService
from src.services.document import DocumentService
from src.services.research import ResearchService
from src.services.search import SearchService
from src.services.workspace import WorkspaceService

_bearer = HTTPBearer(auto_error=False)


def get_services(request: Request) -> Services:
    """The process's one ``Services``, from the lifespan's state (ADR-033)."""
    services: Services = request.state.services
    return services


def get_redis(request: Request) -> aioredis.Redis:
    """The process's Redis client (the rate limiters), from the lifespan's state."""
    redis: aioredis.Redis = request.state.redis
    return redis


def get_auth_service(services: Services = Depends(get_services)) -> AuthService:
    return services.auth


def get_api_key_service(services: Services = Depends(get_services)) -> ApiKeyService:
    return services.api_key


def get_workspace_service(
    services: Services = Depends(get_services),
) -> WorkspaceService:
    return services.workspace


def get_document_service(
    services: Services = Depends(get_services),
) -> DocumentService:
    return services.document


def get_search_service(services: Services = Depends(get_services)) -> SearchService:
    return services.search


def get_ai_service(services: Services = Depends(get_services)) -> AIService:
    return services.ai


def get_research_service(
    services: Services = Depends(get_services),
) -> ResearchService:
    return services.research


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    service: AuthService = Depends(get_auth_service),
) -> User:
    if credentials is None:
        raise UnauthorizedError("Missing authentication token")
    return await service.get_current_user(credentials.credentials)


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
