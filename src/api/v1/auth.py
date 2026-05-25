import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.dependencies import get_current_user, get_db
from src.core.exceptions import ForbiddenError
from src.core.security import create_access_token, create_refresh_token, decode_token
from src.models.user import User
from src.repositories.api_key import SQLAlchemyApiKeyRepository
from src.repositories.user import SQLAlchemyUserRepository
from src.schemas.api_key import ApiKeyCreate, ApiKeyCreateResponse, ApiKeyListItem
from src.schemas.auth import (
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
)
from src.services.api_key import ApiKeyService
from src.services.auth import AuthService

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _auth_service(session: AsyncSession = Depends(get_db)) -> AuthService:
    return AuthService(SQLAlchemyUserRepository(session))


def _api_key_service(session: AsyncSession = Depends(get_db)) -> ApiKeyService:
    return ApiKeyService(SQLAlchemyApiKeyRepository(session))


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    data: UserCreate,
    service: AuthService = Depends(_auth_service),
) -> UserResponse:
    return await service.register(data)


@router.post("/login", response_model=TokenResponse)
async def login(
    data: UserLogin,
    service: AuthService = Depends(_auth_service),
) -> TokenResponse:
    return await service.login(data)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: RefreshRequest,
    session: AsyncSession = Depends(get_db),
) -> TokenResponse:
    payload = decode_token(body.refresh_token)
    if payload.type != "refresh":
        raise ForbiddenError("Invalid token type")
    repo = SQLAlchemyUserRepository(session)
    user = await repo.get_by_id(payload.sub)
    if user is None or not user.is_active:
        raise ForbiddenError("User not found or inactive")
    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(current_user)


@router.post(
    "/api-keys",
    response_model=ApiKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_api_key(
    data: ApiKeyCreate,
    current_user: User = Depends(get_current_user),
    service: ApiKeyService = Depends(_api_key_service),
) -> ApiKeyCreateResponse:
    api_key, raw_key = await service.create(user_id=current_user.id, name=data.name)
    return ApiKeyCreateResponse(
        id=api_key.id,
        key=raw_key,
        prefix=api_key.prefix,
        name=api_key.name,
    )


@router.get("/api-keys", response_model=list[ApiKeyListItem])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    service: ApiKeyService = Depends(_api_key_service),
) -> list[ApiKeyListItem]:
    keys = await service.list_for_user(current_user.id)
    return [ApiKeyListItem.model_validate(k) for k in keys]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: ApiKeyService = Depends(_api_key_service),
) -> None:
    await service.deactivate(key_id=key_id, requesting_user_id=current_user.id)
