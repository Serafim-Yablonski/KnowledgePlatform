import uuid

from fastapi import APIRouter, Depends, status

from src.core.config import get_settings
from src.core.dependencies import (
    get_api_key_service,
    get_auth_service,
    get_current_user,
)
from src.core.rate_limit import ip_rate_limit, rate_limit
from src.models.user import User
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

_cfg = get_settings()
_register_rate_limit = ip_rate_limit(
    "auth_register", _cfg.RATE_LIMIT_REGISTER_REQUESTS, _cfg.RATE_LIMIT_REGISTER_WINDOW
)
_login_rate_limit = ip_rate_limit(
    "auth_login", _cfg.RATE_LIMIT_LOGIN_REQUESTS, _cfg.RATE_LIMIT_LOGIN_WINDOW
)
_refresh_rate_limit = ip_rate_limit(
    "auth_refresh", _cfg.RATE_LIMIT_REFRESH_REQUESTS, _cfg.RATE_LIMIT_REFRESH_WINDOW
)
_api_key_create_rate_limit = rate_limit(
    "api_key_create",
    _cfg.RATE_LIMIT_API_KEY_CREATE_REQUESTS,
    _cfg.RATE_LIMIT_API_KEY_CREATE_WINDOW,
)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=201,
    dependencies=[Depends(_register_rate_limit)],
)
async def register(
    data: UserCreate,
    service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    user = await service.register(data.email, data.password, data.display_name)
    return UserResponse.model_validate(user)


@router.post(
    "/login", response_model=TokenResponse, dependencies=[Depends(_login_rate_limit)]
)
async def login(
    data: UserLogin,
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    pair = await service.login(data.email, data.password)
    return TokenResponse(
        access_token=pair.access_token, refresh_token=pair.refresh_token
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    dependencies=[Depends(_refresh_rate_limit)],
)
async def refresh(
    body: RefreshRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    pair = await service.refresh(body.refresh_token)
    return TokenResponse(
        access_token=pair.access_token, refresh_token=pair.refresh_token
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(current_user)


@router.post(
    "/api-keys",
    response_model=ApiKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_api_key_create_rate_limit)],
)
async def create_api_key(
    data: ApiKeyCreate,
    current_user: User = Depends(get_current_user),
    service: ApiKeyService = Depends(get_api_key_service),
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
    service: ApiKeyService = Depends(get_api_key_service),
) -> list[ApiKeyListItem]:
    keys = await service.list_for_user(current_user.id)
    return [ApiKeyListItem.model_validate(k) for k in keys]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: ApiKeyService = Depends(get_api_key_service),
) -> None:
    await service.deactivate(key_id=key_id, requesting_user_id=current_user.id)
