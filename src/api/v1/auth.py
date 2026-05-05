from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.dependencies import get_current_user, get_db
from src.core.exceptions import ForbiddenError
from src.core.security import create_access_token, create_refresh_token, decode_token
from src.models.user import User
from src.repositories.user import SQLAlchemyUserRepository
from src.schemas.auth import (
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
)
from src.services.auth import AuthService

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _auth_service(session: AsyncSession = Depends(get_db)) -> AuthService:
    return AuthService(SQLAlchemyUserRepository(session))


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
