from sqlalchemy.exc import IntegrityError

from src.core.exceptions import ConflictError, ForbiddenError
from src.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from src.domain.user import UserCreationInput
from src.models.user import User
from src.repositories.protocols import UserRepositoryProtocol
from src.schemas.auth import TokenResponse, UserCreate, UserLogin, UserResponse


class AuthService:
    def __init__(self, user_repo: UserRepositoryProtocol) -> None:
        self._repo = user_repo

    async def register(self, data: UserCreate) -> UserResponse:
        hashed = await hash_password(data.password)
        creation_input = UserCreationInput(
            email=data.email, display_name=data.display_name
        )
        try:
            user = await self._repo.create(creation_input, hashed)
        except IntegrityError as exc:
            raise ConflictError("Email already registered") from exc
        return UserResponse.model_validate(user)

    async def login(self, data: UserLogin) -> TokenResponse:
        user = await self._repo.get_by_email(data.email)
        if user is None or not await verify_password(
            data.password, user.hashed_password
        ):
            raise ForbiddenError("Invalid credentials")
        if not user.is_active:
            raise ForbiddenError("Invalid credentials")
        return TokenResponse(
            access_token=create_access_token(user.id),
            refresh_token=create_refresh_token(user.id),
        )

    async def refresh(self, refresh_token: str) -> TokenResponse:
        payload = decode_token(refresh_token)
        if payload.type != "refresh":
            raise ForbiddenError("Invalid token type")
        user = await self._repo.get_by_id(payload.sub)
        if user is None or not user.is_active:
            raise ForbiddenError("User not found or inactive")
        return TokenResponse(
            access_token=create_access_token(user.id),
            refresh_token=create_refresh_token(user.id),
        )

    async def get_current_user(self, token: str) -> User:
        payload = decode_token(token)
        if payload.type != "access":
            raise ForbiddenError("Invalid token type")
        user = await self._repo.get_by_id(payload.sub)
        if user is None or not user.is_active:
            raise ForbiddenError("User not found or inactive")
        return user
