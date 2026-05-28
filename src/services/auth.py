from sqlalchemy.exc import IntegrityError

from src.core.exceptions import ConflictError, ForbiddenError
from src.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from src.domain.user import TokenPair, UserCreationInput
from src.models.user import User
from src.repositories.protocols import UserRepositoryProtocol


class AuthService:
    def __init__(self, user_repo: UserRepositoryProtocol) -> None:
        self._repo = user_repo

    async def register(
        self, email: str, password: str, display_name: str | None = None
    ) -> User:
        hashed = await hash_password(password)
        creation_input = UserCreationInput(email=email, display_name=display_name)
        try:
            return await self._repo.create(creation_input, hashed)
        except IntegrityError as exc:
            raise ConflictError("Email already registered") from exc

    async def login(self, email: str, password: str) -> TokenPair:
        user = await self._repo.get_by_email(email)
        if user is None or not await verify_password(password, user.hashed_password):
            raise ForbiddenError("Invalid credentials")
        if not user.is_active:
            raise ForbiddenError("Invalid credentials")
        return TokenPair(
            access_token=create_access_token(user.id),
            refresh_token=create_refresh_token(user.id),
        )

    async def refresh(self, refresh_token: str) -> TokenPair:
        payload = decode_token(refresh_token)
        if payload.type != "refresh":
            raise ForbiddenError("Invalid token type")
        user = await self._repo.get_by_id(payload.sub)
        if user is None or not user.is_active:
            raise ForbiddenError("User not found or inactive")
        return TokenPair(
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
