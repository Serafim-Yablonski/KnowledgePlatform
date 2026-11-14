from sqlalchemy.exc import IntegrityError

from src.core.exceptions import (
    ConflictError,
    InvalidTokenError,
    UnauthorizedError,
)
from src.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from src.domain.user import TokenPair, UserCreationInput
from src.models.user import User
from src.repositories.protocols import UnitOfWorkFactory


class AuthService:
    def __init__(self, uow: UnitOfWorkFactory) -> None:
        self._uow = uow

    async def register(
        self, email: str, password: str, display_name: str | None = None
    ) -> User:
        hashed = await hash_password(password)
        creation_input = UserCreationInput(email=email, display_name=display_name)
        async with self._uow() as uow:
            try:
                user = await uow.users.create(creation_input, hashed)
            except IntegrityError as exc:
                raise ConflictError("Email already registered") from exc
            await uow.commit()
        return user

    async def login(self, email: str, password: str) -> TokenPair:
        async with self._uow() as uow:
            user = await uow.users.get_by_email(email)
        if user is None or not await verify_password(password, user.hashed_password):
            raise UnauthorizedError("Invalid credentials")
        if not user.is_active:
            raise UnauthorizedError("Invalid credentials")
        return TokenPair(
            access_token=create_access_token(user.id),
            refresh_token=create_refresh_token(user.id),
        )

    async def refresh(self, refresh_token: str) -> TokenPair:
        payload = decode_token(refresh_token)
        if payload.type != "refresh":
            raise InvalidTokenError("Invalid token type")
        async with self._uow() as uow:
            user = await uow.users.get_by_id(payload.sub)
        if user is None or not user.is_active:
            raise InvalidTokenError("User not found or inactive")
        return TokenPair(
            access_token=create_access_token(user.id),
            refresh_token=create_refresh_token(user.id),
        )

    async def get_current_user(self, token: str) -> User:
        payload = decode_token(token)
        if payload.type != "access":
            raise InvalidTokenError("Invalid token type")
        async with self._uow() as uow:
            user = await uow.users.get_by_id(payload.sub)
        if user is None or not user.is_active:
            raise InvalidTokenError("User not found or inactive")
        return user
