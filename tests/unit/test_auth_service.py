"""Unit tests for AuthService using an in-memory Protocol stub."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError

from src.core.exceptions import ConflictError, ForbiddenError
from src.core.security import create_access_token
from src.models.user import User
from src.schemas.auth import TokenResponse, UserCreate, UserLogin, UserResponse
from src.services.auth import AuthService


class StubUserRepository:
    """In-memory implementation of UserRepositoryProtocol for unit testing."""

    def __init__(self) -> None:
        self._store: dict[uuid.UUID, User] = {}

    def _by_email(self, email: str) -> User | None:
        return next((u for u in self._store.values() if u.email == email), None)

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return self._store.get(user_id)

    async def get_by_email(self, email: str) -> User | None:
        return self._by_email(email)

    async def create(self, data: UserCreate, hashed_password: str) -> User:
        # Simulate partial unique index: raise IntegrityError for active-email clash.
        existing = self._by_email(data.email)
        if existing is not None and existing.is_active:
            raise IntegrityError(None, None, Exception("unique constraint"))
        user = User()
        user.id = uuid.uuid4()
        user.email = data.email
        user.hashed_password = hashed_password
        user.display_name = data.display_name
        user.is_active = True
        user.created_at = datetime.now(UTC).replace(tzinfo=None)
        user.updated_at = datetime.now(UTC).replace(tzinfo=None)
        self._store[user.id] = user
        return user

    async def update(self, user: User, *, is_active: bool | None = None) -> User:
        if is_active is not None:
            user.is_active = is_active
        return user

    async def exists_by_email(self, email: str) -> bool:
        found = self._by_email(email)
        return found is not None and found.is_active


def _make_service() -> tuple[AuthService, StubUserRepository]:
    repo = StubUserRepository()
    return AuthService(repo), repo


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


async def test_register_success() -> None:
    service, _ = _make_service()
    data = UserCreate(
        email="alice@example.com", password="secret99", display_name="Alice"
    )
    result = await service.register(data)
    assert isinstance(result, UserResponse)
    assert result.email == "alice@example.com"
    assert result.display_name == "Alice"
    assert result.is_active is True


async def test_register_conflict() -> None:
    service, _ = _make_service()
    data = UserCreate(email="alice@example.com", password="secret99")
    await service.register(data)
    with pytest.raises(ConflictError, match="Email already registered"):
        await service.register(data)


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------


async def test_login_success() -> None:
    service, _ = _make_service()
    await service.register(UserCreate(email="bob@example.com", password="hunter22"))
    result = await service.login(
        UserLogin(email="bob@example.com", password="hunter22")
    )
    assert isinstance(result, TokenResponse)
    assert result.access_token
    assert result.refresh_token
    assert result.token_type == "bearer"


async def test_login_wrong_password() -> None:
    service, _ = _make_service()
    await service.register(UserCreate(email="bob@example.com", password="hunter22"))
    with pytest.raises(ForbiddenError, match="Invalid credentials"):
        await service.login(UserLogin(email="bob@example.com", password="wrongpass"))


async def test_login_nonexistent_email() -> None:
    service, _ = _make_service()
    with pytest.raises(ForbiddenError, match="Invalid credentials"):
        await service.login(UserLogin(email="ghost@example.com", password="whatever"))


async def test_login_inactive_user() -> None:
    service, repo = _make_service()
    user_data = UserCreate(email="inactive@example.com", password="secret99")
    response = await service.register(user_data)
    user = await repo.get_by_id(response.id)
    assert user is not None
    await repo.update(user, is_active=False)
    with pytest.raises(ForbiddenError, match="Invalid credentials"):
        await service.login(
            UserLogin(email="inactive@example.com", password="secret99")
        )


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------


async def test_get_current_user_success() -> None:
    service, repo = _make_service()
    user_data = UserCreate(email="carol@example.com", password="secret99")
    response = await service.register(user_data)
    token = create_access_token(response.id)
    user = await service.get_current_user(token)
    assert user.email == "carol@example.com"


async def test_get_current_user_not_found() -> None:
    service, _ = _make_service()
    token = create_access_token(uuid.uuid4())
    with pytest.raises(ForbiddenError, match="User not found or inactive"):
        await service.get_current_user(token)


async def test_get_current_user_inactive() -> None:
    service, repo = _make_service()
    user_data = UserCreate(email="dave@example.com", password="secret99")
    response = await service.register(user_data)
    user = await repo.get_by_id(response.id)
    assert user is not None
    await repo.update(user, is_active=False)
    token = create_access_token(response.id)
    with pytest.raises(ForbiddenError, match="User not found or inactive"):
        await service.get_current_user(token)


async def test_get_current_user_invalid_token() -> None:
    service, _ = _make_service()
    with pytest.raises(ForbiddenError):
        await service.get_current_user("not.a.token")


async def test_get_current_user_rejects_refresh_token() -> None:
    from src.core.security import create_refresh_token

    service, _ = _make_service()
    token = create_refresh_token(uuid.uuid4())
    with pytest.raises(ForbiddenError, match="Invalid token type"):
        await service.get_current_user(token)


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


async def test_refresh_success() -> None:
    from src.core.security import create_refresh_token

    service, repo = _make_service()
    user_data = UserCreate(email="frank@example.com", password="secret99")
    response = await service.register(user_data)
    token = create_refresh_token(response.id)

    result = await service.refresh(token)

    assert isinstance(result, TokenResponse)
    assert result.access_token
    assert result.refresh_token
    assert result.token_type == "bearer"


async def test_refresh_rejects_access_token() -> None:
    service, _ = _make_service()
    token = create_access_token(uuid.uuid4())
    with pytest.raises(ForbiddenError, match="Invalid token type"):
        await service.refresh(token)


async def test_refresh_rejects_invalid_token() -> None:
    service, _ = _make_service()
    with pytest.raises(ForbiddenError):
        await service.refresh("not.a.token")


async def test_refresh_rejects_unknown_user() -> None:
    from src.core.security import create_refresh_token

    service, _ = _make_service()
    token = create_refresh_token(uuid.uuid4())
    with pytest.raises(ForbiddenError, match="User not found or inactive"):
        await service.refresh(token)


async def test_refresh_rejects_inactive_user() -> None:
    from src.core.security import create_refresh_token

    service, repo = _make_service()
    user_data = UserCreate(email="grace@example.com", password="secret99")
    response = await service.register(user_data)
    user = await repo.get_by_id(response.id)
    assert user is not None
    await repo.update(user, is_active=False)
    token = create_refresh_token(response.id)
    with pytest.raises(ForbiddenError, match="User not found or inactive"):
        await service.refresh(token)


# ---------------------------------------------------------------------------
# password never stored in plaintext
# ---------------------------------------------------------------------------


async def test_password_is_hashed_in_repository() -> None:
    _, repo = _make_service()
    service = AuthService(repo)
    data = UserCreate(email="eve@example.com", password="plaintext1")
    await service.register(data)
    user = await repo.get_by_email("eve@example.com")
    assert user is not None
    assert user.hashed_password != "plaintext1"
    assert user.hashed_password.startswith("$2")
