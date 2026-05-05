"""Integration tests for SQLAlchemyUserRepository against real PostgreSQL 18."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.user import SQLAlchemyUserRepository
from src.schemas.auth import UserCreate


def _repo(session: AsyncSession) -> SQLAlchemyUserRepository:
    return SQLAlchemyUserRepository(session)


def _user_create(email: str = "alice@example.com") -> UserCreate:
    return UserCreate(email=email, password="doesnotmatter", display_name="Alice")


async def test_create_and_get_by_email(db_session: AsyncSession) -> None:
    repo = _repo(db_session)
    data = _user_create()
    user = await repo.create(data, hashed_password="$2b$hashed")
    found = await repo.get_by_email("alice@example.com")
    assert found is not None
    assert found.id == user.id
    assert found.email == "alice@example.com"
    assert found.display_name == "Alice"
    assert found.is_active is True


async def test_get_by_id(db_session: AsyncSession) -> None:
    repo = _repo(db_session)
    user = await repo.create(_user_create(), hashed_password="$2b$hashed")
    found = await repo.get_by_id(user.id)
    assert found is not None
    assert found.id == user.id


async def test_get_by_id_not_found(db_session: AsyncSession) -> None:
    repo = _repo(db_session)
    found = await repo.get_by_id(uuid.uuid4())
    assert found is None


async def test_exists_by_email_true(db_session: AsyncSession) -> None:
    repo = _repo(db_session)
    await repo.create(_user_create(), hashed_password="$2b$hashed")
    assert await repo.exists_by_email("alice@example.com") is True


async def test_exists_by_email_false(db_session: AsyncSession) -> None:
    repo = _repo(db_session)
    assert await repo.exists_by_email("nobody@example.com") is False


async def test_update_is_active(db_session: AsyncSession) -> None:
    repo = _repo(db_session)
    user = await repo.create(_user_create(), hashed_password="$2b$hashed")
    updated = await repo.update(user, is_active=False)
    assert updated.is_active is False
    refetched = await repo.get_by_id(user.id)
    assert refetched is not None
    assert refetched.is_active is False


async def test_get_by_email_returns_inactive_user(db_session: AsyncSession) -> None:
    # Partial index is for uniqueness only — get_by_email still finds inactive users.
    repo = _repo(db_session)
    user = await repo.create(_user_create(), hashed_password="$2b$hashed")
    await repo.update(user, is_active=False)
    found = await repo.get_by_email("alice@example.com")
    assert found is not None
    assert found.is_active is False


async def test_get_by_email_returns_none_for_missing(db_session: AsyncSession) -> None:
    repo = _repo(db_session)
    assert await repo.get_by_email("ghost@example.com") is None


async def test_create_sets_hashed_password(db_session: AsyncSession) -> None:
    repo = _repo(db_session)
    user = await repo.create(_user_create(), hashed_password="$2b$12$somehash")
    assert user.hashed_password == "$2b$12$somehash"
