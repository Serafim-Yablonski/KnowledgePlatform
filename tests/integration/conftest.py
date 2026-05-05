"""Integration test fixtures — real PostgreSQL 18 via testcontainers-python."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User
from src.repositories.user import SQLAlchemyUserRepository
from src.schemas.auth import UserCreate


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    repo = SQLAlchemyUserRepository(db_session)
    data = UserCreate(email="fixture@example.com", password="fixturepass")
    return await repo.create(data, hashed_password="$2b$12$fixturehash")
