"""Integration test fixtures — real PostgreSQL 18 via testcontainers-python."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User
from src.repositories.user import SQLAlchemyUserRepository
from src.schemas.auth import UserCreate


@pytest.fixture(autouse=True)
def no_celery_broker() -> Generator[None]:
    """Prevent integration tests from connecting to a live Celery broker.

    extract_text dispatches embed_chunks.delay() after success; without Redis
    running in CI/local test runs this raises ConnectionError. Patch it to a
    no-op so extract_text tests stay fast and Redis-free.
    """
    # Patch the Celery task's delay() directly so it never tries to connect
    # to a broker. The lazy import inside extract_text means we must patch
    # at the task object's attribute, not the module-level name.
    from src.workers.tasks.embed_chunks import embed_chunks

    with patch.object(embed_chunks, "delay", new=MagicMock()):
        yield


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    repo = SQLAlchemyUserRepository(db_session)
    data = UserCreate(email="fixture@example.com", password="fixturepass")
    return await repo.create(data, hashed_password="$2b$12$fixturehash")
