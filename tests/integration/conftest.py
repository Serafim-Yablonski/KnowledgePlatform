"""Integration test fixtures — real PostgreSQL 18 via testcontainers-python."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
async def test_user(db_session: AsyncSession) -> None:
    # TODO: implement after src/models/user.py is created
    pytest.skip("User model not yet implemented")
