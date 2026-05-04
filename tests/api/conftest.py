"""API test fixtures — httpx.AsyncClient against the full async FastAPI stack."""

import pytest
from httpx import AsyncClient

# async_client is provided by tests/conftest.py and available here automatically.


@pytest.fixture
async def auth_client(async_client: AsyncClient) -> AsyncClient:
    # TODO: implement after User model and /auth/token endpoint exist.
    # Steps:
    #   1. POST /auth/register with UserFactory-generated credentials
    #   2. POST /auth/token to obtain JWT
    #   3. async_client.headers["Authorization"] = f"Bearer {token}"
    #   4. return async_client
    pytest.skip("Auth not yet implemented")
    return async_client


# ---------------------------------------------------------------------------
# UserFactory — stub until src/models/user.py exists
# ---------------------------------------------------------------------------
# import factory
# from src.models.user import User
# from src.core.security import get_password_hash
#
# class UserFactory(factory.Factory):
#     class Meta:
#         model = User
#     email = factory.Sequence(lambda n: f"user{n}@example.com")
#     hashed_password = factory.LazyFunction(lambda: get_password_hash("password"))
