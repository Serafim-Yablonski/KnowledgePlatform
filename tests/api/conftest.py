"""API test fixtures — httpx.AsyncClient against the full async FastAPI stack."""

import factory
import pytest
from httpx import AsyncClient


class UserFactory(factory.Factory):  # type: ignore[misc]
    class Meta:
        model = dict

    email: str = factory.Sequence(lambda n: f"user{n}@example.com")
    password: str = "password123"
    display_name: str = factory.Sequence(lambda n: f"User {n}")


@pytest.fixture
async def auth_client(async_client: AsyncClient) -> AsyncClient:
    data = UserFactory()
    await async_client.post("/api/v1/auth/register", json=data)
    resp = await async_client.post(
        "/api/v1/auth/login",
        json={"email": data["email"], "password": data["password"]},
    )
    token = resp.json()["access_token"]
    async_client.headers["Authorization"] = f"Bearer {token}"
    return async_client
