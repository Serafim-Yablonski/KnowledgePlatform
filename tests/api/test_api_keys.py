"""API tests for the API key lifecycle endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.api.conftest import UserFactory


class TestApiKeyLifecycle:
    @pytest.mark.asyncio
    async def test_create_returns_raw_key_once(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.post("/api/v1/auth/api-keys", json={"name": "My Key"})
        assert resp.status_code == 201
        data = resp.json()
        assert "key" in data
        assert "prefix" in data
        assert "name" in data
        assert "id" in data
        assert data["name"] == "My Key"
        assert len(data["key"]) > 8
        assert data["key"].startswith(data["prefix"])
        assert "key_hash" not in data

    @pytest.mark.asyncio
    async def test_list_never_exposes_raw_key_or_hash(
        self, auth_client: AsyncClient
    ) -> None:
        await auth_client.post("/api/v1/auth/api-keys", json={"name": "Listed Key"})
        resp = await auth_client.get("/api/v1/auth/api-keys")
        assert resp.status_code == 200
        keys = resp.json()
        assert len(keys) >= 1
        for k in keys:
            assert "key" not in k
            assert "key_hash" not in k
            assert "prefix" in k
            assert "name" in k
            assert "is_active" in k
            assert "created_at" in k

    @pytest.mark.asyncio
    async def test_delete_deactivates_key(self, auth_client: AsyncClient) -> None:
        create_resp = await auth_client.post(
            "/api/v1/auth/api-keys", json={"name": "To Delete"}
        )
        key_id = create_resp.json()["id"]

        del_resp = await auth_client.delete(f"/api/v1/auth/api-keys/{key_id}")
        assert del_resp.status_code == 204

        list_resp = await auth_client.get("/api/v1/auth/api-keys")
        matching = [k for k in list_resp.json() if k["id"] == key_id]
        assert len(matching) == 1
        assert matching[0]["is_active"] is False

    @pytest.mark.asyncio
    async def test_max_five_active_keys(self, auth_client: AsyncClient) -> None:
        for i in range(5):
            r = await auth_client.post(
                "/api/v1/auth/api-keys", json={"name": f"key{i}"}
            )
            assert r.status_code == 201

        r = await auth_client.post("/api/v1/auth/api-keys", json={"name": "overflow"})
        assert r.status_code == 409

    @pytest.mark.asyncio
    async def test_cannot_delete_another_users_key(
        self, auth_client: AsyncClient, async_client: AsyncClient
    ) -> None:
        create_resp = await auth_client.post(
            "/api/v1/auth/api-keys", json={"name": "Owner's Key"}
        )
        key_id = create_resp.json()["id"]

        # Register and log in as a second user
        other_data = UserFactory()
        other_email: str = other_data["email"]
        await async_client.post(
            "/api/v1/auth/register",
            json={"email": other_email, "password": "password123"},
        )
        login_resp = await async_client.post(
            "/api/v1/auth/login",
            json={"email": other_email, "password": "password123"},
        )
        other_token = login_resp.json()["access_token"]
        async_client.headers["Authorization"] = f"Bearer {other_token}"

        del_resp = await async_client.delete(f"/api/v1/auth/api-keys/{key_id}")
        assert del_resp.status_code == 404
