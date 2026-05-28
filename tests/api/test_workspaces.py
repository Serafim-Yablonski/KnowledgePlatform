"""Full-stack API tests for workspace endpoints."""

import pytest
from httpx import AsyncClient

from tests.api.conftest import UserFactory

_REGISTER = "/api/v1/auth/register"
_LOGIN = "/api/v1/auth/login"
_WORKSPACES = "/api/v1/workspaces"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _register_and_token(
    client: AsyncClient, password: str = "password123"
) -> tuple[str, str]:
    data = UserFactory()
    email: str = data["email"]
    await client.post(_REGISTER, json={"email": email, "password": password})
    resp = await client.post(_LOGIN, json={"email": email, "password": password})
    return resp.json()["access_token"], email  # type: ignore[no-any-return]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Create workspace
# ---------------------------------------------------------------------------


async def test_create_workspace_returns_201(async_client: AsyncClient) -> None:
    token, _ = await _register_and_token(async_client)
    resp = await async_client.post(
        _WORKSPACES, json={"name": "My Team"}, headers=_auth(token)
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "My Team"
    assert body["member_count"] == 1
    assert "slug" in body
    assert body["is_active"] is True


async def test_create_workspace_unauthenticated(async_client: AsyncClient) -> None:
    resp = await async_client.post(_WORKSPACES, json={"name": "Team"})
    assert resp.status_code == 401


async def test_create_workspace_name_too_long(async_client: AsyncClient) -> None:
    token, _ = await _register_and_token(async_client)
    resp = await async_client.post(
        _WORKSPACES, json={"name": "x" * 101}, headers=_auth(token)
    )
    assert resp.status_code == 422


async def test_slug_uniqueness_two_workspaces_same_name(
    async_client: AsyncClient,
) -> None:
    token, _ = await _register_and_token(async_client)
    r1 = await async_client.post(
        _WORKSPACES, json={"name": "Engineering"}, headers=_auth(token)
    )
    r2 = await async_client.post(
        _WORKSPACES, json={"name": "Engineering"}, headers=_auth(token)
    )
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["slug"] != r2.json()["slug"]


# ---------------------------------------------------------------------------
# List workspaces
# ---------------------------------------------------------------------------


async def test_list_workspaces_returns_only_own(async_client: AsyncClient) -> None:
    token_a, _ = await _register_and_token(async_client)
    token_b, _ = await _register_and_token(async_client)

    await async_client.post(
        _WORKSPACES, json={"name": "A Team"}, headers=_auth(token_a)
    )
    await async_client.post(
        _WORKSPACES, json={"name": "B Team"}, headers=_auth(token_b)
    )

    resp_a = await async_client.get(_WORKSPACES, headers=_auth(token_a))
    assert resp_a.status_code == 200
    names_a = [w["name"] for w in resp_a.json()]
    assert "A Team" in names_a
    assert "B Team" not in names_a

    resp_b = await async_client.get(_WORKSPACES, headers=_auth(token_b))
    names_b = [w["name"] for w in resp_b.json()]
    assert "B Team" in names_b
    assert "A Team" not in names_b


# ---------------------------------------------------------------------------
# Get workspace detail
# ---------------------------------------------------------------------------


async def test_get_workspace_as_member(async_client: AsyncClient) -> None:
    token, _ = await _register_and_token(async_client)
    create_resp = await async_client.post(
        _WORKSPACES, json={"name": "Detail WS"}, headers=_auth(token)
    )
    ws_id = create_resp.json()["id"]
    resp = await async_client.get(f"{_WORKSPACES}/{ws_id}", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["id"] == ws_id


async def test_get_workspace_non_member_forbidden(async_client: AsyncClient) -> None:
    token_owner, _ = await _register_and_token(async_client)
    token_stranger, _ = await _register_and_token(async_client)
    create_resp = await async_client.post(
        _WORKSPACES, json={"name": "Private WS"}, headers=_auth(token_owner)
    )
    ws_id = create_resp.json()["id"]
    resp = await async_client.get(
        f"{_WORKSPACES}/{ws_id}", headers=_auth(token_stranger)
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Members — full flow
# ---------------------------------------------------------------------------


async def test_full_member_flow(async_client: AsyncClient) -> None:
    """Create workspace → add member → member can access → remove → access denied."""
    token_owner, _ = await _register_and_token(async_client)
    token_member, member_email = await _register_and_token(async_client)

    # Create workspace as owner
    ws_resp = await async_client.post(
        _WORKSPACES, json={"name": "Flow WS"}, headers=_auth(token_owner)
    )
    assert ws_resp.status_code == 201
    ws_id = ws_resp.json()["id"]

    # Add member
    add_resp = await async_client.post(
        f"{_WORKSPACES}/{ws_id}/members",
        json={"user_email": member_email},
        headers=_auth(token_owner),
    )
    assert add_resp.status_code == 201
    assert add_resp.json()["role"] == "member"

    # Member can access the workspace
    access_resp = await async_client.get(
        f"{_WORKSPACES}/{ws_id}", headers=_auth(token_member)
    )
    assert access_resp.status_code == 200

    # Get member's user_id from the members list
    members_resp = await async_client.get(
        f"{_WORKSPACES}/{ws_id}/members", headers=_auth(token_owner)
    )
    member_entry = next(m for m in members_resp.json() if m["email"] == member_email)
    member_user_id = member_entry["user_id"]

    # Remove the member
    remove_resp = await async_client.delete(
        f"{_WORKSPACES}/{ws_id}/members/{member_user_id}",
        headers=_auth(token_owner),
    )
    assert remove_resp.status_code == 204

    # Former member can no longer access
    denied_resp = await async_client.get(
        f"{_WORKSPACES}/{ws_id}", headers=_auth(token_member)
    )
    assert denied_resp.status_code == 403


async def test_member_cannot_add_new_members(async_client: AsyncClient) -> None:
    token_owner, _ = await _register_and_token(async_client)
    token_member, member_email = await _register_and_token(async_client)
    _, target_email = await _register_and_token(async_client)

    ws_resp = await async_client.post(
        _WORKSPACES, json={"name": "Perm WS"}, headers=_auth(token_owner)
    )
    ws_id = ws_resp.json()["id"]

    await async_client.post(
        f"{_WORKSPACES}/{ws_id}/members",
        json={"user_email": member_email},
        headers=_auth(token_owner),
    )

    # Member tries to add someone — should be 403
    resp = await async_client.post(
        f"{_WORKSPACES}/{ws_id}/members",
        json={"user_email": target_email},
        headers=_auth(token_member),
    )
    assert resp.status_code == 403


async def test_cannot_remove_last_owner(async_client: AsyncClient) -> None:
    token_owner, _ = await _register_and_token(async_client)
    ws_resp = await async_client.post(
        _WORKSPACES, json={"name": "Solo WS"}, headers=_auth(token_owner)
    )
    ws_id = ws_resp.json()["id"]

    members_resp = await async_client.get(
        f"{_WORKSPACES}/{ws_id}/members", headers=_auth(token_owner)
    )
    owner_id = members_resp.json()[0]["user_id"]

    resp = await async_client.delete(
        f"{_WORKSPACES}/{ws_id}/members/{owner_id}",
        headers=_auth(token_owner),
    )
    assert resp.status_code == 409


async def test_list_members_returns_all(async_client: AsyncClient) -> None:
    token_owner, _ = await _register_and_token(async_client)
    _, member_email = await _register_and_token(async_client)

    ws_resp = await async_client.post(
        _WORKSPACES, json={"name": "List WS"}, headers=_auth(token_owner)
    )
    ws_id = ws_resp.json()["id"]
    await async_client.post(
        f"{_WORKSPACES}/{ws_id}/members",
        json={"user_email": member_email},
        headers=_auth(token_owner),
    )

    resp = await async_client.get(
        f"{_WORKSPACES}/{ws_id}/members", headers=_auth(token_owner)
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.parametrize(
    "method,path_suffix",
    [
        ("GET", ""),
        ("POST", "/members"),
        ("GET", "/members"),
    ],
)
async def test_workspace_routes_require_auth(
    async_client: AsyncClient,
    method: str,
    path_suffix: str,
) -> None:
    fake_id = "00000000-0000-0000-0000-000000000001"
    url = f"{_WORKSPACES}/{fake_id}{path_suffix}"
    resp = await async_client.request(method, url)
    assert resp.status_code == 401
