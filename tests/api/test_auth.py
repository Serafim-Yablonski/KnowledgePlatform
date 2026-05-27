"""API-level auth tests via httpx.AsyncClient against the full async stack."""

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.api.conftest import UserFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGISTER_URL = "/api/v1/auth/register"
_LOGIN_URL = "/api/v1/auth/login"
_REFRESH_URL = "/api/v1/auth/refresh"
_ME_URL = "/api/v1/auth/me"

_DEFAULT_PASSWORD = "password123"


async def _register_and_login(client: AsyncClient) -> tuple[dict[str, str], str]:
    data = UserFactory()
    email: str = data["email"]
    await client.post(
        _REGISTER_URL,
        json={"email": email, "password": _DEFAULT_PASSWORD},
    )
    resp = await client.post(
        _LOGIN_URL,
        json={"email": email, "password": _DEFAULT_PASSWORD},
    )
    return resp.json(), email  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


async def test_register_creates_user(async_client: AsyncClient) -> None:
    email = UserFactory()["email"]
    resp = await async_client.post(
        _REGISTER_URL,
        json={"email": email, "password": _DEFAULT_PASSWORD},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == email
    assert body["is_active"] is True
    assert "id" in body
    assert "created_at" in body
    assert "hashed_password" not in body


async def test_register_with_display_name(async_client: AsyncClient) -> None:
    email = UserFactory()["email"]
    resp = await async_client.post(
        _REGISTER_URL,
        json={
            "email": email,
            "password": _DEFAULT_PASSWORD,
            "display_name": "Alice",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["display_name"] == "Alice"


async def test_register_duplicate_email(async_client: AsyncClient) -> None:
    email = UserFactory()["email"]
    payload = {"email": email, "password": _DEFAULT_PASSWORD}
    await async_client.post(_REGISTER_URL, json=payload)
    resp = await async_client.post(_REGISTER_URL, json=payload)
    assert resp.status_code == 409


async def test_register_short_password(async_client: AsyncClient) -> None:
    email = UserFactory()["email"]
    resp = await async_client.post(
        _REGISTER_URL, json={"email": email, "password": "short"}
    )
    assert resp.status_code == 422


async def test_register_invalid_email(async_client: AsyncClient) -> None:
    resp = await async_client.post(
        _REGISTER_URL, json={"email": "not-an-email", "password": _DEFAULT_PASSWORD}
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


async def test_login_returns_tokens(async_client: AsyncClient) -> None:
    email = UserFactory()["email"]
    await async_client.post(
        _REGISTER_URL,
        json={"email": email, "password": _DEFAULT_PASSWORD},
    )
    resp = await async_client.post(
        _LOGIN_URL,
        json={"email": email, "password": _DEFAULT_PASSWORD},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert "refresh_token" in body
    assert body["token_type"] == "bearer"


async def test_login_wrong_password(async_client: AsyncClient) -> None:
    email = UserFactory()["email"]
    await async_client.post(
        _REGISTER_URL,
        json={"email": email, "password": _DEFAULT_PASSWORD},
    )
    resp = await async_client.post(
        _LOGIN_URL,
        json={"email": email, "password": "wrongpassword"},
    )
    assert resp.status_code == 403


async def test_login_nonexistent_email(async_client: AsyncClient) -> None:
    resp = await async_client.post(
        _LOGIN_URL,
        json={"email": "nobody@example.com", "password": _DEFAULT_PASSWORD},
    )
    assert resp.status_code == 403


async def test_login_error_message_does_not_reveal_existence(
    async_client: AsyncClient,
) -> None:
    email = UserFactory()["email"]
    await async_client.post(
        _REGISTER_URL,
        json={"email": email, "password": _DEFAULT_PASSWORD},
    )
    wrong_pw = await async_client.post(
        _LOGIN_URL,
        json={"email": email, "password": "wrongpassword"},
    )
    no_user = await async_client.post(
        _LOGIN_URL,
        json={"email": "ghost@example.com", "password": _DEFAULT_PASSWORD},
    )
    assert wrong_pw.json()["error"]["message"] == no_user.json()["error"]["message"]


# ---------------------------------------------------------------------------
# /me
# ---------------------------------------------------------------------------


async def test_me_with_valid_token(async_client: AsyncClient) -> None:
    tokens, email = await _register_and_login(async_client)
    resp = await async_client.get(
        _ME_URL, headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == email


async def test_me_without_token(async_client: AsyncClient) -> None:
    resp = await async_client.get(_ME_URL)
    assert resp.status_code == 401


async def test_me_with_invalid_token(async_client: AsyncClient) -> None:
    resp = await async_client.get(
        _ME_URL, headers={"Authorization": "Bearer invalid.token.here"}
    )
    assert resp.status_code == 403


async def test_me_with_refresh_token_fails(async_client: AsyncClient) -> None:
    tokens, _ = await _register_and_login(async_client)
    resp = await async_client.get(
        _ME_URL, headers={"Authorization": f"Bearer {tokens['refresh_token']}"}
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


async def test_refresh_returns_new_access_token(async_client: AsyncClient) -> None:
    tokens, _ = await _register_and_login(async_client)
    resp = await async_client.post(
        _REFRESH_URL, json={"refresh_token": tokens["refresh_token"]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


async def test_refresh_with_access_token_fails(async_client: AsyncClient) -> None:
    tokens, _ = await _register_and_login(async_client)
    resp = await async_client.post(
        _REFRESH_URL, json={"refresh_token": tokens["access_token"]}
    )
    assert resp.status_code == 403


async def test_refresh_with_invalid_token(async_client: AsyncClient) -> None:
    resp = await async_client.post(
        _REFRESH_URL, json={"refresh_token": "not.a.valid.token"}
    )
    assert resp.status_code == 403


async def test_refresh_with_deactivated_user(
    async_client: AsyncClient, db_session: AsyncSession
) -> None:
    from sqlalchemy import update

    from src.models.user import User

    tokens, _ = await _register_and_login(async_client)
    await db_session.execute(update(User).values(is_active=False))
    await db_session.commit()
    resp = await async_client.post(
        _REFRESH_URL, json={"refresh_token": tokens["refresh_token"]}
    )
    assert resp.status_code == 403
