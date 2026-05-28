"""Full-stack API tests for AI ask endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient

from src.domain.ai import Answer, SourceReference
from tests.api.conftest import UserFactory

_REGISTER = "/api/v1/auth/register"
_LOGIN = "/api/v1/auth/login"
_WORKSPACES = "/api/v1/workspaces"


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


async def _create_workspace(
    client: AsyncClient, token: str, name: str = "AI WS"
) -> str:
    resp = await client.post(_WORKSPACES, json={"name": name}, headers=_auth(token))
    assert resp.status_code == 201
    return resp.json()["id"]  # type: ignore[no-any-return]


def _ask_url(workspace_id: str) -> str:
    return f"{_WORKSPACES}/{workspace_id}/ai/ask"


def _stub_ai_service(answer: str = "Test answer", confidence: float = 0.9) -> MagicMock:
    import uuid

    response = Answer(
        answer=answer,
        sources=[
            SourceReference(
                document_id=uuid.uuid4(),
                document_title="Test Doc",
                chunk_text="relevant excerpt",
                relevance_score=0.9,
            )
        ],
        confidence=confidence,
        reasoning="Found in test document.",
    )
    svc = MagicMock()
    svc.ask = AsyncMock(return_value=response)
    return svc


class TestAskEndpoint:
    async def test_ask_returns_200_with_answer_response_structure(
        self, async_client: AsyncClient
    ) -> None:
        token, _ = await _register_and_token(async_client)
        ws_id = await _create_workspace(async_client, token)

        from src.core.dependencies import get_ai_service

        with patch.object(
            __import__("src.core.dependencies", fromlist=["get_ai_service"]),
            "get_ai_service",
            return_value=_stub_ai_service(),
        ):
            from src.main import app

            app.dependency_overrides[get_ai_service] = lambda: _stub_ai_service()
            try:
                resp = await async_client.post(
                    _ask_url(ws_id),
                    json={"question": "What is the deployment process?"},
                    headers=_auth(token),
                )
            finally:
                app.dependency_overrides.pop(get_ai_service, None)

        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert "confidence" in body
        assert "reasoning" in body
        assert isinstance(body["sources"], list)
        assert 0.0 <= body["confidence"] <= 1.0

    async def test_ask_includes_rate_limit_headers(
        self, async_client: AsyncClient
    ) -> None:
        token, _ = await _register_and_token(async_client)
        ws_id = await _create_workspace(async_client, token)

        from src.core.dependencies import get_ai_service
        from src.main import app

        app.dependency_overrides[get_ai_service] = lambda: _stub_ai_service()
        try:
            resp = await async_client.post(
                _ask_url(ws_id),
                json={"question": "What is X?"},
                headers=_auth(token),
            )
        finally:
            app.dependency_overrides.pop(get_ai_service, None)

        assert resp.status_code == 200
        assert "x-ratelimit-limit" in resp.headers
        assert "x-ratelimit-remaining" in resp.headers
        assert "x-ratelimit-reset" in resp.headers

    async def test_ask_requires_authentication(self, async_client: AsyncClient) -> None:
        import uuid

        ws_id = str(uuid.uuid4())
        resp = await async_client.post(
            _ask_url(ws_id),
            json={"question": "What is X?"},
        )
        assert resp.status_code == 401

    async def test_workspace_isolation_non_member_is_forbidden(
        self, async_client: AsyncClient
    ) -> None:
        owner_token, _ = await _register_and_token(async_client)
        other_token, _ = await _register_and_token(async_client)
        ws_id = await _create_workspace(async_client, owner_token, name="Private WS")

        resp = await async_client.post(
            _ask_url(ws_id),
            json={"question": "What is in this workspace?"},
            headers=_auth(other_token),
        )
        assert resp.status_code == 403

    async def test_ask_validates_question_min_length(
        self, async_client: AsyncClient
    ) -> None:
        token, _ = await _register_and_token(async_client)
        ws_id = await _create_workspace(async_client, token)

        resp = await async_client.post(
            _ask_url(ws_id),
            json={"question": ""},
            headers=_auth(token),
        )
        assert resp.status_code == 422

    async def test_ask_validates_question_max_length(
        self, async_client: AsyncClient
    ) -> None:
        token, _ = await _register_and_token(async_client)
        ws_id = await _create_workspace(async_client, token)

        from src.core.dependencies import get_ai_service
        from src.main import app

        app.dependency_overrides[get_ai_service] = lambda: _stub_ai_service()
        try:
            resp = await async_client.post(
                _ask_url(ws_id),
                json={"question": "x" * 2001},
                headers=_auth(token),
            )
        finally:
            app.dependency_overrides.pop(get_ai_service, None)

        assert resp.status_code == 422
