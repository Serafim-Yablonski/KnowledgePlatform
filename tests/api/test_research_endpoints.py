"""API tests for the LangGraph research workflow endpoints."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient

from src.core.dependencies import get_research_service
from src.main import app
from src.schemas.research import (
    ResearchPlanResponse,
    ResearchStatusResponse,
)
from src.services.research import ResearchService

_THREAD_ID = "00000000-0000-0000-0000-000000000001"
_THREAD_ID_STATUS = "00000000-0000-0000-0000-000000000002"
_THREAD_ID_DONE = "00000000-0000-0000-0000-000000000003"
_THREAD_ID_STREAM = "00000000-0000-0000-0000-000000000004"
_THREAD_ID_REVIEW_A = "00000000-0000-0000-0000-000000000005"
_THREAD_ID_REVIEW_B = "00000000-0000-0000-0000-000000000006"


def _make_research_service(
    thread_id: str = _THREAD_ID,
    status: str = "running",
    synthesis: str | None = None,
) -> ResearchService:
    svc = MagicMock(spec=ResearchService)
    svc.start_research = AsyncMock(return_value=thread_id)
    svc.get_status = AsyncMock(
        return_value=ResearchStatusResponse(
            thread_id=thread_id,
            status=status,  # type: ignore[arg-type]
            topic="test topic",
            plan=ResearchPlanResponse(
                queries=["q1"],
                scope="scope",
                expected_sections=["A"],
            ),
            findings_count=3,
            synthesis=synthesis,
            human_approved=False,
        )
    )

    async def _stream_gen(*args, **kwargs):  # type: ignore[no-untyped-def]
        yield "Hello "
        yield "world"
        yield "__DONE__"

    svc.stream_synthesis = _stream_gen
    svc.review = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def research_service_mock() -> MagicMock:
    svc = _make_research_service()
    app.dependency_overrides[get_research_service] = lambda: svc
    yield svc
    app.dependency_overrides.pop(get_research_service, None)


class TestStartResearch:
    @pytest.mark.asyncio
    async def test_returns_202_with_thread_id(
        self,
        auth_client: AsyncClient,
        research_service_mock: MagicMock,
    ) -> None:
        resp = await auth_client.post(
            f"/api/v1/workspaces/{await _get_workspace_id(auth_client)}/ai/research",
            json={"topic": "remote work productivity", "max_iterations": 2},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "thread_id" in body
        assert body["status"] == "running"

    @pytest.mark.asyncio
    async def test_topic_required(
        self,
        auth_client: AsyncClient,
        research_service_mock: MagicMock,
    ) -> None:
        workspace_id = await _get_workspace_id(auth_client)
        resp = await auth_client.post(
            f"/api/v1/workspaces/{workspace_id}/ai/research",
            json={"max_iterations": 2},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_max_iterations_clamped(
        self,
        auth_client: AsyncClient,
        research_service_mock: MagicMock,
    ) -> None:
        workspace_id = await _get_workspace_id(auth_client)
        resp = await auth_client.post(
            f"/api/v1/workspaces/{workspace_id}/ai/research",
            json={"topic": "test", "max_iterations": 99},
        )
        assert resp.status_code == 422


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_returns_status(
        self,
        auth_client: AsyncClient,
    ) -> None:
        thread_id = _THREAD_ID_STATUS
        svc = _make_research_service(thread_id=thread_id, status="running")
        app.dependency_overrides[get_research_service] = lambda: svc

        try:
            workspace_id = await _get_workspace_id(auth_client)
            resp = await auth_client.get(
                f"/api/v1/workspaces/{workspace_id}/ai/research/{thread_id}"
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["thread_id"] == thread_id
            assert body["status"] == "running"
            assert body["findings_count"] == 3
        finally:
            app.dependency_overrides.pop(get_research_service, None)

    @pytest.mark.asyncio
    async def test_completed_status_includes_synthesis(
        self,
        auth_client: AsyncClient,
    ) -> None:
        thread_id = _THREAD_ID_DONE
        svc = _make_research_service(
            thread_id=thread_id,
            status="completed",
            synthesis="The research report text.",
        )
        app.dependency_overrides[get_research_service] = lambda: svc

        try:
            workspace_id = await _get_workspace_id(auth_client)
            resp = await auth_client.get(
                f"/api/v1/workspaces/{workspace_id}/ai/research/{thread_id}"
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "completed"
            assert body["synthesis"] == "The research report text."
        finally:
            app.dependency_overrides.pop(get_research_service, None)


class TestStreamResearch:
    @pytest.mark.asyncio
    async def test_sse_yields_tokens_then_done(
        self,
        auth_client: AsyncClient,
        research_service_mock: MagicMock,
    ) -> None:
        workspace_id = await _get_workspace_id(auth_client)
        thread_id = _THREAD_ID_STREAM

        async with auth_client.stream(
            "GET",
            f"/api/v1/workspaces/{workspace_id}/ai/research/{thread_id}/stream",
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            chunks = []
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line[6:])

        import json

        events = [json.loads(c) for c in chunks if c]
        token_events = [e for e in events if e.get("type") == "token"]
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(token_events) == 2
        assert token_events[0]["data"] == "Hello "
        assert token_events[1]["data"] == "world"
        assert len(done_events) == 1


class TestReviewResearch:
    @pytest.mark.asyncio
    async def test_approve_returns_resumed(
        self,
        auth_client: AsyncClient,
        research_service_mock: MagicMock,
    ) -> None:
        workspace_id = await _get_workspace_id(auth_client)
        thread_id = _THREAD_ID_REVIEW_A

        resp = await auth_client.post(
            f"/api/v1/workspaces/{workspace_id}/ai/research/{thread_id}/review",
            json={"approved": True, "feedback": None},
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "resumed"}
        research_service_mock.review.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reject_with_feedback(
        self,
        auth_client: AsyncClient,
        research_service_mock: MagicMock,
    ) -> None:
        workspace_id = await _get_workspace_id(auth_client)
        thread_id = _THREAD_ID_REVIEW_B

        resp = await auth_client.post(
            f"/api/v1/workspaces/{workspace_id}/ai/research/{thread_id}/review",
            json={"approved": False, "feedback": "Please add more detail."},
        )
        assert resp.status_code == 200
        call_kwargs = research_service_mock.review.call_args
        assert call_kwargs.kwargs["approved"] is False
        assert call_kwargs.kwargs["feedback"] == "Please add more detail."


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_sixth_request_is_rate_limited(
        self,
        auth_client: AsyncClient,
        research_service_mock: MagicMock,
    ) -> None:
        workspace_id = await _get_workspace_id(auth_client)
        url = f"/api/v1/workspaces/{workspace_id}/ai/research"
        payload = {"topic": "test topic", "max_iterations": 1}

        for _ in range(5):
            resp = await auth_client.post(url, json=payload)
            assert resp.status_code == 202

        resp = await auth_client.post(url, json=payload)
        assert resp.status_code == 429


async def _get_workspace_id(client: AsyncClient) -> uuid.UUID:
    """Create a workspace and return its id."""
    resp = await client.post(
        "/api/v1/workspaces",
        json={"name": "research-test-ws", "description": "test"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]
