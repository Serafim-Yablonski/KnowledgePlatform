"""Unit tests for ResearchService using mocked LangGraph graph."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import ForbiddenError, NotFoundError
from src.services.research import ResearchService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WS_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


def _make_service() -> ResearchService:
    search = MagicMock()
    redis = MagicMock()
    return ResearchService(search_service=search, redis_client=redis)


def _make_snapshot(
    *,
    workspace_id: str = str(_WS_ID),
    user_id: str = str(_USER_ID),
    synthesis: str | None = None,
    next_nodes: tuple[str, ...] = (),
    has_interrupt: bool = False,
) -> MagicMock:
    snap = MagicMock()
    snap.values = {
        "workspace_id": workspace_id,
        "user_id": user_id,
        "topic": "test topic",
        "plan": None,
        "findings": [],
        "synthesis": synthesis,
        "human_approved": False,
    }
    snap.next = next_nodes
    task_mock = MagicMock()
    task_mock.interrupts = ["interrupt"] if has_interrupt else []
    snap.tasks = [task_mock]
    return snap


# ---------------------------------------------------------------------------
# _verify_ownership
# ---------------------------------------------------------------------------


async def test_verify_ownership_snapshot_none_raises_not_found() -> None:
    service = _make_service()
    mock_graph = MagicMock()
    mock_graph.aget_state = AsyncMock(return_value=None)

    with (
        patch("src.services.research.build_research_graph", return_value=mock_graph),
        pytest.raises(NotFoundError),
    ):
        await service._verify_ownership(_WS_ID, _USER_ID, "thread-1")


async def test_verify_ownership_empty_values_raises_not_found() -> None:
    service = _make_service()
    snap = MagicMock()
    snap.values = {}
    mock_graph = MagicMock()
    mock_graph.aget_state = AsyncMock(return_value=snap)

    with (
        patch("src.services.research.build_research_graph", return_value=mock_graph),
        pytest.raises(NotFoundError),
    ):
        await service._verify_ownership(_WS_ID, _USER_ID, "thread-1")


async def test_verify_ownership_workspace_mismatch_raises_not_found() -> None:
    service = _make_service()
    snap = MagicMock()
    snap.values = {"workspace_id": "other-ws", "user_id": str(_USER_ID)}
    mock_graph = MagicMock()
    mock_graph.aget_state = AsyncMock(return_value=snap)

    with (
        patch("src.services.research.build_research_graph", return_value=mock_graph),
        pytest.raises(NotFoundError),
    ):
        await service._verify_ownership(_WS_ID, _USER_ID, "thread-1")


async def test_verify_ownership_success_returns_snapshot() -> None:
    service = _make_service()
    snap = MagicMock()
    snap.values = {"workspace_id": str(_WS_ID), "user_id": str(_USER_ID)}
    mock_graph = MagicMock()
    mock_graph.aget_state = AsyncMock(return_value=snap)

    with patch("src.services.research.build_research_graph", return_value=mock_graph):
        result = await service._verify_ownership(_WS_ID, _USER_ID, "thread-ok")

    assert result is snap


async def test_verify_ownership_user_mismatch_raises_forbidden() -> None:
    service = _make_service()
    snap = MagicMock()
    snap.values = {"workspace_id": str(_WS_ID), "user_id": str(uuid.uuid4())}
    mock_graph = MagicMock()
    mock_graph.aget_state = AsyncMock(return_value=snap)

    with (
        patch("src.services.research.build_research_graph", return_value=mock_graph),
        pytest.raises(ForbiddenError),
    ):
        await service._verify_ownership(_WS_ID, _USER_ID, "thread-1")


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


async def test_get_status_with_plan() -> None:
    service = _make_service()
    plan_data = MagicMock()
    plan_data.queries = ["q1", "q2"]
    plan_data.scope = "test scope"
    plan_data.expected_sections = ["A", "B"]

    snap = MagicMock()
    snap.values = {
        "workspace_id": str(_WS_ID),
        "user_id": str(_USER_ID),
        "topic": "test topic",
        "plan": plan_data,
        "findings": [],
        "synthesis": None,
        "human_approved": False,
    }
    snap.next = ()
    snap.tasks = []

    with patch.object(service, "_verify_ownership", new=AsyncMock(return_value=snap)):
        result = await service.get_status(_WS_ID, _USER_ID, "t-plan")

    assert result.plan is not None
    assert result.plan.queries == ["q1", "q2"]


async def test_get_status_awaiting_review() -> None:
    service = _make_service()
    snap = _make_snapshot(has_interrupt=True)

    with patch.object(service, "_verify_ownership", new=AsyncMock(return_value=snap)):
        result = await service.get_status(_WS_ID, _USER_ID, "t1")

    assert result.status == "awaiting_review"


async def test_get_status_completed() -> None:
    service = _make_service()
    snap = _make_snapshot(synthesis="Final report.", next_nodes=())

    with patch.object(service, "_verify_ownership", new=AsyncMock(return_value=snap)):
        result = await service.get_status(_WS_ID, _USER_ID, "t2")

    assert result.status == "completed"
    assert result.synthesis == "Final report."


async def test_get_status_failed_no_synthesis() -> None:
    service = _make_service()
    snap = _make_snapshot(synthesis=None, next_nodes=())

    with patch.object(service, "_verify_ownership", new=AsyncMock(return_value=snap)):
        result = await service.get_status(_WS_ID, _USER_ID, "t3")

    assert result.status == "failed"


async def test_get_status_running() -> None:
    service = _make_service()
    thread_id = f"running-{uuid.uuid4()}"
    snap = _make_snapshot(next_nodes=("some_node",))

    with patch.object(service, "_verify_ownership", new=AsyncMock(return_value=snap)):
        result = await service.get_status(_WS_ID, _USER_ID, thread_id)

    assert result.status == "running"


# ---------------------------------------------------------------------------
# start_research
# ---------------------------------------------------------------------------


async def test_start_research_task_failure_logs_error() -> None:
    """Failed research tasks are logged; no longer stored in process memory."""
    service = _make_service()
    mock_graph = MagicMock()

    async def _failing_ainvoke(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("graph failed")

    mock_graph.ainvoke = _failing_ainvoke

    with patch("src.services.research.build_research_graph", return_value=mock_graph):
        thread_id = await service.start_research(
            workspace_id=_WS_ID,
            user_id=_USER_ID,
            topic="fail topic",
            max_iterations=1,
        )

    # Let the background task complete.
    await asyncio.sleep(0.05)

    # thread_id is a valid UUID string; error is logged but not in-memory tracked.
    assert isinstance(thread_id, str)
    assert len(thread_id) == 36


async def test_start_research_returns_thread_id() -> None:
    service = _make_service()
    mock_graph = MagicMock()

    async def _fake_ainvoke(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    mock_graph.ainvoke = _fake_ainvoke

    with patch("src.services.research.build_research_graph", return_value=mock_graph):
        thread_id = await service.start_research(
            workspace_id=_WS_ID,
            user_id=_USER_ID,
            topic="AI in healthcare",
            max_iterations=2,
        )

    assert isinstance(thread_id, str)
    assert len(thread_id) == 36

    # Let the background task complete before the event loop tears down.
    await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


async def test_review_calls_graph_ainvoke() -> None:
    service = _make_service()
    snap = _make_snapshot()
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={})

    with (
        patch.object(service, "_verify_ownership", new=AsyncMock(return_value=snap)),
        patch("src.services.research.build_research_graph", return_value=mock_graph),
    ):
        await service.review(_WS_ID, _USER_ID, "t-review", approved=True, feedback=None)

    mock_graph.ainvoke.assert_awaited_once()


# ---------------------------------------------------------------------------
# stream_synthesis
# ---------------------------------------------------------------------------


async def test_stream_synthesis_from_pubsub_yields_tokens() -> None:
    """Covers the pubsub listen path when lrange returns no cached tokens."""
    service = _make_service()
    snap = _make_snapshot()

    pubsub = MagicMock()
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.aclose = AsyncMock()

    async def _listen_gen() -> Any:
        yield {"type": "subscribe", "data": None}  # skipped by continue
        yield {"type": "message", "data": b"token1"}
        yield {"type": "message", "data": b"__DONE__"}

    pubsub.listen = _listen_gen
    service._redis.pubsub = MagicMock(return_value=pubsub)
    service._redis.lrange = AsyncMock(return_value=[])

    tokens: list[str] = []
    with patch.object(service, "_verify_ownership", new=AsyncMock(return_value=snap)):
        async for token in service.stream_synthesis(_WS_ID, _USER_ID, "t-pubsub"):
            tokens.append(token)

    assert tokens == ["token1", "__DONE__"]


async def test_stream_synthesis_replays_lrange_tokens() -> None:
    service = _make_service()
    snap = _make_snapshot()

    pubsub = MagicMock()
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.aclose = AsyncMock()

    async def _empty_listen() -> Any:  # pragma: no cover
        return
        yield  # make it an async generator that yields nothing

    pubsub.listen = _empty_listen
    service._redis.pubsub = MagicMock(return_value=pubsub)
    service._redis.lrange = AsyncMock(return_value=[b"hello", b"world", b"__DONE__"])

    tokens: list[str] = []
    with patch.object(service, "_verify_ownership", new=AsyncMock(return_value=snap)):
        async for token in service.stream_synthesis(_WS_ID, _USER_ID, "t-stream"):
            tokens.append(token)

    assert tokens == ["hello", "world", "__DONE__"]


async def test_stream_synthesis_timeout() -> None:
    service = _make_service()
    snap = _make_snapshot()

    pubsub = MagicMock()
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.aclose = AsyncMock()

    async def _hanging_listen() -> Any:
        await asyncio.sleep(9999)
        yield {"type": "message", "data": b"token"}

    pubsub.listen = _hanging_listen
    service._redis.pubsub = MagicMock(return_value=pubsub)
    service._redis.lrange = AsyncMock(return_value=[])

    tokens: list[str] = []
    with (
        patch.object(service, "_verify_ownership", new=AsyncMock(return_value=snap)),
        patch("src.services.research._STREAM_TIMEOUT", 0.01),
    ):
        async for token in service.stream_synthesis(_WS_ID, _USER_ID, "t-timeout"):
            tokens.append(token)

    assert tokens == []
