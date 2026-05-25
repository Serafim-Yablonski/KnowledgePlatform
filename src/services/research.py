from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import structlog

from src.ai.graphs.research import build_research_graph
from src.ai.graphs.state import ResearchState
from src.core.exceptions import ForbiddenError, NotFoundError
from src.schemas.research import (
    ResearchPlanResponse,
    ResearchStatusResponse,
)
from src.services.search import SearchService

logger = structlog.get_logger(__name__)

_running_tasks: dict[str, asyncio.Task[Any]] = {}
_task_errors: dict[str, str] = {}

_STREAM_SENTINEL = "__DONE__"


def _safe_error(exc: BaseException) -> str:
    from pydantic_ai.exceptions import ModelHTTPError  # noqa: PLC0415

    if isinstance(exc, ModelHTTPError):
        return f"LLM service error (HTTP {exc.status_code}). Please retry."
    return f"Research failed: {type(exc).__name__}"


_STREAM_TIMEOUT = 300.0  # 5-minute ceiling for synthesis streaming


class ResearchService:
    def __init__(
        self,
        search_service: SearchService,
        redis_client: Any,
    ) -> None:
        self._search = search_service
        self._redis = redis_client

    async def start_research(
        self,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        topic: str,
        max_iterations: int,
    ) -> str:
        thread_id = str(uuid.uuid4())
        graph = build_research_graph(self._search, self._redis)

        initial_state = ResearchState(
            topic=topic,
            workspace_id=str(workspace_id),
            user_id=str(user_id),
            plan=None,
            findings=[],
            evaluation=None,
            gap_queries=[],
            synthesis=None,
            iteration_count=0,
            max_iterations=max_iterations,
            is_sufficient=False,
            human_approved=False,
            human_feedback=None,
        )
        config = {"configurable": {"thread_id": thread_id}}

        task = asyncio.create_task(
            graph.ainvoke(initial_state, config),
            name=f"research:{thread_id}",
        )
        _running_tasks[thread_id] = task

        def _on_done(t: asyncio.Task[Any]) -> None:
            _running_tasks.pop(thread_id, None)
            if not t.cancelled() and (exc := t.exception()):
                logger.error(
                    "research task failed",
                    thread_id=thread_id,
                    exc_info=exc,
                )
                _task_errors[thread_id] = _safe_error(exc)

        task.add_done_callback(_on_done)

        logger.info("research started", thread_id=thread_id, topic=topic)
        return thread_id

    async def _verify_ownership(self, workspace_id: uuid.UUID, thread_id: str) -> Any:
        """Load checkpoint state and assert it belongs to workspace_id."""
        graph = build_research_graph(self._search, self._redis)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config)

        if not snapshot or not snapshot.values:
            raise NotFoundError(f"Research thread {thread_id!r} not found")

        if snapshot.values.get("workspace_id") != str(workspace_id):
            raise ForbiddenError("Research thread does not belong to this workspace")

        return snapshot

    async def get_status(
        self,
        workspace_id: uuid.UUID,
        thread_id: str,
    ) -> ResearchStatusResponse:
        snapshot = await self._verify_ownership(workspace_id, thread_id)

        values = snapshot.values
        topic: str = values.get("topic", "")
        plan_data = values.get("plan")
        findings: list[object] = values.get("findings", [])
        synthesis: str | None = values.get("synthesis")
        human_approved: bool = values.get("human_approved", False)

        plan_response: ResearchPlanResponse | None = None
        if plan_data is not None:
            plan_response = ResearchPlanResponse(
                queries=plan_data.queries,
                scope=plan_data.scope,
                expected_sections=plan_data.expected_sections,
            )

        has_interrupt = any(bool(task.interrupts) for task in snapshot.tasks)

        if has_interrupt:
            status = "awaiting_review"
        elif snapshot.next == ():
            status = "completed" if synthesis else "failed"
        elif thread_id in _running_tasks and not _running_tasks[thread_id].done():
            status = "running"
        else:
            status = "failed"

        return ResearchStatusResponse(
            thread_id=thread_id,
            status=status,  # type: ignore[arg-type]
            topic=topic,
            plan=plan_response,
            findings_count=len(findings),
            synthesis=synthesis,
            human_approved=human_approved,
            error=_task_errors.get(thread_id),
        )

    async def stream_synthesis(
        self,
        workspace_id: uuid.UUID,
        thread_id: str,
    ) -> AsyncGenerator[str]:
        await self._verify_ownership(workspace_id, thread_id)

        stream_key = f"research:stream:{thread_id}"

        # Subscribe BEFORE lrange to prevent token loss. In the rare overlap
        # window between subscribe() and lrange(), a token may appear in both
        # the replay and the pubsub stream — the client sees a duplicate token,
        # which is preferable to a permanent gap.
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(stream_key)
        try:
            existing: list[bytes] = await self._redis.lrange(stream_key, 0, -1)
            for token_bytes in existing:
                token = (
                    token_bytes.decode()
                    if isinstance(token_bytes, bytes)
                    else token_bytes
                )
                if token == _STREAM_SENTINEL:
                    yield _STREAM_SENTINEL
                    return
                yield token

            try:
                async with asyncio.timeout(_STREAM_TIMEOUT):
                    async for message in pubsub.listen():
                        if message["type"] != "message":
                            continue
                        data = message["data"]
                        token = data.decode() if isinstance(data, bytes) else data
                        if token == _STREAM_SENTINEL:
                            yield _STREAM_SENTINEL
                            return
                        yield token
            except TimeoutError:
                logger.warning(
                    "stream_synthesis timed out",
                    thread_id=thread_id,
                    timeout=_STREAM_TIMEOUT,
                )
        finally:
            await pubsub.unsubscribe(stream_key)
            await pubsub.aclose()

    async def review(
        self,
        workspace_id: uuid.UUID,
        thread_id: str,
        approved: bool,
        feedback: str | None,
    ) -> None:
        from langgraph.types import Command  # noqa: PLC0415

        await self._verify_ownership(workspace_id, thread_id)

        graph = build_research_graph(self._search, self._redis)
        config = {"configurable": {"thread_id": thread_id}}

        await graph.ainvoke(
            Command(resume={"approved": approved, "feedback": feedback}),
            config,
        )
        logger.info(
            "research review submitted",
            thread_id=thread_id,
            approved=approved,
        )
