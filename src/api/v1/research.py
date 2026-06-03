from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.core.config import get_settings
from src.core.dependencies import (
    get_current_user,
    get_current_workspace,
    get_research_service,
)
from src.core.rate_limit import rate_limit
from src.models.user import User
from src.models.workspace import Workspace
from src.schemas.research import (
    ResearchPlanResponse,
    ResearchReviewRequest,
    ResearchStartRequest,
    ResearchStartResponse,
    ResearchStatusResponse,
)
from src.services.research import ResearchService

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/ai/research",
    tags=["research"],
)

_cfg = get_settings()
_start_rate_limit = rate_limit(
    "ai_research_start",
    _cfg.RATE_LIMIT_RESEARCH_START_REQUESTS,
    _cfg.RATE_LIMIT_RESEARCH_START_WINDOW,
)
_status_rate_limit = rate_limit(
    "ai_research_status",
    _cfg.RATE_LIMIT_RESEARCH_STATUS_REQUESTS,
    _cfg.RATE_LIMIT_RESEARCH_STATUS_WINDOW,
)
_stream_rate_limit = rate_limit(
    "ai_research_stream",
    _cfg.RATE_LIMIT_RESEARCH_STREAM_REQUESTS,
    _cfg.RATE_LIMIT_RESEARCH_STREAM_WINDOW,
)
_review_rate_limit = rate_limit(
    "ai_research_review",
    _cfg.RATE_LIMIT_RESEARCH_REVIEW_REQUESTS,
    _cfg.RATE_LIMIT_RESEARCH_REVIEW_WINDOW,
)


@router.post(
    "",
    response_model=ResearchStartResponse,
    status_code=202,
    dependencies=[Depends(_start_rate_limit)],
)
async def start_research(
    body: ResearchStartRequest,
    workspace: Workspace = Depends(get_current_workspace),
    user: User = Depends(get_current_user),
    service: ResearchService = Depends(get_research_service),
) -> ResearchStartResponse:
    thread_id = await service.start_research(
        workspace_id=workspace.id,
        user_id=user.id,
        topic=body.topic,
        max_iterations=body.max_iterations,
    )
    return ResearchStartResponse(thread_id=thread_id)


@router.get(
    "/{thread_id}",
    response_model=ResearchStatusResponse,
    dependencies=[Depends(_status_rate_limit)],
)
async def get_research_status(
    thread_id: uuid.UUID,
    workspace: Workspace = Depends(get_current_workspace),
    user: User = Depends(get_current_user),
    service: ResearchService = Depends(get_research_service),
) -> ResearchStatusResponse:
    status = await service.get_status(
        workspace_id=workspace.id, user_id=user.id, thread_id=str(thread_id)
    )
    plan = None
    if status.plan is not None:
        plan = ResearchPlanResponse(
            queries=status.plan.queries,
            scope=status.plan.scope,
            expected_sections=status.plan.expected_sections,
        )
    return ResearchStatusResponse(
        thread_id=status.thread_id,
        status=status.status,
        topic=status.topic,
        plan=plan,
        findings_count=status.findings_count,
        synthesis=status.synthesis,
        human_approved=status.human_approved,
        error=status.error,
    )


@router.get(
    "/{thread_id}/stream",
    dependencies=[Depends(_stream_rate_limit)],
)
async def stream_research(
    thread_id: uuid.UUID,
    workspace: Workspace = Depends(get_current_workspace),
    user: User = Depends(get_current_user),
    service: ResearchService = Depends(get_research_service),
) -> StreamingResponse:
    logger.info(
        "stream_research_started",
        user_id=str(user.id),
        workspace_id=str(workspace.id),
        thread_id=str(thread_id),
    )

    async def _generate() -> AsyncGenerator[str]:
        async for token in service.stream_synthesis(
            workspace_id=workspace.id, user_id=user.id, thread_id=str(thread_id)
        ):
            if token == "__DONE__":
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/{thread_id}/review",
    dependencies=[Depends(_review_rate_limit)],
)
async def review_research(
    thread_id: uuid.UUID,
    body: ResearchReviewRequest,
    workspace: Workspace = Depends(get_current_workspace),
    user: User = Depends(get_current_user),
    service: ResearchService = Depends(get_research_service),
) -> dict[str, Any]:
    await service.review(
        workspace_id=workspace.id,
        user_id=user.id,
        thread_id=str(thread_id),
        approved=body.approved,
        feedback=body.feedback,
    )
    return {"status": "resumed"}
