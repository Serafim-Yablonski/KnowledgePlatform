from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.core.dependencies import (
    get_current_user,
    get_current_workspace,
    get_research_service,
)
from src.core.rate_limit import rate_limit
from src.models.user import User
from src.models.workspace import Workspace
from src.schemas.research import (
    ResearchReviewRequest,
    ResearchStartRequest,
    ResearchStartResponse,
    ResearchStatusResponse,
)
from src.services.research import ResearchService

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/ai/research",
    tags=["research"],
)

_start_rate_limit = rate_limit("ai_research_start", 5, 60)
_stream_rate_limit = rate_limit("ai_research_stream", 30, 60)
_review_rate_limit = rate_limit("ai_research_review", 10, 60)


@router.post(
    "",
    response_model=ResearchStartResponse,
    status_code=202,
    dependencies=[Depends(_start_rate_limit)],
)
async def start_research(
    workspace_id: uuid.UUID,  # noqa: ARG001
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


@router.get("/{thread_id}", response_model=ResearchStatusResponse)
async def get_research_status(
    workspace_id: uuid.UUID,  # noqa: ARG001
    thread_id: uuid.UUID,
    workspace: Workspace = Depends(get_current_workspace),
    service: ResearchService = Depends(get_research_service),
) -> ResearchStatusResponse:
    return await service.get_status(workspace_id=workspace.id, thread_id=str(thread_id))


@router.get(
    "/{thread_id}/stream",
    dependencies=[Depends(_stream_rate_limit)],
)
async def stream_research(
    workspace_id: uuid.UUID,  # noqa: ARG001
    thread_id: uuid.UUID,
    workspace: Workspace = Depends(get_current_workspace),
    service: ResearchService = Depends(get_research_service),
) -> StreamingResponse:
    async def _generate() -> AsyncGenerator[str]:
        async for token in service.stream_synthesis(
            workspace_id=workspace.id, thread_id=str(thread_id)
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
    workspace_id: uuid.UUID,  # noqa: ARG001
    thread_id: uuid.UUID,
    body: ResearchReviewRequest,
    workspace: Workspace = Depends(get_current_workspace),
    service: ResearchService = Depends(get_research_service),
) -> dict[str, Any]:
    await service.review(
        workspace_id=workspace.id,
        thread_id=str(thread_id),
        approved=body.approved,
        feedback=body.feedback,
    )
    return {"status": "resumed"}
