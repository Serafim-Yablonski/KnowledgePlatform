from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request

from src.core.dependencies import (
    get_ai_service,
    get_current_user,
    get_current_workspace,
)
from src.core.rate_limit import rate_limit
from src.domain.roles import WorkspaceRole
from src.models.user import User
from src.models.workspace import Workspace
from src.schemas.ai import AnswerResponse, AskRequest
from src.services.ai import AIService

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/ai",
    tags=["ai"],
)

_ask_rate_limit = rate_limit("ai_ask", 20, 60)


@router.post(
    "/ask",
    response_model=AnswerResponse,
    dependencies=[Depends(_ask_rate_limit)],
)
async def ask_question(
    # For routing only — use workspace.id (auth-verified by get_current_workspace).
    workspace_id: uuid.UUID,  # noqa: ARG001
    body: AskRequest,
    request: Request,
    workspace: Workspace = Depends(get_current_workspace),
    user: User = Depends(get_current_user),
    service: AIService = Depends(get_ai_service),
) -> AnswerResponse:
    role: WorkspaceRole = request.state.workspace_role
    return await service.ask(
        workspace_id=workspace.id,
        user_id=user.id,
        question=body.question,
        role=role,
    )
