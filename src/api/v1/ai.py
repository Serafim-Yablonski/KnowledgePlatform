from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.core.config import get_settings
from src.core.dependencies import (
    get_ai_service,
    get_current_user,
    get_current_workspace,
)
from src.core.rate_limit import rate_limit
from src.domain.ai import Answer
from src.domain.roles import WorkspaceRole
from src.models.user import User
from src.models.workspace import Workspace
from src.schemas.ai import AskRequest
from src.services.ai import AIService

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/ai",
    tags=["ai"],
)

_cfg = get_settings()
_ask_rate_limit = rate_limit(
    "ai_ask", _cfg.RATE_LIMIT_AI_ASK_REQUESTS, _cfg.RATE_LIMIT_AI_ASK_WINDOW
)


@router.post(
    "/ask",
    response_model=Answer,
    dependencies=[Depends(_ask_rate_limit)],
)
async def ask_question(
    body: AskRequest,
    request: Request,
    workspace: Workspace = Depends(get_current_workspace),
    user: User = Depends(get_current_user),
    service: AIService = Depends(get_ai_service),
) -> Answer:
    role: WorkspaceRole = request.state.workspace_role
    return await service.ask(
        workspace_id=workspace.id,
        user_id=user.id,
        question=body.question,
        role=role,
    )
