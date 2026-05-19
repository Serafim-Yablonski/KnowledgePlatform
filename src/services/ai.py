from __future__ import annotations

import uuid

import logfire
import structlog
from pydantic_ai.messages import ModelResponse, ToolCallPart

from src.ai.agents.question import WorkspaceDeps, agent
from src.core.exceptions import ForbiddenError
from src.domain.roles import PERMISSIONS, WorkspaceRole
from src.schemas.ai import AnswerResponse
from src.services.document import DocumentService
from src.services.search import SearchService

logger = structlog.get_logger(__name__)


class AIService:
    def __init__(
        self,
        search_service: SearchService,
        document_service: DocumentService,
    ) -> None:
        self._search_service = search_service
        self._document_service = document_service

    async def ask(
        self,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        question: str,
        role: WorkspaceRole,
    ) -> AnswerResponse:
        if "read" not in PERMISSIONS[role]:
            raise ForbiddenError("Insufficient permissions")

        deps = WorkspaceDeps(
            workspace_id=workspace_id,
            user_id=user_id,
            search_service=self._search_service,
            document_service=self._document_service,
        )

        with logfire.span(
            "ai_ask",
            workspace_id=str(workspace_id),
            question_length=len(question),
        ) as span:
            result = await agent.run(question, deps=deps)

            tool_calls_count = sum(
                1
                for msg in result.all_messages()
                if isinstance(msg, ModelResponse)
                for part in msg.parts
                if isinstance(part, ToolCallPart)
            )
            usage = result.usage()

            span.set_attribute("answer_length", len(result.output.answer))
            span.set_attribute("confidence", result.output.confidence)
            span.set_attribute("source_count", len(result.output.sources))
            span.set_attribute("tool_calls_count", tool_calls_count)
            span.set_attribute(
                "total_tokens",
                usage.total_tokens if usage.total_tokens is not None else 0,
            )

        logger.info(
            "ai_ask completed",
            workspace_id=str(workspace_id),
            confidence=result.output.confidence,
            source_count=len(result.output.sources),
            tool_calls_count=tool_calls_count,
        )

        return result.output
