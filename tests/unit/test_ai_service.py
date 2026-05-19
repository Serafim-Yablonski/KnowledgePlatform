"""Unit tests for AIService."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import ForbiddenError
from src.domain.roles import WorkspaceRole
from src.schemas.ai import AnswerResponse
from src.services.ai import AIService


def _make_answer_response(confidence: float = 0.9) -> AnswerResponse:
    return AnswerResponse(
        answer="The answer is X.",
        sources=[],
        confidence=confidence,
        reasoning="Found in documents.",
    )


def _make_agent_result(output: AnswerResponse) -> MagicMock:
    mock_result = MagicMock()
    mock_result.output = output
    mock_result.all_messages.return_value = []
    mock_result.usage.return_value = MagicMock(total_tokens=42)
    return mock_result


@contextmanager
def _fake_span(*args: Any, **kwargs: Any):  # type: ignore[misc]
    span = MagicMock()
    span.set_attribute = MagicMock()
    yield span


def _make_service() -> tuple[AIService, MagicMock, MagicMock]:
    search_svc = MagicMock()
    doc_svc = MagicMock()
    return (
        AIService(search_service=search_svc, document_service=doc_svc),
        search_svc,
        doc_svc,
    )


class TestAIService:
    @pytest.mark.asyncio
    async def test_ask_returns_answer_response(self) -> None:
        service, _, _ = _make_service()
        expected = _make_answer_response()

        with (
            patch("src.services.ai.agent") as mock_agent,
            patch("src.services.ai.logfire") as mock_logfire,
        ):
            mock_agent.run = AsyncMock(return_value=_make_agent_result(expected))
            mock_logfire.span = _fake_span
            result = await service.ask(
                workspace_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                question="What is X?",
                role=WorkspaceRole.MEMBER,
            )

        assert result is expected

    @pytest.mark.asyncio
    async def test_ask_passes_correct_workspace_deps(self) -> None:
        service, search_svc, doc_svc = _make_service()
        workspace_id = uuid.uuid4()
        user_id = uuid.uuid4()

        with (
            patch("src.services.ai.agent") as mock_agent,
            patch("src.services.ai.logfire") as mock_logfire,
        ):
            mock_agent.run = AsyncMock(
                return_value=_make_agent_result(_make_answer_response())
            )
            mock_logfire.span = _fake_span
            await service.ask(
                workspace_id=workspace_id,
                user_id=user_id,
                question="test question",
                role=WorkspaceRole.MEMBER,
            )
            deps = mock_agent.run.call_args.kwargs["deps"]

        assert deps.workspace_id == workspace_id
        assert deps.user_id == user_id
        assert deps.search_service is search_svc
        assert deps.document_service is doc_svc

    @pytest.mark.asyncio
    async def test_ask_zero_confidence_passes_through(self) -> None:
        service, _, _ = _make_service()
        low_conf = _make_answer_response(confidence=0.0)

        with (
            patch("src.services.ai.agent") as mock_agent,
            patch("src.services.ai.logfire") as mock_logfire,
        ):
            mock_agent.run = AsyncMock(return_value=_make_agent_result(low_conf))
            mock_logfire.span = _fake_span
            result = await service.ask(
                workspace_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                question="Something unknown?",
                role=WorkspaceRole.VIEWER,
            )

        assert result.confidence == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_ask_passes_question_text_to_agent(self) -> None:
        service, _, _ = _make_service()
        question = "What is the deployment process?"

        with (
            patch("src.services.ai.agent") as mock_agent,
            patch("src.services.ai.logfire") as mock_logfire,
        ):
            mock_agent.run = AsyncMock(
                return_value=_make_agent_result(_make_answer_response())
            )
            mock_logfire.span = _fake_span
            await service.ask(
                workspace_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                question=question,
                role=WorkspaceRole.MEMBER,
            )
            positional_args = mock_agent.run.call_args.args

        assert positional_args[0] == question

    @pytest.mark.asyncio
    async def test_ask_raises_forbidden_when_role_lacks_read(self) -> None:
        """Permission check fires before the agent when PERMISSIONS has no 'read'."""
        service, _, _ = _make_service()
        # All real roles have 'read', so inject a stripped-down permissions map.
        no_read_perms = {role: set() for role in WorkspaceRole}

        with (
            patch("src.services.ai.agent") as mock_agent,
            patch("src.services.ai.PERMISSIONS", no_read_perms),
        ):
            mock_agent.run = AsyncMock()
            with pytest.raises(ForbiddenError):
                await service.ask(
                    workspace_id=uuid.uuid4(),
                    user_id=uuid.uuid4(),
                    question="test",
                    role=WorkspaceRole.VIEWER,
                )
            mock_agent.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_viewer_role_is_allowed(self) -> None:
        """VIEWER has 'read' and must be permitted."""
        service, _, _ = _make_service()
        expected = _make_answer_response()

        with (
            patch("src.services.ai.agent") as mock_agent,
            patch("src.services.ai.logfire") as mock_logfire,
        ):
            mock_agent.run = AsyncMock(return_value=_make_agent_result(expected))
            mock_logfire.span = _fake_span
            result = await service.ask(
                workspace_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                question="Can I read this?",
                role=WorkspaceRole.VIEWER,
            )

        assert result is expected
