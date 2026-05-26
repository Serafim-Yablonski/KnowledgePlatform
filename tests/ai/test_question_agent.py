"""Tests for the PydanticAI question-answering agent using TestModel."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.test import TestModel

from src.ai.agents.question import WorkspaceDeps, agent
from src.domain.ai import Answer, SourceReference
from src.domain.search import SearchResult, SearchResults

_NOT_FOUND = "I could not find information about this in the workspace documents."


def _make_search_response(
    results: list[dict],  # type: ignore[type-arg]
) -> SearchResults:
    items = [
        SearchResult(
            chunk_id=uuid.uuid4(),
            chunk_text=r["chunk_text"],
            document_id=r["document_id"],
            document_title=r.get("title", "Test Doc"),
            score=r.get("score", 0.9),
        )
        for r in results
    ]
    return SearchResults(results=items, query="test", total_results=len(items))


def _make_deps(
    search_results: list[dict] | None = None,  # type: ignore[type-arg]
) -> WorkspaceDeps:
    search_svc = MagicMock()
    search_svc.search = AsyncMock(
        return_value=_make_search_response(search_results or [])
    )
    doc_svc = MagicMock()
    return WorkspaceDeps(
        workspace_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        search_service=search_svc,
        document_service=doc_svc,
    )


def _search_tool_calls(result: object) -> list[ToolCallPart]:
    return [
        part
        for msg in result.all_messages()  # type: ignore[union-attr]
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart) and part.tool_name == "search_documents"
    ]


class TestQuestionAgent:
    @pytest.mark.asyncio
    async def test_search_documents_tool_is_called(self) -> None:
        doc_id = uuid.uuid4()
        deps = _make_deps([{"chunk_text": "relevant text", "document_id": doc_id}])
        output_args = {
            "answer": "The answer is X.",
            "sources": [
                {
                    "document_id": str(doc_id),
                    "document_title": "Test Doc",
                    "chunk_text": "relevant text",
                    "relevance_score": 0.9,
                }
            ],
            "confidence": 0.9,
            "reasoning": "Found directly in documents.",
        }

        with agent.override(
            model=TestModel(
                call_tools=["search_documents"],
                custom_output_args=output_args,
            )
        ):
            result = await agent.run("What is X?", deps=deps)

        assert len(_search_tool_calls(result)) >= 1
        called_workspace_id = deps.search_service.search.call_args.args[0]
        assert called_workspace_id == deps.workspace_id

    @pytest.mark.asyncio
    async def test_returns_structured_answer_response(self) -> None:
        deps = _make_deps([])
        output_args = {
            "answer": "This is the answer.",
            "sources": [],
            "confidence": 0.8,
            "reasoning": "Based on document content.",
        }

        with agent.override(
            model=TestModel(call_tools=[], custom_output_args=output_args)
        ):
            result = await agent.run("What is X?", deps=deps)

        assert isinstance(result.output, Answer)
        assert result.output.answer == "This is the answer."
        assert result.output.confidence == pytest.approx(0.8)
        assert result.output.reasoning == "Based on document content."

    @pytest.mark.asyncio
    async def test_confidence_zero_when_no_results(self) -> None:
        deps = _make_deps([])
        output_args = {
            "answer": _NOT_FOUND,
            "sources": [],
            "confidence": 0.0,
            "reasoning": "No relevant documents were found by the search tool.",
        }

        with agent.override(
            model=TestModel(
                call_tools=["search_documents"],
                custom_output_args=output_args,
            )
        ):
            result = await agent.run("What is an unknown concept?", deps=deps)

        assert result.output.confidence == pytest.approx(0.0)
        assert result.output.sources == []

    @pytest.mark.asyncio
    async def test_source_document_ids_match_search_results(self) -> None:
        doc_id = uuid.uuid4()
        deps = _make_deps(
            [
                {
                    "chunk_text": "specific excerpt",
                    "document_id": doc_id,
                    "title": "Source Doc",
                }
            ]
        )
        output_args = {
            "answer": "Answer derived from Source Doc.",
            "sources": [
                {
                    "document_id": str(doc_id),
                    "document_title": "Source Doc",
                    "chunk_text": "specific excerpt",
                    "relevance_score": 0.95,
                }
            ],
            "confidence": 1.0,
            "reasoning": "Directly stated in Source Doc.",
        }

        with agent.override(
            model=TestModel(
                call_tools=["search_documents"],
                custom_output_args=output_args,
            )
        ):
            result = await agent.run("What does Source Doc say?", deps=deps)

        assert len(result.output.sources) == 1
        assert result.output.sources[0].document_id == doc_id
        assert result.output.sources[0].document_title == "Source Doc"
        called_workspace_id = deps.search_service.search.call_args.args[0]
        assert called_workspace_id == deps.workspace_id

    @pytest.mark.asyncio
    async def test_graceful_empty_search_list(self) -> None:
        deps = _make_deps([])
        output_args = {
            "answer": _NOT_FOUND,
            "sources": [],
            "confidence": 0.0,
            "reasoning": "Search returned no results.",
        }

        with agent.override(
            model=TestModel(call_tools=[], custom_output_args=output_args)
        ):
            result = await agent.run("Something obscure?", deps=deps)

        assert isinstance(result.output, Answer)
        assert result.output.sources == []

    @pytest.mark.asyncio
    async def test_invalid_output_exhausts_retries(self) -> None:
        """PydanticAI retries on bad output and raises UnexpectedModelBehavior."""
        deps = _make_deps([])
        with (
            agent.override(
                model=TestModel(
                    call_tools=[], custom_output_args={"invalid_field": "garbage"}
                )
            ),
            pytest.raises(UnexpectedModelBehavior, match="Exceeded maximum retries"),
        ):
            await agent.run("test", deps=deps)

    @pytest.mark.asyncio
    async def test_sources_are_validated_as_source_references(self) -> None:
        doc_id_a = uuid.uuid4()
        doc_id_b = uuid.uuid4()
        deps = _make_deps(
            [
                {"chunk_text": "chunk A", "document_id": doc_id_a, "title": "Doc A"},
                {"chunk_text": "chunk B", "document_id": doc_id_b, "title": "Doc B"},
            ]
        )
        output_args = {
            "answer": "Answer synthesised from two documents.",
            "sources": [
                {
                    "document_id": str(doc_id_a),
                    "document_title": "Doc A",
                    "chunk_text": "chunk A",
                    "relevance_score": 0.9,
                },
                {
                    "document_id": str(doc_id_b),
                    "document_title": "Doc B",
                    "chunk_text": "chunk B",
                    "relevance_score": 0.8,
                },
            ],
            "confidence": 0.7,
            "reasoning": "Inferred from two documents.",
        }

        with agent.override(
            model=TestModel(
                call_tools=["search_documents"],
                custom_output_args=output_args,
            )
        ):
            result = await agent.run("Synthesise A and B.", deps=deps)

        assert len(result.output.sources) == 2
        assert all(isinstance(s, SourceReference) for s in result.output.sources)
        assert {s.document_id for s in result.output.sources} == {doc_id_a, doc_id_b}
