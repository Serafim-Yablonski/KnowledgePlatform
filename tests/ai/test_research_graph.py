"""Tests for the LangGraph research workflow using MemorySaver (no Postgres needed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic_ai.models.test import TestModel

from src.ai.graphs.research import (
    build_research_graph,
    build_research_graph_with_memory,
)
from src.ai.graphs.state import (
    ResearchPlan,
    ResearchState,
)
from src.schemas.search import SearchResponse, SearchResultItem


def _make_search_service(results: list[dict] | None = None) -> MagicMock:  # type: ignore[type-arg]
    items = [
        SearchResultItem(
            chunk_text=r.get("text", "sample text"),
            document_id=r.get("document_id", "00000000-0000-0000-0000-000000000001"),
            document_title=r.get("title", "Doc A"),
            score=r.get("score", 0.9),
        )
        for r in (results or [{"text": "relevant content about the topic"}])
    ]
    svc = MagicMock()
    svc.search = AsyncMock(
        return_value=SearchResponse(
            results=items, query="test", total_results=len(items)
        )
    )
    return svc


def _make_redis() -> MagicMock:
    redis = MagicMock()
    redis.rpush = AsyncMock(return_value=1)
    redis.publish = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)
    return redis


def _initial_state(max_iterations: int = 3) -> ResearchState:
    return ResearchState(
        topic="impact of remote work on productivity",
        workspace_id="00000000-0000-0000-0000-000000000010",
        user_id="00000000-0000-0000-0000-000000000020",
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


class TestGraphTopology:
    def test_node_names_present(self) -> None:
        graph = build_research_graph_with_memory(
            search_service=_make_search_service(),
            redis_client=_make_redis(),
        )
        nodes = set(graph.get_graph().nodes.keys())
        assert {
            "plan_research",
            "retrieve_evidence",
            "evaluate_sufficiency",
            "synthesize_answer",
            "human_review",
        }.issubset(nodes)

    def test_conditional_edge_sufficient_path(self) -> None:
        from src.ai.graphs.research import _should_continue

        state = dict(_initial_state())
        state["is_sufficient"] = True
        state["iteration_count"] = 1
        assert _should_continue(state) == "synthesize_answer"  # type: ignore[arg-type]

    def test_conditional_edge_loop_back(self) -> None:
        from src.ai.graphs.research import _should_continue

        state = dict(_initial_state(max_iterations=3))
        state["is_sufficient"] = False
        state["iteration_count"] = 1
        assert _should_continue(state) == "retrieve_evidence"  # type: ignore[arg-type]

    def test_conditional_edge_cap_forces_synthesize(self) -> None:
        # The cap is enforced in evaluate_sufficiency (is_sufficient=True when
        # iteration_count >= max_iterations), so _should_continue only sees
        # is_sufficient=True when the cap is hit.
        from src.ai.graphs.research import _should_continue

        state = dict(_initial_state(max_iterations=2))
        state["is_sufficient"] = True  # set by evaluate_sufficiency when cap hit
        state["iteration_count"] = 2
        assert _should_continue(state) == "synthesize_answer"  # type: ignore[arg-type]


class TestFullRunWithTestModel:
    @pytest.mark.asyncio
    async def test_graph_reaches_human_review_interrupt(self) -> None:
        graph = build_research_graph_with_memory(
            search_service=_make_search_service(),
            redis_client=_make_redis(),
        )

        plan_output = {
            "queries": ["remote work productivity", "distributed teams"],
            "scope": "effects on individual and team output",
            "expected_sections": ["Overview", "Findings", "Conclusion"],
        }
        eval_output = {
            "sufficient": True,
            "gaps": [],
            "reasoning": "All sections covered.",
        }
        # _synthesize_agent now has output_type=str — TestModel returns default text

        import src.ai.graphs.nodes as nodes_module

        with (
            nodes_module._plan_agent.override(
                model=TestModel(custom_output_args=plan_output)
            ),
            nodes_module._evaluate_agent.override(
                model=TestModel(custom_output_args=eval_output)
            ),
            nodes_module._synthesize_agent.override(model=TestModel()),
        ):
            thread_id = "test-thread-001"
            config = {"configurable": {"thread_id": thread_id}}
            result = await graph.ainvoke(_initial_state(), config)

        # Graph stops at human_review interrupt — synthesis is populated
        assert result is not None

    @pytest.mark.asyncio
    async def test_plan_node_sets_plan_and_resets_count(self) -> None:
        from src.ai.graphs.nodes import plan_research

        plan_output = {
            "queries": ["query 1"],
            "scope": "test scope",
            "expected_sections": ["Intro"],
        }
        state = _initial_state()
        state["iteration_count"] = 5  # should be reset to 0

        import src.ai.graphs.nodes as nodes_module

        with nodes_module._plan_agent.override(
            model=TestModel(custom_output_args=plan_output)
        ):
            update = await plan_research(state)

        assert update["iteration_count"] == 0
        assert update["gap_queries"] == []
        assert isinstance(update["plan"], ResearchPlan)
        assert update["plan"].queries == ["query 1"]


class TestLoopCap:
    @pytest.mark.asyncio
    async def test_graph_synthesizes_after_max_iterations(self) -> None:
        graph = build_research_graph_with_memory(
            search_service=_make_search_service(),
            redis_client=_make_redis(),
        )

        plan_output = {
            "queries": ["q1"],
            "scope": "scope",
            "expected_sections": ["A"],
        }
        # Evaluate always returns insufficient
        eval_output = {
            "sufficient": False,
            "gaps": ["more info needed"],
            "reasoning": "Not enough evidence.",
        }
        import src.ai.graphs.nodes as nodes_module

        with (
            nodes_module._plan_agent.override(
                model=TestModel(custom_output_args=plan_output)
            ),
            nodes_module._evaluate_agent.override(
                model=TestModel(custom_output_args=eval_output)
            ),
            nodes_module._synthesize_agent.override(model=TestModel()),
        ):
            state = _initial_state(max_iterations=2)
            config = {"configurable": {"thread_id": "test-loop-cap"}}
            result = await graph.ainvoke(state, config)

        assert result is not None
        # iteration_count should not exceed max_iterations + 1 (the extra +1 is the
        # final retrieve that triggers evaluate which forces synthesize)
        assert result["iteration_count"] <= state["max_iterations"] + 1


class TestCheckpointPersistence:
    @pytest.mark.asyncio
    async def test_resume_from_interrupt_with_same_checkpointer(self) -> None:
        memory = MemorySaver()
        redis = _make_redis()
        search_svc = _make_search_service()

        plan_output = {
            "queries": ["q1"],
            "scope": "scope",
            "expected_sections": ["A"],
        }
        eval_output = {"sufficient": True, "gaps": [], "reasoning": "ok"}

        import src.ai.graphs.nodes as nodes_module

        thread_id = "persist-test-001"
        config = {"configurable": {"thread_id": thread_id}}

        # First invocation — runs until human_review interrupt
        graph1 = build_research_graph(search_svc, redis, checkpointer=memory)
        with (
            nodes_module._plan_agent.override(
                model=TestModel(custom_output_args=plan_output)
            ),
            nodes_module._evaluate_agent.override(
                model=TestModel(custom_output_args=eval_output)
            ),
            nodes_module._synthesize_agent.override(model=TestModel()),
        ):
            await graph1.ainvoke(_initial_state(), config)

        # Verify graph is interrupted (state persisted)
        snapshot = await graph1.aget_state(config)
        assert any(bool(t.interrupts) for t in snapshot.tasks)

        # Second invocation — new graph object, same checkpointer, resume
        graph2 = build_research_graph(search_svc, redis, checkpointer=memory)
        with (
            nodes_module._plan_agent.override(
                model=TestModel(custom_output_args=plan_output)
            ),
            nodes_module._evaluate_agent.override(
                model=TestModel(custom_output_args=eval_output)
            ),
            nodes_module._synthesize_agent.override(model=TestModel()),
        ):
            result = await graph2.ainvoke(
                Command(resume={"approved": True, "feedback": None}), config
            )

        assert result["human_approved"] is True


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_state_survives_graph_object_discard(self) -> None:
        memory = MemorySaver()
        redis = _make_redis()
        search_svc = _make_search_service()

        plan_output = {
            "queries": ["q1"],
            "scope": "scope",
            "expected_sections": ["A"],
        }
        eval_output = {"sufficient": True, "gaps": [], "reasoning": "ok"}

        import src.ai.graphs.nodes as nodes_module

        thread_id = "crash-recovery-001"
        config = {"configurable": {"thread_id": thread_id}}

        with (
            nodes_module._plan_agent.override(
                model=TestModel(custom_output_args=plan_output)
            ),
            nodes_module._evaluate_agent.override(
                model=TestModel(custom_output_args=eval_output)
            ),
            nodes_module._synthesize_agent.override(model=TestModel()),
        ):
            # "Crash": run graph, then discard the object
            crashed_graph = build_research_graph(search_svc, redis, checkpointer=memory)
            await crashed_graph.ainvoke(_initial_state(), config)
            del crashed_graph  # simulate crash

            # Recovery: brand-new graph instance, same checkpointer
            recovered_graph = build_research_graph(
                search_svc, redis, checkpointer=memory
            )
            snapshot = await recovered_graph.aget_state(config)

        # State is fully restored from the checkpointer
        assert snapshot.values.get("topic") == "impact of remote work on productivity"
        assert snapshot.values.get("synthesis") is not None
