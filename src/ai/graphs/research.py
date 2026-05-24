from __future__ import annotations

from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.ai.graphs.nodes import (
    evaluate_sufficiency,
    make_retrieve_node,
    make_synthesize_node,
    plan_research,
)
from src.ai.graphs.state import ResearchState
from src.services.search import SearchService


def _should_continue(state: ResearchState) -> str:
    # evaluate_sufficiency already enforces the iteration cap by setting
    # is_sufficient=True when iteration_count >= max_iterations.
    if state["is_sufficient"]:
        return "synthesize_answer"
    return "retrieve_evidence"


async def human_review(state: ResearchState) -> dict[str, object]:
    review_result: dict[str, Any] = interrupt(
        {"requires_review": True, "synthesis": state["synthesis"]}
    )
    approved: bool = review_result.get("approved", False)
    feedback: str | None = review_result.get("feedback")
    return {"human_approved": approved, "human_feedback": feedback}


def build_research_graph(
    search_service: SearchService,
    redis_client: Any,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    retrieve_node = make_retrieve_node(search_service)
    synthesize_node = make_synthesize_node(redis_client)

    builder: StateGraph[ResearchState] = StateGraph(ResearchState)

    builder.add_node("plan_research", plan_research)
    # cast: LangGraph's _Node type is incompatible with Callable[..., Awaitable]
    builder.add_node("retrieve_evidence", cast(Any, retrieve_node))
    builder.add_node("evaluate_sufficiency", evaluate_sufficiency)
    builder.add_node("synthesize_answer", cast(Any, synthesize_node))
    builder.add_node("human_review", human_review)

    builder.add_edge(START, "plan_research")
    builder.add_edge("plan_research", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "evaluate_sufficiency")
    builder.add_conditional_edges(
        "evaluate_sufficiency",
        _should_continue,
        {
            "synthesize_answer": "synthesize_answer",
            "retrieve_evidence": "retrieve_evidence",
        },
    )
    builder.add_edge("synthesize_answer", "human_review")
    builder.add_edge("human_review", END)

    effective_checkpointer = checkpointer
    if effective_checkpointer is None:
        from src.ai.graphs.checkpointer import get_checkpointer  # noqa: PLC0415

        effective_checkpointer = get_checkpointer()

    return builder.compile(checkpointer=effective_checkpointer)


def build_research_graph_with_memory(
    search_service: SearchService,
    redis_client: Any,
) -> Any:
    """Build a graph with an in-memory checkpointer — for tests only."""
    return build_research_graph(
        search_service=search_service,
        redis_client=redis_client,
        checkpointer=MemorySaver(),
    )
