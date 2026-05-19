from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.core.config import get_settings

_FAITHFULNESS_DESC = (
    "1.0=fully supported by sources, 0.5=partially supported, "
    "0.0=unsupported/hallucinated"
)
_RELEVANCE_DESC = (
    "Does the answer address the question? 1.0=fully, 0.5=partially, 0.0=not at all"
)


class JudgeScore(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0, description=_FAITHFULNESS_DESC)
    relevance: float = Field(ge=0.0, le=1.0, description=_RELEVANCE_DESC)
    reasoning: str = Field(description="Brief explanation of both scores")


judge_agent: Agent[None, JudgeScore] = Agent(
    get_settings().LLM_STRONG_MODEL,
    output_type=JudgeScore,
    retries=2,
    defer_model_check=True,
    system_prompt=(
        "You are an evaluation judge for a RAG system.\n"
        "Assess the quality of AI-generated answers against source document chunks.\n"
        "\n"
        "Faithfulness (is the answer supported by the sources?):\n"
        "  1.0 = every claim in the answer is directly present in the source chunks\n"
        "  0.5 = most claims are supported but some are lightly extrapolated\n"
        "  0.0 = the answer contains claims not present in the sources\n"
        "\n"
        "Relevance (does the answer address the question?):\n"
        "  1.0 = the answer directly and completely addresses what was asked\n"
        "  0.5 = the answer partially addresses the question\n"
        "  0.0 = the answer does not address the question at all\n"
        "\n"
        "Be strict and objective. Favour lower scores when in doubt."
    ),
)
