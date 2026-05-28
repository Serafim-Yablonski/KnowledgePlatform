from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    queries: list[str] = Field(description="Search queries to execute")
    scope: str = Field(description="What the research should cover")
    expected_sections: list[str] = Field(
        description="Sections the final report should have"
    )


class Finding(BaseModel):
    text: str
    source_document_id: str
    source_document_title: str
    relevance_score: float
    query_that_found_it: str


class EvaluationResult(BaseModel):
    sufficient: bool
    gaps: Annotated[
        list[Annotated[str, Field(max_length=200)]],
        Field(max_length=10, description="Search queries to fill missing evidence"),
    ] = []
    reasoning: str


class ReportSection(BaseModel):
    title: str
    content: str


class Citation(BaseModel):
    document_title: str
    document_id: str


class ResearchReport(BaseModel):
    title: str
    summary: str
    sections: list[ReportSection]
    citations: list[Citation]


class ResearchState(TypedDict):
    topic: str
    workspace_id: str
    user_id: str
    plan: ResearchPlan | None
    findings: Annotated[list[Finding], operator.add]
    evaluation: str | None
    gap_queries: list[str]
    synthesis: str | None
    iteration_count: int
    max_iterations: int
    is_sufficient: bool
    human_approved: bool
    human_feedback: str | None
