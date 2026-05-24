from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ResearchStartRequest(BaseModel):
    topic: Annotated[str, Field(min_length=1, max_length=500, strip_whitespace=True)]
    max_iterations: Annotated[int, Field(ge=1, le=5)] = 3


class ResearchStartResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    thread_id: str
    status: str = "running"


class ResearchPlanResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    queries: list[str]
    scope: str
    expected_sections: list[str]


class ResearchStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    thread_id: str
    status: Literal["running", "awaiting_review", "completed", "failed"]
    topic: str
    plan: ResearchPlanResponse | None
    findings_count: int
    synthesis: str | None
    human_approved: bool = False


class ResearchReviewRequest(BaseModel):
    approved: bool
    feedback: Annotated[str, Field(min_length=1, max_length=2000)] | None = None
