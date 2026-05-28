from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ResearchPlan:
    queries: list[str]
    scope: str
    expected_sections: list[str] = field(default_factory=list)


@dataclass
class ResearchStatus:
    thread_id: str
    status: Literal["running", "awaiting_review", "completed", "failed"]
    topic: str
    plan: ResearchPlan | None
    findings_count: int
    synthesis: str | None
    human_approved: bool = False
    error: str | None = None
