"""Data models for the RAG evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    id: str
    question: str
    expected_answer: str
    expected_source_doc_ids: list[str]
    difficulty: str
    category: str
    tags: list[str]

    @property
    def is_negative(self) -> bool:
        return self.category == "negative"


@dataclass
class EvalCaseResult:
    case_id: str
    question: str
    category: str
    expected_source_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    retrieved_chunks: list[str]
    answer_text: str
    latency_ms: float
    precision_at_k: float
    recall: float
    reciprocal_rank: float
    correctly_rejected: bool
    answer_faithfulness: float = 0.0
    answer_relevance: float = 0.0
    negative_handling: bool = False


@dataclass
class EvalMetrics:
    precision_at_k: float
    recall: float
    mrr: float
    negative_rejection_rate: float
    p95_latency_ms: float
    total_cases: int
    negative_cases: int
    positive_cases: int
    answer_faithfulness: float = 0.0
    answer_relevance: float = 0.0
    negative_handling_rate: float = 0.0


@dataclass
class EvalResults:
    timestamp: str
    git_commit: str
    metrics: EvalMetrics
    cases: list[EvalCaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "metrics": {
                "precision_at_k": self.metrics.precision_at_k,
                "recall": self.metrics.recall,
                "mrr": self.metrics.mrr,
                "negative_rejection_rate": self.metrics.negative_rejection_rate,
                "p95_latency_ms": self.metrics.p95_latency_ms,
                "total_cases": self.metrics.total_cases,
                "negative_cases": self.metrics.negative_cases,
                "positive_cases": self.metrics.positive_cases,
                "answer_faithfulness": self.metrics.answer_faithfulness,
                "answer_relevance": self.metrics.answer_relevance,
                "negative_handling_rate": self.metrics.negative_handling_rate,
            },
            "cases": [
                {
                    "case_id": c.case_id,
                    "question": c.question,
                    "category": c.category,
                    "expected_source_doc_ids": c.expected_source_doc_ids,
                    "retrieved_doc_ids": c.retrieved_doc_ids,
                    "retrieved_chunks": c.retrieved_chunks,
                    "answer_text": c.answer_text,
                    "latency_ms": c.latency_ms,
                    "precision_at_k": c.precision_at_k,
                    "recall": c.recall,
                    "reciprocal_rank": c.reciprocal_rank,
                    "correctly_rejected": c.correctly_rejected,
                    "answer_faithfulness": c.answer_faithfulness,
                    "answer_relevance": c.answer_relevance,
                    "negative_handling": c.negative_handling,
                }
                for c in self.cases
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResults:
        m = data["metrics"]
        metrics = EvalMetrics(
            precision_at_k=m["precision_at_k"],
            recall=m["recall"],
            mrr=m["mrr"],
            negative_rejection_rate=m["negative_rejection_rate"],
            p95_latency_ms=m["p95_latency_ms"],
            total_cases=m["total_cases"],
            negative_cases=m["negative_cases"],
            positive_cases=m["positive_cases"],
            answer_faithfulness=m.get("answer_faithfulness", 0.0),
            answer_relevance=m.get("answer_relevance", 0.0),
            negative_handling_rate=m.get("negative_handling_rate", 0.0),
        )
        cases = [
            EvalCaseResult(
                case_id=c["case_id"],
                question=c["question"],
                category=c["category"],
                expected_source_doc_ids=c["expected_source_doc_ids"],
                retrieved_doc_ids=c["retrieved_doc_ids"],
                retrieved_chunks=c["retrieved_chunks"],
                answer_text=c["answer_text"],
                latency_ms=c["latency_ms"],
                precision_at_k=c["precision_at_k"],
                recall=c["recall"],
                reciprocal_rank=c["reciprocal_rank"],
                correctly_rejected=c["correctly_rejected"],
                answer_faithfulness=c.get("answer_faithfulness", 0.0),
                answer_relevance=c.get("answer_relevance", 0.0),
                negative_handling=c.get("negative_handling", False),
            )
            for c in data.get("cases", [])
        ]
        return cls(
            timestamp=data["timestamp"],
            git_commit=data["git_commit"],
            metrics=metrics,
            cases=cases,
        )
