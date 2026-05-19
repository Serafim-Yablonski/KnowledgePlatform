"""Pure metric computation functions for RAG evaluation.

All functions are stateless and take only primitive inputs so they can be
unit-tested without any database or service dependencies.
"""

from __future__ import annotations

import statistics


def precision_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int = 5,
) -> float:
    """Fraction of the top-k retrieved docs that are relevant."""
    if not retrieved or not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(top_k)


def recall(retrieved: list[str], relevant: list[str]) -> float:
    """Fraction of relevant docs that appear anywhere in the retrieved list."""
    if not relevant:
        return 1.0
    if not retrieved:
        return 0.0
    retrieved_set = set(retrieved)
    hits = sum(1 for doc_id in relevant if doc_id in retrieved_set)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1 / rank of the first relevant result. 0.0 if no relevant doc is found."""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def p95_latency(latencies_ms: list[float]) -> float:
    """95th percentile latency in milliseconds."""
    if not latencies_ms:
        return 0.0
    sorted_latencies = sorted(latencies_ms)
    idx = int(len(sorted_latencies) * 0.95)
    idx = min(idx, len(sorted_latencies) - 1)
    return sorted_latencies[idx]


def aggregate_precision(precisions: list[float]) -> float:
    return statistics.mean(precisions) if precisions else 0.0


def aggregate_recall(recalls: list[float]) -> float:
    return statistics.mean(recalls) if recalls else 0.0


def mean_reciprocal_rank(reciprocal_ranks: list[float]) -> float:
    return statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def negative_rejection_rate(
    correctly_rejected: list[bool],
) -> float:
    """For negative cases, fraction where the system correctly returned no results."""
    if not correctly_rejected:
        return 1.0
    return sum(correctly_rejected) / len(correctly_rejected)
