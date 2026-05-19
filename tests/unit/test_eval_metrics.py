"""Unit tests for RAG evaluation metric functions."""

from __future__ import annotations

import pytest

from src.ai.eval.metrics import (
    aggregate_precision,
    aggregate_recall,
    mean_reciprocal_rank,
    negative_rejection_rate,
    p95_latency,
    precision_at_k,
    recall,
    reciprocal_rank,
)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_none_relevant(self) -> None:
        assert precision_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_partial_relevant(self) -> None:
        result = precision_at_k(["a", "x", "b", "y"], ["a", "b"], k=4)
        assert result == pytest.approx(0.5)

    def test_k_truncates_retrieved(self) -> None:
        # Only the top-2 are considered; "b" at position 3 is excluded
        result = precision_at_k(["a", "x", "b"], ["a", "b"], k=2)
        assert result == pytest.approx(0.5)

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], ["a", "b"], k=5) == 0.0

    def test_empty_relevant(self) -> None:
        assert precision_at_k(["a", "b"], [], k=5) == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        result = precision_at_k(["a", "b"], ["a", "b", "c"], k=10)
        assert result == pytest.approx(1.0)

    def test_order_does_not_matter_for_top_k(self) -> None:
        # Both are in top-3 → precision = 2/3
        result = precision_at_k(["a", "x", "b"], ["a", "b"], k=3)
        assert result == pytest.approx(2 / 3)


class TestRecall:
    def test_all_found(self) -> None:
        assert recall(["a", "b", "c"], ["a", "b"]) == 1.0

    def test_none_found(self) -> None:
        assert recall(["x", "y"], ["a", "b"]) == 0.0

    def test_partial(self) -> None:
        assert recall(["a", "x"], ["a", "b"]) == pytest.approx(0.5)

    def test_empty_retrieved(self) -> None:
        assert recall([], ["a", "b"]) == 0.0

    def test_empty_relevant_returns_one(self) -> None:
        # No expected docs → trivially satisfied
        assert recall(["a", "b"], []) == 1.0

    def test_both_empty(self) -> None:
        assert recall([], []) == 1.0


class TestReciprocalRank:
    def test_first_result_relevant(self) -> None:
        assert reciprocal_rank(["a", "b", "c"], ["a"]) == pytest.approx(1.0)

    def test_second_result_relevant(self) -> None:
        assert reciprocal_rank(["x", "a", "c"], ["a"]) == pytest.approx(0.5)

    def test_third_result_relevant(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_no_relevant_found(self) -> None:
        assert reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0

    def test_empty_retrieved(self) -> None:
        assert reciprocal_rank([], ["a"]) == 0.0

    def test_multiple_relevant_uses_first_hit(self) -> None:
        # "b" is at rank 2 but "a" is at rank 1 — MRR uses rank 1
        assert reciprocal_rank(["a", "b"], ["b", "a"]) == pytest.approx(1.0)


class TestP95Latency:
    def test_single_value(self) -> None:
        assert p95_latency([100.0]) == pytest.approx(100.0)

    def test_sorted_list(self) -> None:
        latencies = [float(i) for i in range(1, 21)]  # 1..20
        # 95th percentile index = int(20 * 0.95) = 19 → value 20
        assert p95_latency(latencies) == pytest.approx(20.0)

    def test_unsorted_input(self) -> None:
        latencies = [50.0, 10.0, 90.0, 30.0, 70.0]
        # sorted: [10, 30, 50, 70, 90], p95 index = int(5*0.95) = 4 → 90
        assert p95_latency(latencies) == pytest.approx(90.0)

    def test_empty_returns_zero(self) -> None:
        assert p95_latency([]) == 0.0


class TestAggregates:
    def test_aggregate_precision_mean(self) -> None:
        assert aggregate_precision([0.5, 1.0, 0.0]) == pytest.approx(0.5)

    def test_aggregate_precision_empty(self) -> None:
        assert aggregate_precision([]) == 0.0

    def test_aggregate_recall_mean(self) -> None:
        assert aggregate_recall([0.0, 1.0]) == pytest.approx(0.5)

    def test_mrr_mean(self) -> None:
        assert mean_reciprocal_rank([1.0, 0.5]) == pytest.approx(0.75)

    def test_mrr_empty(self) -> None:
        assert mean_reciprocal_rank([]) == 0.0


class TestNegativeRejectionRate:
    def test_all_correctly_rejected(self) -> None:
        assert negative_rejection_rate([True, True, True]) == 1.0

    def test_none_correctly_rejected(self) -> None:
        assert negative_rejection_rate([False, False]) == 0.0

    def test_partial(self) -> None:
        assert negative_rejection_rate([True, False, True, True]) == pytest.approx(0.75)

    def test_empty_returns_one(self) -> None:
        # No negative cases → vacuously 100% rejection rate
        assert negative_rejection_rate([]) == 1.0


class TestCompareRegression:
    """Verify that the compare module detects regressions at the 5% threshold."""

    def test_regression_detected_above_threshold(self) -> None:
        from src.ai.eval.compare import compare_metrics
        from src.ai.eval.models import EvalMetrics, EvalResults

        def _results(precision: float) -> EvalResults:
            m = EvalMetrics(
                precision_at_k=precision,
                recall=0.8,
                mrr=0.7,
                negative_rejection_rate=1.0,
                p95_latency_ms=100.0,
                total_cases=10,
                negative_cases=2,
                positive_cases=8,
            )
            return EvalResults(
                timestamp="2024-01-01T00:00:00Z", git_commit="abc", metrics=m
            )

        baseline = _results(0.8)
        # Drop of 0.051 > 0.05 threshold → should flag
        current = _results(0.749)
        report = compare_metrics(baseline, current)
        precision_comparison = next(
            c for c in report.metric_comparisons if c.name == "precision_at_k"
        )
        assert precision_comparison.regressed is True
        assert report.has_regression is True

    def test_no_regression_at_threshold(self) -> None:
        from src.ai.eval.compare import compare_metrics
        from src.ai.eval.models import EvalMetrics, EvalResults

        def _results(precision: float) -> EvalResults:
            m = EvalMetrics(
                precision_at_k=precision,
                recall=0.8,
                mrr=0.7,
                negative_rejection_rate=1.0,
                p95_latency_ms=100.0,
                total_cases=10,
                negative_cases=2,
                positive_cases=8,
            )
            return EvalResults(
                timestamp="2024-01-01T00:00:00Z", git_commit="abc", metrics=m
            )

        baseline = _results(0.8)
        # Drop of 0.04 < 0.05 threshold → should NOT trigger
        current = _results(0.76)
        report = compare_metrics(baseline, current)
        precision_comparison = next(
            c for c in report.metric_comparisons if c.name == "precision_at_k"
        )
        assert precision_comparison.regressed is False

    def test_improvement_does_not_regress(self) -> None:
        from src.ai.eval.compare import compare_metrics
        from src.ai.eval.models import EvalMetrics, EvalResults

        m_base = EvalMetrics(
            precision_at_k=0.6,
            recall=0.7,
            mrr=0.65,
            negative_rejection_rate=0.9,
            p95_latency_ms=200.0,
            total_cases=10,
            negative_cases=2,
            positive_cases=8,
        )
        m_cur = EvalMetrics(
            precision_at_k=0.9,
            recall=0.95,
            mrr=0.9,
            negative_rejection_rate=1.0,
            p95_latency_ms=150.0,
            total_cases=10,
            negative_cases=2,
            positive_cases=8,
        )
        baseline = EvalResults(
            timestamp="2024-01-01T00:00:00Z", git_commit="abc", metrics=m_base
        )
        current = EvalResults(
            timestamp="2024-01-02T00:00:00Z", git_commit="def", metrics=m_cur
        )
        report = compare_metrics(baseline, current)
        assert report.has_regression is False
        assert all(not c.regressed for c in report.metric_comparisons)
