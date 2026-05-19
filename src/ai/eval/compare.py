"""Compare two evaluation runs and flag regressions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from src.ai.eval.models import EvalResults

_REGRESSION_THRESHOLD = 0.05
_HIGHER_IS_BETTER = {
    "precision_at_k",
    "recall",
    "mrr",
    "negative_rejection_rate",
}
_LOWER_IS_BETTER = {
    "p95_latency_ms",
}


@dataclass
class MetricComparison:
    name: str
    baseline: float
    current: float
    delta: float
    regressed: bool


@dataclass
class ComparisonReport:
    metric_comparisons: list[MetricComparison]
    improved_cases: list[str]
    regressed_cases: list[str]
    has_regression: bool


def _load(path: str) -> EvalResults:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalResults.from_dict(data)


def compare_metrics(
    baseline: EvalResults,
    current: EvalResults,
    threshold: float = _REGRESSION_THRESHOLD,
) -> ComparisonReport:
    b = baseline.metrics
    c = current.metrics

    comparisons: list[MetricComparison] = []
    for name in _HIGHER_IS_BETTER:
        bval = getattr(b, name)
        cval = getattr(c, name)
        delta = cval - bval
        regressed = delta < -threshold
        comparisons.append(
            MetricComparison(
                name=name, baseline=bval, current=cval, delta=delta, regressed=regressed
            )
        )

    for name in _LOWER_IS_BETTER:
        bval = getattr(b, name)
        cval = getattr(c, name)
        delta = cval - bval
        # For latency, a large positive delta (got slower) is a regression.
        # We use threshold as an absolute ms cutoff for latency.
        regressed = delta > threshold * 1000
        comparisons.append(
            MetricComparison(
                name=name, baseline=bval, current=cval, delta=delta, regressed=regressed
            )
        )

    # Per-case comparison: find cases that changed recall or precision significantly.
    baseline_by_id = {r.case_id: r for r in baseline.cases}
    current_by_id = {r.case_id: r for r in current.cases}

    improved_cases: list[str] = []
    regressed_cases: list[str] = []

    for case_id, cur in current_by_id.items():
        base = baseline_by_id.get(case_id)
        if base is None:
            continue
        base_score = (base.precision_at_k + base.recall) / 2
        cur_score = (cur.precision_at_k + cur.recall) / 2
        delta = cur_score - base_score
        if delta > 0.1:
            improved_cases.append(case_id)
        elif delta < -0.1:
            regressed_cases.append(case_id)

    has_regression = any(c.regressed for c in comparisons) or bool(regressed_cases)
    return ComparisonReport(
        metric_comparisons=comparisons,
        improved_cases=improved_cases,
        regressed_cases=regressed_cases,
        has_regression=has_regression,
    )


def _print_table(report: ComparisonReport, verbose: bool = False) -> None:
    print(f"\n{'Metric':<30} {'Baseline':>10} {'Current':>10} {'Delta':>10}  Status")
    print("-" * 70)
    for m in report.metric_comparisons:
        status = "❌ REGRESSION" if m.regressed else "✅"
        if m.name == "p95_latency_ms":
            print(
                f"  {m.name:<28} {m.baseline:>9.0f}ms"
                f" {m.current:>9.0f}ms {m.delta:>+9.0f}ms  {status}"
            )
        else:
            print(
                f"  {m.name:<28} {m.baseline:>10.3f}"
                f" {m.current:>10.3f} {m.delta:>+10.3f}  {status}"
            )

    if report.improved_cases:
        print(f"\n✅ Improved cases ({len(report.improved_cases)}):")
        for case_id in sorted(report.improved_cases):
            print(f"   + {case_id}")

    if report.regressed_cases:
        print(f"\n❌ Regressed cases ({len(report.regressed_cases)}):")
        for case_id in sorted(report.regressed_cases):
            print(f"   - {case_id}")

    if verbose and report.regressed_cases:
        print("\n── Regressed case details ──")
        # detailed info would be populated if we pass in the full EvalResults
        # (kept as a placeholder; runner output includes full per-case data)


def main(baseline_path: str, current_path: str, verbose: bool = False) -> int:
    baseline = _load(baseline_path)
    current = _load(current_path)

    print("\nComparing eval runs:")
    print(f"  baseline:  {baseline.timestamp}  commit={baseline.git_commit}")
    print(f"  current:   {current.timestamp}  commit={current.git_commit}")

    report = compare_metrics(baseline, current)
    _print_table(report, verbose=verbose)

    if report.has_regression:
        print("\n❌ REGRESSION DETECTED — failing with exit code 1")
        return 1

    print("\n✅ No regressions detected")
    return 0


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Compare two eval result files")
    parser.add_argument("baseline", help="Path to baseline results JSON")
    parser.add_argument("current", help="Path to current results JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    sys.exit(main(args.baseline, args.current, verbose=args.verbose))
