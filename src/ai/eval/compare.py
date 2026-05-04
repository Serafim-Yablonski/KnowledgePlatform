"""Compare two evaluation runs and flag regressions."""

import sys


def main(baseline: str, current: str) -> None:
    """Entry point — not yet implemented."""
    _ = baseline, current


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 3:
        print("Usage: python -m src.ai.eval.compare <baseline.json> <current.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
