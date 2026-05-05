"""Regression gate — fails if any metric dropped >10% from the baseline.

Reads current scores from tests/golden/current_scores.json (written by
test_golden_regression.py after each run) and compares against the saved
baseline in tests/golden/baseline_scores.json.

Flow in CD pipeline:
  1. deepeval test run → writes current_scores.json
  2. This script reads current vs baseline
  3. Any metric dropped >10% → exit 1 (fail CD)
  4. If pass → baseline updated with current scores

Usage:
  uv run python scripts/regression_gate.py

Exit codes:
  0 — all metrics within tolerance, baseline updated
  1 — regression detected (>10% drop in at least one metric)
"""
import json
import sys
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent.parent / "tests/golden"
BASELINE_FILE = GOLDEN_DIR / "baseline_scores.json"
CURRENT_FILE = GOLDEN_DIR / "current_scores.json"
REGRESSION_TOLERANCE = 0.10  # 10% drop allowed


def main() -> None:
    # Read current scores (written by test_golden_regression.py)
    if not CURRENT_FILE.exists():
        print(f"No current scores found at {CURRENT_FILE}")
        print("Skipping regression gate (scores not written by test runner).")
        sys.exit(0)

    current = json.loads(CURRENT_FILE.read_text())
    print(f"Current scores:  {json.dumps(current)}")

    # If no baseline, save and exit
    if not BASELINE_FILE.exists():
        _save_baseline(current)
        print("No baseline found. Saved current scores as first baseline.")
        sys.exit(0)

    baseline = json.loads(BASELINE_FILE.read_text())
    print(f"Baseline scores: {json.dumps(baseline)}")
    print(f"Tolerance: {REGRESSION_TOLERANCE:.0%}")
    print()

    # Compare
    failed = False
    for metric_name, prev_score in baseline.items():
        curr_score = current.get(metric_name)

        if curr_score is None:
            print(f"  SKIP  {metric_name}: not in current run")
            continue

        if prev_score == 0:
            print(f"  SKIP  {metric_name}: baseline=0")
            continue

        drop = (prev_score - curr_score) / prev_score

        if drop > REGRESSION_TOLERANCE:
            print(f"  FAIL  {metric_name}: {prev_score:.2f} -> {curr_score:.2f} (dropped {drop:.1%}, exceeds {REGRESSION_TOLERANCE:.0%})")
            failed = True
        elif drop < 0:
            print(f"  OK    {metric_name}: {prev_score:.2f} -> {curr_score:.2f} (improved)")
        else:
            print(f"  OK    {metric_name}: {prev_score:.2f} -> {curr_score:.2f} (within tolerance)")

    print()
    if failed:
        print("REGRESSION DETECTED. Baseline NOT updated.")
        sys.exit(1)
    else:
        print("All metrics within tolerance. Baseline updated.")
        _save_baseline(current)
        sys.exit(0)


def _save_baseline(scores: dict[str, float]) -> None:
    BASELINE_FILE.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
