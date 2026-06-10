"""Performance benchmark integration test (Task 21.5).

Requirements:
- 10.1 / 10.6: the full 100k-candidate pipeline completes within 300 seconds.
- 10.4: peak memory stays under 14 GB.

This test times ``rank.run_pipeline`` on the real ``candidates.jsonl`` using a
wall-clock measurement (``time.perf_counter``) and asserts it finishes within a
generous 300-second budget (the real run takes only a few seconds). It also
asserts the output contains exactly 100 data rows.

Memory is measured best-effort via ``resource.getrusage(...).ru_maxrss``. Peak
RSS reporting is platform-dependent (bytes on macOS, kilobytes on Linux) and can
be unreliable inside test harnesses, so the memory assertion is kept lenient and
skipped when a reliable reading is unavailable, to avoid flaky failures. The
runtime and row-count assertions are the primary guarantees.

These tests use the full 100k dataset and are marked ``integration`` and ``slow``
so they can be deselected with ``-m "not slow"``. If ``candidates.jsonl`` is not
present the test is skipped gracefully.
"""

from __future__ import annotations

import csv
import resource
import sys
import time
from pathlib import Path

import pytest

from rank import run_pipeline

# Absolute path to the full dataset at the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "candidates.jsonl"

TOP_N = 100
MAX_RUNTIME_SECONDS = 300.0
MAX_MEMORY_BYTES = 14 * 1024 * 1024 * 1024  # 14 GB


def _count_data_rows(csv_path: Path) -> int:
    """Return the number of data rows (excluding the header) in a CSV."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # First row is the header; data rows follow.
    return max(0, len(rows) - 1)


def _peak_rss_bytes() -> int:
    """Best-effort peak resident-set size in bytes.

    ``ru_maxrss`` units differ by platform: bytes on macOS/Darwin, kilobytes on
    Linux. Normalize to bytes so the 14 GB comparison is meaningful.
    """
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(max_rss)  # already bytes on macOS
    return int(max_rss) * 1024  # kilobytes -> bytes on Linux and most Unixes


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_runtime_and_output_size(tmp_path):
    """Full 100k pipeline finishes < 300s and emits exactly 100 rows (Req 10.1, 10.6)."""
    if not DATASET_PATH.exists():
        pytest.skip(f"Full dataset not found at {DATASET_PATH}; skipping integration test.")

    output_path = tmp_path / "submission.csv"

    start = time.perf_counter()
    exit_code = run_pipeline(str(DATASET_PATH), str(output_path))
    elapsed = time.perf_counter() - start

    print(f"[performance] run_pipeline wall-clock time: {elapsed:.2f}s")

    assert exit_code == 0, "Pipeline should complete successfully (exit code 0)."
    assert output_path.exists(), "Pipeline should have produced an output CSV."

    # Req 10.1 / 10.6: within the 300-second budget (generous; real run is seconds).
    assert elapsed < MAX_RUNTIME_SECONDS, (
        f"Pipeline took {elapsed:.2f}s which exceeds the "
        f"{MAX_RUNTIME_SECONDS:.0f}s limit (Req 10.1, 10.6)."
    )

    # Exactly 100 data rows in the submission.
    data_rows = _count_data_rows(output_path)
    assert data_rows == TOP_N, f"Expected exactly {TOP_N} data rows, got {data_rows}."


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_peak_memory_under_limit(tmp_path):
    """Peak RSS stays under 14 GB (Req 10.4); lenient/skippable to avoid flakiness."""
    if not DATASET_PATH.exists():
        pytest.skip(f"Full dataset not found at {DATASET_PATH}; skipping integration test.")

    output_path = tmp_path / "submission.csv"

    exit_code = run_pipeline(str(DATASET_PATH), str(output_path))
    assert exit_code == 0, "Pipeline should complete successfully (exit code 0)."

    peak_bytes = _peak_rss_bytes()
    if peak_bytes <= 0:
        pytest.skip("Peak RSS reading unavailable/unreliable on this platform.")

    peak_gb = peak_bytes / (1024 ** 3)
    print(f"[performance] peak RSS: {peak_gb:.2f} GB")

    # Req 10.4: under 14 GB. ru_maxrss is process-wide (includes the test
    # harness), so this is a generous upper bound rather than a tight measure.
    assert peak_bytes < MAX_MEMORY_BYTES, (
        f"Peak RSS {peak_gb:.2f} GB exceeds the 14 GB limit (Req 10.4)."
    )
