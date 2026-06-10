"""Integration test for the honeypot rate in the final submission (Task 21.4).

Requirement 2.6: on the provided dataset, honeypots SHALL be < 10% of the top 100.

This test runs the full pipeline on the real ``candidates.jsonl`` (100k records)
and then independently recomputes the set of honeypot-flagged candidate_ids using
the same deterministic ``eval_date`` the pipeline uses. It then measures the
fraction of the output's top-100 candidate_ids that fall in that flagged set and
asserts the fraction is < 0.10 (Req 2.6).

Important note on the expected value
------------------------------------
The pipeline *already excludes* honeypot-flagged candidates from the ranked
output (see ``rank.run_pipeline`` step 3 and ``HoneypotDetector.detect``; Req
2.5). Therefore the honeypot rate in the output is expected to be exactly 0.0.
The < 10% assertion is the requirement-level guarantee (Req 2.6); we additionally
assert the stronger, design-implied invariant that the rate is exactly 0 so a
regression in the exclusion step would be caught here.

These tests use the full 100k dataset and are marked ``integration`` and ``slow``
so they can be deselected with ``-m "not slow"``. If ``candidates.jsonl`` is not
present the test is skipped gracefully.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from rank import determine_eval_date, run_pipeline
from ranking.honeypot import HoneypotDetector
from ranking.loader import DataLoader

# Absolute path to the full dataset at the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "candidates.jsonl"

TOP_N = 100
MAX_HONEYPOT_FRACTION = 0.10


def _read_output_candidate_ids(csv_path: Path) -> list[str]:
    """Return the ordered list of candidate_ids from a submission CSV."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row["candidate_id"] for row in reader]


@pytest.mark.integration
@pytest.mark.slow
def test_honeypot_rate_in_top_100_below_threshold(tmp_path):
    """The honeypot fraction among the top-100 output must be < 10% (Req 2.6).

    Because the pipeline excludes honeypots before ranking, the observed rate is
    expected to be exactly 0; we assert both the requirement bound and the
    stronger zero-rate invariant.
    """
    if not DATASET_PATH.exists():
        pytest.skip(f"Full dataset not found at {DATASET_PATH}; skipping integration test.")

    output_path = tmp_path / "submission.csv"

    # 1. Run the full pipeline on the real dataset.
    exit_code = run_pipeline(str(DATASET_PATH), str(output_path))
    assert exit_code == 0, "Pipeline should complete successfully (exit code 0)."
    assert output_path.exists(), "Pipeline should have produced an output CSV."

    # 2. Independently recompute the honeypot-flagged ids using the same
    #    deterministic eval_date the pipeline derives from the dataset.
    all_candidates, _stats = DataLoader().load(str(DATASET_PATH))
    eval_date = determine_eval_date(all_candidates)
    _clean, flagged_ids = HoneypotDetector().detect(all_candidates, eval_date)

    # 3. Read the top-100 candidate_ids from the produced submission.
    output_ids = _read_output_candidate_ids(output_path)
    assert len(output_ids) == TOP_N, f"Expected {TOP_N} output rows, got {len(output_ids)}."

    # 4. Measure the fraction of output ids that are honeypots.
    honeypot_in_output = [cid for cid in output_ids if cid in flagged_ids]
    honeypot_fraction = len(honeypot_in_output) / len(output_ids)

    print(
        f"[honeypot-rate] eval_date={eval_date.isoformat()} "
        f"total_flagged={len(flagged_ids)} "
        f"honeypots_in_top_100={len(honeypot_in_output)} "
        f"fraction={honeypot_fraction:.4f}"
    )

    # Requirement 2.6: strictly below 10%.
    assert honeypot_fraction < MAX_HONEYPOT_FRACTION, (
        f"Honeypot fraction {honeypot_fraction:.4f} in the top 100 exceeds the "
        f"{MAX_HONEYPOT_FRACTION:.0%} limit (Req 2.6)."
    )

    # Stronger design-implied invariant: the pipeline excludes honeypots, so the
    # rate should be exactly 0 (Req 2.5). A non-zero value here would indicate a
    # regression in the exclusion step.
    assert honeypot_fraction == 0.0, (
        "Honeypots are excluded before ranking (Req 2.5), so the expected rate "
        f"is 0; found {len(honeypot_in_output)} honeypot(s) in the top 100."
    )
