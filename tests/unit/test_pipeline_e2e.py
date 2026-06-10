"""End-to-end smoke test for the full ranking pipeline (task 19.2).

Runs :func:`rank.run_pipeline` (and :func:`rank.main`) on a *small* subset of
the real ``candidates.jsonl`` and asserts the emitted CSV obeys the submission
contract: a header plus exactly 100 data rows, ranks 1..100, and non-increasing
scores by rank.

_Requirements: 9.1, 9.2, 8.5_
"""

from __future__ import annotations

import csv

import rank
from tests.integration._helpers import make_sample_jsonl

SAMPLE_SIZE = 300


def _read_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_run_pipeline_emits_valid_top_100(tmp_path):
    """run_pipeline on a small sample produces a header + 100 ranked rows."""
    sample = make_sample_jsonl(tmp_path / "sample.jsonl", n=SAMPLE_SIZE)
    out_path = tmp_path / "submission.csv"

    exit_code = rank.run_pipeline(str(sample), str(out_path))

    assert exit_code == 0
    assert out_path.exists()

    rows = _read_rows(out_path)
    # Row 0 is the header; rows 1..100 are data (Req 9.1, 9.2).
    assert rows[0] == ["candidate_id", "rank", "score", "reasoning"]
    data_rows = rows[1:]
    assert len(data_rows) == 100

    ranks = [int(r[1]) for r in data_rows]
    assert ranks == list(range(1, 101))  # ranks 1..100 in order (Req 8.5)

    scores = [float(r[2]) for r in data_rows]
    # Scores must be non-increasing by rank position (Req 8.5).
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_main_returns_zero_on_small_sample(tmp_path):
    """main([...]) wires args through run_pipeline and returns exit code 0."""
    sample = make_sample_jsonl(tmp_path / "sample.jsonl", n=SAMPLE_SIZE)
    out_path = tmp_path / "submission.csv"

    exit_code = rank.main(["--candidates", str(sample), "--out", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()
