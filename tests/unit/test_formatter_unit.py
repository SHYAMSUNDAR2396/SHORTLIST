"""Unit tests for :class:`ranking.formatter.SubmissionFormatter` (task 18.3).

Example-based checks for the CSV contract that are awkward to express as
universal properties:

- the first physical line of the output is exactly the required header (Req 9.1);
- the formatter writes to the ``.csv`` path it is given;
- a reasoning field containing a comma round-trips intact (the csv module
  quotes it, Req 9.7); and
- a newline inside reasoning is scrubbed to a space so it cannot split a CSV
  row (Req 11.5).
"""

from __future__ import annotations

import csv

from ranking.formatter import SubmissionFormatter
from ranking.models import RankedCandidate


def _ranked(candidate_id: str, rank: int, score: float, reasoning: str):
    return RankedCandidate(
        candidate_id=candidate_id, rank=rank, score=score, reasoning=reasoning
    )


def test_header_is_exact_first_line(tmp_path):
    """The first line of the output is exactly the required header (Req 9.1).

    A short (< 100 row) submission is fine here; write() warns to stderr about
    the row count but still emits the header and rows.
    """
    out_path = tmp_path / "team_001.csv"
    ranked = [
        _ranked("CAND_0000001", 1, 0.9, "good fit"),
        _ranked("CAND_0000002", 2, 0.8, "also good"),
    ]
    SubmissionFormatter().write(ranked, str(out_path), valid_candidate_ids=None)

    with open(out_path, "r", encoding="utf-8", newline="") as f:
        first_line = f.readline().rstrip("\r\n")

    assert first_line == "candidate_id,rank,score,reasoning"


def test_output_path_uses_csv_extension(tmp_path):
    """The formatter writes to the given ``.csv`` path (Req 9.1)."""
    out_path = tmp_path / "submission.csv"
    SubmissionFormatter().write(
        [_ranked("CAND_0000001", 1, 0.5, "fit")], str(out_path)
    )

    assert out_path.exists()
    assert str(out_path).endswith(".csv")


def test_reasoning_with_comma_is_quoted_and_round_trips(tmp_path):
    """A comma inside reasoning is preserved as a single field (Req 9.7)."""
    out_path = tmp_path / "submission.csv"
    reasoning = "Senior AI Engineer, 7 yrs; top factor: skill=0.92"
    SubmissionFormatter().write(
        [_ranked("CAND_0000001", 1, 0.92, reasoning)], str(out_path)
    )

    with open(out_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    # rows[0] is the header; rows[1] is the data row.
    assert len(rows) == 2
    data = rows[1]
    assert len(data) == 4  # the comma did not split the field
    assert data[3] == reasoning


def test_reasoning_newline_is_scrubbed_to_space(tmp_path):
    """A newline inside reasoning is replaced with a space (Req 11.5)."""
    out_path = tmp_path / "submission.csv"
    reasoning = "line one\nline two"
    SubmissionFormatter().write(
        [_ranked("CAND_0000001", 1, 0.7, reasoning)], str(out_path)
    )

    with open(out_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 2  # no extra row introduced by the newline
    assert rows[1][3] == "line one line two"
