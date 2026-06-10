"""Property-based tests for :class:`ranking.formatter.SubmissionFormatter`.

Covers design correctness Property 21 (task 18.2): output candidate_id
integrity and score precision. The tests build a full 100-row ranked
submission with Hypothesis-generated score sequences, write it to a temporary
CSV, then read the file back and assert every invariant the official
``validate_submission.py`` enforces:

- the header row is exactly ``candidate_id,rank,score,reasoning``;
- there are exactly 100 data rows;
- ranks are the integers 1..100, each appearing once as a plain integer;
- every candidate_id matches ``CAND_XXXXXXX``, appears once, and is a member of
  the supplied input set;
- every score is formatted with exactly 4 decimal places; and
- scores are non-increasing down the rows.

A separate test confirms that rows whose candidate_id is malformed or absent
from ``valid_candidate_ids`` are excluded from the output (Req 9.8).
"""

from __future__ import annotations

import csv
import os
import re
import tempfile

from hypothesis import given
from hypothesis import strategies as st

from ranking.formatter import (
    CANDIDATE_ID_PATTERN,
    EXPECTED_ROWS,
    HEADER,
    SubmissionFormatter,
    format_score,
)
from ranking.models import RankedCandidate

# A score column cell must be a plain decimal with exactly 4 fractional digits.
_SCORE_RE = re.compile(r"^\d+\.\d{4}$")


def _candidate_id(index: int) -> str:
    """Build a unique, well-formed candidate_id (``CAND_`` + 7 digits)."""
    return f"CAND_{index:07d}"


def _read_csv(path: str):
    """Read a written submission back as ``(header, data_rows)``."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _new_csv_path() -> str:
    """Allocate a fresh temp ``.csv`` path (Hypothesis runs many examples, so a
    per-example file avoids the function-scoped-fixture health check)."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    return path


# Feature: candidate-ranking-system, Property 21: Output candidate_id integrity and score precision
@given(
    # 100 raw scores in [0, 1]; sorted descending below to build a valid,
    # non-increasing ranked submission.
    raw_scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=EXPECTED_ROWS,
        max_size=EXPECTED_ROWS,
    ),
    # A starting offset so candidate_ids vary across examples while staying
    # unique and within the 7-digit pattern.
    id_offset=st.integers(min_value=0, max_value=9_000_000),
)
def test_output_integrity_and_score_precision(raw_scores, id_offset):
    """Property 21: a 100-row submission round-trips with all invariants intact.

    **Validates: Requirements 8.8, 9.2, 9.3, 9.6, 9.8**
    """
    scores = sorted(raw_scores, reverse=True)
    candidate_ids = [_candidate_id(id_offset + i + 1) for i in range(EXPECTED_ROWS)]
    valid_ids = set(candidate_ids)

    ranked = [
        RankedCandidate(
            candidate_id=cid,
            rank=i + 1,
            score=scores[i],
            reasoning=f"row {i + 1} reasoning",
        )
        for i, cid in enumerate(candidate_ids)
    ]

    out_path = _new_csv_path()
    SubmissionFormatter().write(ranked, out_path, valid_candidate_ids=valid_ids)

    header, data_rows = _read_csv(out_path)

    # Header row is exactly the required columns in order (Req 9.1).
    assert header == HEADER
    # Exactly 100 data rows (Req 9.2).
    assert len(data_rows) == EXPECTED_ROWS

    seen_ranks = []
    seen_ids = set()
    prev_score = None
    for cells in data_rows:
        assert len(cells) == len(HEADER)
        cid, rank_s, score_s, _reasoning = cells

        # Rank is a plain integer (str(int)) (Req 9.2).
        assert rank_s == str(int(rank_s))
        seen_ranks.append(int(rank_s))

        # candidate_id matches the pattern, is unique, and came from the input
        # set (Req 9.3, 9.8).
        assert CANDIDATE_ID_PATTERN.match(cid)
        assert cid not in seen_ids
        seen_ids.add(cid)
        assert cid in valid_ids

        # Score has exactly 4 decimal places (Req 8.8, 9.6).
        assert _SCORE_RE.match(score_s)

        # Scores are non-increasing down the rows (Req 9.4).
        score = float(score_s)
        if prev_score is not None:
            assert score <= prev_score
        prev_score = score

    # Ranks are exactly 1..100, each once (Req 9.2).
    assert sorted(seen_ranks) == list(range(1, EXPECTED_ROWS + 1))


# Feature: candidate-ranking-system, Property 21: Output candidate_id integrity and score precision
@given(
    x=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
def test_format_score_always_four_decimals(x):
    """``format_score`` yields a 4-decimal string equal to ``round(x, 4)``.

    **Validates: Requirements 8.8, 9.6**
    """
    s = format_score(x)
    assert _SCORE_RE.match(s)
    assert float(s) == round(x, 4)


def test_format_score_known_values():
    """Spot-check the 4-decimal contract on representative inputs.

    **Validates: Requirements 8.8, 9.6**
    """
    assert format_score(0) == "0.0000"
    assert format_score(1) == "1.0000"
    # Banker's rounding via str formatting: 0.12345 -> "0.1235" or "0.1234".
    s = format_score(0.12345)
    assert _SCORE_RE.match(s)
    assert float(s) == round(0.12345, 4)


# Feature: candidate-ranking-system, Property 21: Output candidate_id integrity and score precision
@given(
    raw_scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=EXPECTED_ROWS,
        max_size=EXPECTED_ROWS,
    )
)
def test_invalid_candidate_ids_excluded(raw_scores):
    """Rows with malformed or unknown candidate_ids are excluded (Req 9.8).

    Two bad rows are interleaved among 100 good ones. The written file should
    contain only the 100 good rows; the count being != 100 is acceptable for
    this targeted exclusion check (it is not the main 100-row test).

    **Validates: Requirements 9.8**
    """
    scores = sorted(raw_scores, reverse=True)
    good_ids = [_candidate_id(i + 1) for i in range(EXPECTED_ROWS)]
    valid_ids = set(good_ids)

    good = [
        RankedCandidate(
            candidate_id=cid,
            rank=i + 1,
            score=scores[i],
            reasoning="ok",
        )
        for i, cid in enumerate(good_ids)
    ]

    # A malformed id (wrong pattern) and a well-formed id that is not in the
    # input set; both must be filtered out.
    malformed = RankedCandidate(
        candidate_id="BADID_123", rank=101, score=0.5, reasoning="malformed"
    )
    unknown = RankedCandidate(
        candidate_id="CAND_9999999", rank=102, score=0.4, reasoning="unknown"
    )

    ranked = good[:50] + [malformed] + good[50:] + [unknown]

    out_path = _new_csv_path()
    SubmissionFormatter().write(ranked, out_path, valid_candidate_ids=valid_ids)

    _header, data_rows = _read_csv(out_path)
    written_ids = {cells[0] for cells in data_rows}

    assert "BADID_123" not in written_ids
    assert "CAND_9999999" not in written_ids
    assert written_ids == valid_ids
    assert len(data_rows) == EXPECTED_ROWS
