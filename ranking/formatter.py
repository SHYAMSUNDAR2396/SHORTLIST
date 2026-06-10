"""Submission CSV formatter for the Candidate Ranking System.

This module emits the final ranked output as a CSV file that passes the
official ``validate_submission.py`` checks. The contract is intentionally
strict and mirrors the validator:

- Header row is **exactly** ``candidate_id,rank,score,reasoning`` (Req 9.1).
- Exactly 100 non-empty data rows follow, with ranks ``1..100`` appearing once
  each as plain integers with no leading zeros or decimal point (Req 9.2).
- ``candidate_id`` values match ``CAND_XXXXXXX`` (7 digits) and each appears at
  most once (Req 9.3); any id not present in the supplied
  ``valid_candidate_ids`` set is excluded and reported to stderr (Req 9.8).
- ``score`` is formatted with **exactly 4 decimal places** (Req 8.8, 9.6) and
  the rounded scores are non-increasing by rank (Req 9.4); ties are expected to
  already be ordered by ``candidate_id`` ascending (Req 9.5).
- ``reasoning`` is written as-is and confined to a single line (Req 9.7, 11.5).
- The file is UTF-8 encoded and written with ``newline=''`` per the Python csv
  module contract (Req 9.7).

Design notes
------------
The formatter **trusts the input order** produced by the ranker (which already
sorts by ``(-final_score, candidate_id)``). It does not re-sort. It does,
however, defensively enforce the validator's invariants so a subtle upstream
issue cannot produce an invalid submission:

- *Score monotonicity safeguard*: scores are rounded to 4 decimals before being
  written. The raw input is already non-increasing, and rounding is applied
  consistently, so monotonicity is preserved in the common case. In the rare
  event that rounding produces a tiny inversion between two adjacent rows (the
  later rounded score exceeds the earlier one), the later score is clamped down
  to the earlier value. This keeps the emitted column non-increasing without
  reordering candidates.
- *Newline scrubbing*: although the reasoning generator already guarantees a
  single-line string, any stray newline/carriage-return characters are replaced
  with a single space defensively (Req 11.5).

The single-row formatting helper :func:`format_row` is exposed so that tests can
verify candidate_id integrity and score precision in isolation.
"""

import csv
import re
import sys
from typing import List, Optional, Set, Tuple

from .models import RankedCandidate

# Header columns, in exact order required by the validator (Req 9.1).
HEADER = ["candidate_id", "rank", "score", "reasoning"]

# candidate_id must be CAND_ followed by exactly 7 digits (Req 9.3).
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")

# Number of data rows the validator expects (Req 9.2).
EXPECTED_ROWS = 100


def format_score(score: float) -> str:
    """Format a score with exactly 4 decimal places (Req 8.8, 9.6).

    Always produces a fixed-point string such as ``"0.9200"`` or ``"1.0000"``.
    This is the canonical precision used both in the output column and when the
    monotonicity safeguard compares adjacent rows.
    """
    return f"{float(score):.4f}"


def _scrub_reasoning(reasoning: str) -> str:
    """Collapse any line-break characters in reasoning to spaces (Req 11.5).

    The reasoning generator already produces single-line text; this is a
    defensive measure so a stray ``\\n`` or ``\\r`` can never split a CSV row.
    """
    return reasoning.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def format_row(
    candidate_id: str, rank: int, score: float, reasoning: str
) -> Tuple[str, str, str, str]:
    """Format a single submission row as a 4-tuple of strings.

    Returns ``(candidate_id, rank_str, score_str, reasoning)`` where:

    - ``rank`` is rendered as a plain integer via ``str(int(rank))`` so there is
      no decimal point or leading zero (Req 9.2).
    - ``score`` is rendered with exactly 4 decimal places (Req 8.8, 9.6).
    - ``reasoning`` has any line breaks scrubbed to spaces (Req 11.5).

    The candidate_id is returned unchanged; callers are responsible for
    validating it against the known dataset (see :meth:`SubmissionFormatter.write`).
    The csv module handles quoting (QUOTE_MINIMAL), so a reasoning value
    containing commas is quoted automatically by the writer.
    """
    return (
        candidate_id,
        str(int(rank)),
        format_score(score),
        _scrub_reasoning(reasoning),
    )


class SubmissionFormatter:
    """Write a list of :class:`RankedCandidate` records to a submission CSV.

    The formatter trusts the rank order of the input list (the ranker emits
    candidates already sorted by ``(-final_score, candidate_id)``) and enforces
    the validator's structural invariants without reordering candidates.
    """

    def write(
        self,
        ranked_candidates: List[RankedCandidate],
        output_path: str,
        valid_candidate_ids: Optional[Set[str]] = None,
    ) -> None:
        """Write the ranked candidates to ``output_path`` as a CSV (Req 9.1-9.8).

        Args:
            ranked_candidates: Candidates in final rank order. Each carries its
                ``rank``, ``candidate_id``, ``score`` and ``reasoning``. When the
                list contains exactly 100 entries they are written in the given
                order.
            output_path: Destination file path. Should end in ``.csv``.
            valid_candidate_ids: Optional set of candidate_ids known to exist in
                the input dataset. When provided, any row whose candidate_id is
                not in this set is excluded and reported to stderr (Req 9.8).

        Behavior:
            - Rows are filtered to those whose candidate_id matches
              ``CAND_XXXXXXX`` and (when supplied) is a member of
              ``valid_candidate_ids``. Rejected ids are reported to stderr.
            - Scores are formatted to 4 decimals; a monotonicity safeguard
              clamps any post-rounding inversion so the score column is
              non-increasing by rank (Req 9.4).
            - The file is opened with ``newline=''`` and ``encoding='utf-8'`` and
              written with ``csv.writer`` using the default QUOTE_MINIMAL dialect
              so reasoning containing commas is quoted (Req 9.7).

        Note:
            If, after filtering, the number of rows is not exactly 100, a warning
            is written to stderr. The formatter still writes whatever valid rows
            remain; producing exactly 100 candidates is the ranker's
            responsibility (Req 9.2).
        """
        # Filter out any candidate_id that is malformed or unknown (Req 9.3, 9.8).
        accepted: List[RankedCandidate] = []
        for rc in ranked_candidates:
            cid = rc.candidate_id
            if not CANDIDATE_ID_PATTERN.match(cid):
                print(
                    f"Excluding row with malformed candidate_id: {cid!r}",
                    file=sys.stderr,
                )
                continue
            if valid_candidate_ids is not None and cid not in valid_candidate_ids:
                print(
                    f"Excluding unrecognized candidate_id not in input dataset: {cid}",
                    file=sys.stderr,
                )
                continue
            accepted.append(rc)

        if len(accepted) != EXPECTED_ROWS:
            print(
                f"Warning: expected {EXPECTED_ROWS} data rows but have "
                f"{len(accepted)} after filtering.",
                file=sys.stderr,
            )

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

            prev_score_str: Optional[str] = None
            for rc in accepted:
                score_str = format_score(rc.score)

                # Monotonicity safeguard (Req 9.4): the input is already
                # non-increasing, but rounding could in rare cases nudge an
                # adjacent score above its predecessor. If that happens, clamp
                # the later rounded score down to the previous one rather than
                # reorder candidates.
                if prev_score_str is not None and float(score_str) > float(prev_score_str):
                    score_str = prev_score_str

                row = (
                    rc.candidate_id,
                    str(int(rc.rank)),
                    score_str,
                    _scrub_reasoning(rc.reasoning),
                )
                writer.writerow(row)
                prev_score_str = score_str
