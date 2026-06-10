"""Education scoring for the Candidate Ranking System.

The :class:`EducationScorer` computes an ``education_score`` in ``[0.0, 1.0]``
for a candidate based on their education history (Req 8.6, 8.7).

Per education entry the score is a weighted blend of three signals:

    entry_score = 0.4 × tier_value(tier)
                + 0.3 × degree_to_level(degree)
                + 0.3 × field_relevance(field_of_study)

The candidate's final ``education_score`` is the MAXIMUM ``entry_score`` across
all of their education entries (Req 8.7). When a candidate has no education
entries, the score defaults to ``0.25`` (the "unknown tier" behavior described
in the design's Error Handling section).

This module is a pure function over structured input: it reads no clock and
uses no randomness, which keeps the pipeline deterministic (Req 10.5).
"""

from typing import Optional

from ranking.constants import (
    EDUCATION_DEGREE_WEIGHT,
    EDUCATION_FIELD_WEIGHT,
    EDUCATION_TIER_WEIGHT,
    degree_to_level,
    field_relevance,
    tier_value,
)
from ranking.models import CandidateProfile, EducationEntry

# Default education_score when a candidate has no education entries. This matches
# the "unknown" tier value (0.25) so candidates without education data are
# treated the same as those with an unknown-tier institution.
EMPTY_EDUCATION_DEFAULT: float = 0.25


def _clamp(value: float) -> float:
    """Constrain a score to the inclusive range [0.0, 1.0] (Req 8.3)."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def score_entry(entry: EducationEntry) -> float:
    """Compute the education score for a single education entry.

    Combines tier, degree level, and field relevance using the
    ``EDUCATION_*_WEIGHT`` constants (0.4 / 0.3 / 0.3):

        0.4 × tier_value + 0.3 × degree_to_level + 0.3 × field_relevance

    The result is clamped to ``[0.0, 1.0]``. Exposed as a module-level helper so
    that property tests can target the per-entry formula and max-selection
    independently of the candidate-level aggregation.
    """
    entry_score = (
        EDUCATION_TIER_WEIGHT * tier_value(entry.tier)
        + EDUCATION_DEGREE_WEIGHT * degree_to_level(entry.degree)
        + EDUCATION_FIELD_WEIGHT * field_relevance(entry.field_of_study)
    )
    return _clamp(entry_score)


class EducationScorer:
    """Scores a candidate's education on a [0.0, 1.0] scale (Req 8.6, 8.7)."""

    def score(self, candidate: CandidateProfile) -> float:
        """Return the candidate's ``education_score`` in ``[0.0, 1.0]``.

        Uses the maximum per-entry score across all education entries. Falls
        back to :data:`EMPTY_EDUCATION_DEFAULT` (0.25) when the candidate has no
        education entries.
        """
        education = candidate.education
        if not education:
            return EMPTY_EDUCATION_DEFAULT

        best: Optional[float] = None
        for entry in education:
            entry_score = score_entry(entry)
            if best is None or entry_score > best:
                best = entry_score

        # ``best`` is non-None here because ``education`` is non-empty.
        return _clamp(best if best is not None else EMPTY_EDUCATION_DEFAULT)
