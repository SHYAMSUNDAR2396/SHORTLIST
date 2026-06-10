# Feature: candidate-ranking-system, Property 12: Experience-fit piecewise function
# Feature: candidate-ranking-system, Property 13: Experience validation override
"""Property tests for the ExperienceScorer.

Covers two design properties:

- **Property 12: Experience-fit piecewise function** (**Validates: Requirements
  5.1, 5.2, 5.3, 5.4**) — the piecewise mapping from an effective years value to
  a score in ``[0.2, 1.0]``.
- **Property 13: Experience validation override** (**Validates: Requirements
  5.5, 5.6**) — the stated ``years_of_experience`` is replaced by the
  career-derived total only when they disagree by more than 2.0 years; an empty
  or all-zero-duration career history leaves the stated value unchanged.
"""

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ranking.scorers.experience import (
    career_derived_years,
    effective_years,
    experience_fit,
)
from tests.property._factories import build_candidate, career_entry

# Fixed start date for controlled career intervals. Far enough in the past that
# any reasonable duration produces coherent (start, end) intervals.
_START = date(2000, 1, 1)


def _expected_experience_fit(v: float) -> float:
    """Independently computed reference for the piecewise experience-fit fn."""
    if 5.0 <= v <= 9.0:
        return 1.0
    if 4.0 <= v < 5.0:
        return 0.5 + 0.5 * ((v - 4.0) / 1.0)
    if 9.0 < v <= 11.0:
        return 1.0 - 0.5 * ((v - 9.0) / 2.0)
    return 0.2


# Feature: candidate-ranking-system, Property 12: Experience-fit piecewise function
@given(v=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False))
def test_experience_fit_matches_piecewise(v):
    """experience_fit(v) equals the independently-computed piecewise value, and
    always lies in [0.2, 1.0]. **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
    """
    result = experience_fit(v)
    assert result == pytest.approx(_expected_experience_fit(v))
    assert 0.2 <= result <= 1.0


# Feature: candidate-ranking-system, Property 13: Experience validation override
@given(
    duration_months=st.integers(min_value=1, max_value=480),
    stated=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_validation_override_uses_derived_when_disagreement_exceeds_threshold(
    duration_months, stated
):
    """When |stated - derived| > 2.0 the career-derived value is used; otherwise
    the stated value is used. **Validates: Requirements 5.5, 5.6**
    """
    entry = career_entry(start_date=_START, duration_months=duration_months)
    candidate = build_candidate(
        career_history=[entry], years_of_experience=stated
    )

    # An exact whole-month interval yields derived == duration_months / 12.
    derived = career_derived_years(candidate.career_history)
    assert derived is not None
    assert derived == pytest.approx(duration_months / 12.0)

    expected = derived if abs(stated - derived) > 2.0 else stated
    assert effective_years(candidate) == pytest.approx(expected)


# Feature: candidate-ranking-system, Property 13: Experience validation override
@given(
    stated=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    n_entries=st.integers(min_value=0, max_value=4),
)
def test_validation_skipped_when_no_derivable_span(stated, n_entries):
    """Empty career history, or entries that all have duration_months == 0,
    leave the stated value unchanged. **Validates: Requirements 5.6**
    """
    # All entries have duration_months == 0 (default) so no span is derivable.
    entries = [career_entry(start_date=_START) for _ in range(n_entries)]
    candidate = build_candidate(
        career_history=entries, years_of_experience=stated
    )

    assert career_derived_years(candidate.career_history) is None
    assert effective_years(candidate) == pytest.approx(stated)
