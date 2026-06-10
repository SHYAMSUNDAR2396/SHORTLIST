# Feature: candidate-ranking-system, Property 3: Honeypot experience-span mismatch
"""Property 3: Honeypot experience-span mismatch.

**Validates: Requirements 2.2**

The detector SHALL flag the candidate if and only if the absolute difference
between ``years_of_experience`` (in months) and the total career span (earliest
start_date to latest end_date or eval_date, in months) exceeds 36 months.
"""

from datetime import date

from hypothesis import given
from hypothesis import strategies as st

from ranking.honeypot import (
    EXPERIENCE_SPAN_MISMATCH_THRESHOLD_MONTHS,
    HoneypotDetector,
    months_between,
)
from tests.property._factories import EVAL_DATE, build_candidate, career_entry


def _date_months_before(n: int) -> date:
    total = EVAL_DATE.year * 12 + (EVAL_DATE.month - 1) - n
    year, month = divmod(total, 12)
    return date(year, month + 1, 1)


# Career entries described by (start_months_back, end_months_back_or_None).
# end_months_back=None => current role (treated as eval_date).
_entry = st.tuples(
    st.integers(min_value=0, max_value=300),
    st.one_of(st.none(), st.integers(min_value=0, max_value=300)),
)


def _build_entry(start_back, end_back):
    start = _date_months_before(start_back)
    if end_back is None:
        return career_entry(start_date=start, end_date=None, is_current=True), start, EVAL_DATE
    # Ensure end is not before start so dates are coherent.
    end_back = min(end_back, start_back)
    end = _date_months_before(end_back)
    return career_entry(start_date=start, end_date=end, is_current=False), start, end


@given(
    specs=st.lists(_entry, min_size=1, max_size=5),
    years_of_experience=st.floats(min_value=0.0, max_value=40.0, allow_nan=False),
)
def test_experience_span_rule_matches_predicate(specs, years_of_experience):
    detector = HoneypotDetector()

    entries = []
    starts = []
    ends = []
    for start_back, end_back in specs:
        entry, start, end = _build_entry(start_back, end_back)
        entries.append(entry)
        starts.append(start)
        ends.append(end)

    earliest_start = min(starts)
    latest_end = max(ends)
    span_months = months_between(earliest_start, latest_end)
    experience_months = years_of_experience * 12
    expected = (
        abs(experience_months - span_months)
        > EXPERIENCE_SPAN_MISMATCH_THRESHOLD_MONTHS
    )

    candidate = build_candidate(
        career_history=entries, years_of_experience=years_of_experience
    )

    assert (
        detector._violates_experience_span_rule(candidate, EVAL_DATE) is expected
    )


@given(
    years=st.integers(min_value=4, max_value=20),
    excess_months=st.integers(min_value=-12, max_value=12),
)
def test_experience_span_threshold_boundary(years, excess_months):
    """span = experience - 36 - excess => |diff| = |36 + excess|.

    Flagged iff |36 + excess| > 36, i.e. iff excess > 0 (excess in [-12, 12]).
    Years are integers so ``years_of_experience * 12`` is exact (no float drift
    at the boundary).
    """
    detector = HoneypotDetector()
    experience_months = years * 12
    span_months = experience_months - EXPERIENCE_SPAN_MISMATCH_THRESHOLD_MONTHS - excess_months

    start = _date_months_before(span_months)
    entry = career_entry(start_date=start, end_date=None, is_current=True)
    candidate = build_candidate(
        career_history=[entry], years_of_experience=float(years)
    )

    assert detector._violates_experience_span_rule(candidate, EVAL_DATE) is (
        excess_months > 0
    )
