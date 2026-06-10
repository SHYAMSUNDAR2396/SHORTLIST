# Feature: candidate-ranking-system, Property 2: Honeypot duration anomaly detection
"""Property 2: Honeypot duration anomaly detection.

**Validates: Requirements 2.1**

The detector's per-entry duration rule SHALL flag a candidate if and only if at
least one career_history entry has ``duration_months`` exceeding the number of
months between that entry's ``start_date`` and the evaluation date by more than
24 months.
"""

from datetime import date

from hypothesis import given
from hypothesis import strategies as st

from ranking.honeypot import (
    DURATION_ANOMALY_THRESHOLD_MONTHS,
    HoneypotDetector,
    months_between,
)
from tests.property._factories import EVAL_DATE, build_candidate, career_entry


def _date_months_before(n: int) -> date:
    """Return a first-of-month date ``n`` months before EVAL_DATE."""
    total = EVAL_DATE.year * 12 + (EVAL_DATE.month - 1) - n
    year, month = divmod(total, 12)
    return date(year, month + 1, 1)


# A single career-entry spec: how many months before eval it started, and its
# claimed duration_months. The duration range straddles the +24 threshold.
_entry_specs = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=300),   # months_back (start offset)
        st.integers(min_value=0, max_value=400),   # duration_months
    ),
    min_size=0,
    max_size=6,
)


@given(specs=_entry_specs)
def test_duration_rule_matches_predicate(specs):
    detector = HoneypotDetector()

    entries = []
    expected = False
    for months_back, duration_months in specs:
        start = _date_months_before(months_back)
        entries.append(career_entry(start_date=start, duration_months=duration_months))
        span = months_between(start, EVAL_DATE)
        if duration_months - span > DURATION_ANOMALY_THRESHOLD_MONTHS:
            expected = True

    # Keep the experience/skill/expert rules quiet by overriding only careers.
    candidate = build_candidate(career_history=entries)

    assert (
        detector._violates_duration_rule(candidate, EVAL_DATE) is expected
    )


@given(
    months_back=st.integers(min_value=0, max_value=120),
    excess=st.integers(min_value=-12, max_value=12),
)
def test_duration_rule_threshold_boundary(months_back, excess):
    """Around the threshold: duration = span + 24 + excess.

    Flagged iff excess > 0 (i.e. strictly more than +24).
    """
    detector = HoneypotDetector()
    start = _date_months_before(months_back)
    span = months_between(start, EVAL_DATE)
    duration = span + DURATION_ANOMALY_THRESHOLD_MONTHS + excess
    candidate = build_candidate(
        career_history=[career_entry(start_date=start, duration_months=duration)]
    )

    assert detector._violates_duration_rule(candidate, EVAL_DATE) is (excess > 0)
