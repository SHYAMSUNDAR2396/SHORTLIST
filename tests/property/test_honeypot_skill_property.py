# Feature: candidate-ranking-system, Property 4: Honeypot skill-duration anomaly
"""Property 4: Honeypot skill-duration anomaly.

**Validates: Requirements 2.3**

The detector SHALL flag the candidate if and only if at least one skill's
``duration_months`` exceeds the candidate's total career span (earliest
start_date to eval_date, in months) by more than 12 months.
"""

from datetime import date

from hypothesis import given
from hypothesis import strategies as st

from ranking.honeypot import (
    SKILL_DURATION_ANOMALY_THRESHOLD_MONTHS,
    HoneypotDetector,
    months_between,
)
from tests.property._factories import EVAL_DATE, build_candidate, career_entry, skill


def _date_months_before(n: int) -> date:
    total = EVAL_DATE.year * 12 + (EVAL_DATE.month - 1) - n
    year, month = divmod(total, 12)
    return date(year, month + 1, 1)


@given(
    start_back=st.integers(min_value=0, max_value=300),
    skill_durations=st.lists(
        st.integers(min_value=0, max_value=500), min_size=0, max_size=6
    ),
)
def test_skill_duration_rule_matches_predicate(start_back, skill_durations):
    detector = HoneypotDetector()
    start = _date_months_before(start_back)
    span_months = months_between(start, EVAL_DATE)

    skills = [skill(duration_months=d) for d in skill_durations]
    expected = any(
        d - span_months > SKILL_DURATION_ANOMALY_THRESHOLD_MONTHS
        for d in skill_durations
    )

    # Single benign career entry whose own duration matches its span so Rule 1
    # stays quiet; experience derived from span keeps Rule 2 quiet.
    entry = career_entry(
        start_date=start,
        end_date=None,
        duration_months=span_months,
        is_current=True,
    )
    candidate = build_candidate(
        career_history=[entry],
        skills=skills,
        years_of_experience=span_months / 12.0,
    )

    assert detector._violates_skill_duration_rule(candidate, EVAL_DATE) is expected


@given(
    start_back=st.integers(min_value=0, max_value=240),
    excess=st.integers(min_value=-12, max_value=12),
)
def test_skill_duration_threshold_boundary(start_back, excess):
    """skill duration = span + 12 + excess => flagged iff excess > 0."""
    detector = HoneypotDetector()
    start = _date_months_before(start_back)
    span_months = months_between(start, EVAL_DATE)
    duration = span_months + SKILL_DURATION_ANOMALY_THRESHOLD_MONTHS + excess

    entry = career_entry(
        start_date=start,
        end_date=None,
        duration_months=span_months,
        is_current=True,
    )
    candidate = build_candidate(
        career_history=[entry],
        skills=[skill(duration_months=duration)],
        years_of_experience=span_months / 12.0,
    )

    assert detector._violates_skill_duration_rule(candidate, EVAL_DATE) is (
        excess > 0
    )
