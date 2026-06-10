# Feature: candidate-ranking-system, Property 15: Behavioral score multiplicative combination
"""Property 15: Behavioral score multiplicative combination.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8**

For any candidate and evaluation date, the behavioral score SHALL equal the
product, starting from base 1.0, of:

* an engagement modifier (1.2 if ``recruiter_response_rate > 0.6`` else 1.0),
* a technical-activity modifier (1.15 if ``github_activity_score > 50``; 1.0 if
  ``github_activity_score`` is the ``-1`` sentinel or in ``[0, 50]``),
* a staleness modifier (0.8 if ``last_active_date`` is more than 180 days before
  the evaluation date, 1.0 otherwise; ``None`` is neutral 1.0),

then normalized by ``MAX_RAW_PRODUCT = 1.2 * 1.15 = 1.38`` and clamped to
``[0.0, 1.0]``.
"""

from datetime import date, timedelta
from typing import Optional

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ranking.scorers.behavioral import MAX_RAW_PRODUCT, BehavioralSignalEvaluator
from tests.property._factories import EVAL_DATE, build_candidate


def _make_candidate(
    recruiter_response_rate: float,
    github_activity_score: float,
    last_active_date: Optional[date],
):
    """Build a valid candidate then override the three behavioral signal fields.

    Mutating the returned candidate's ``redrob_signals`` is the simplest, safest
    way to inject custom behavioral inputs without disturbing existing factory
    defaults used by the honeypot tests.
    """
    candidate = build_candidate()
    candidate.redrob_signals.recruiter_response_rate = recruiter_response_rate
    candidate.redrob_signals.github_activity_score = github_activity_score
    candidate.redrob_signals.last_active_date = last_active_date
    return candidate


def _expected_score(
    recruiter_response_rate: float,
    github_activity_score: float,
    last_active_date: Optional[date],
    eval_date: date,
) -> float:
    engagement = 1.2 if recruiter_response_rate > 0.6 else 1.0
    technical = 1.15 if github_activity_score > 50 else 1.0
    if last_active_date is None:
        staleness = 1.0
    else:
        staleness = 0.8 if (eval_date - last_active_date).days > 180 else 1.0
    raw = 1.0 * engagement * technical * staleness
    normalized = raw / MAX_RAW_PRODUCT
    return min(1.0, max(0.0, normalized))


# github_activity_score: the -1 "no GitHub" sentinel plus the [0, 100] band.
_github_scores = st.one_of(
    st.just(-1.0),
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)

# last_active_date: EVAL_DATE minus 0..400 days, plus the None (unknown) case.
_last_active_dates = st.one_of(
    st.none(),
    st.integers(min_value=0, max_value=400).map(lambda d: EVAL_DATE - timedelta(days=d)),
)


@given(
    recruiter_response_rate=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    github_activity_score=_github_scores,
    last_active_date=_last_active_dates,
)
def test_behavioral_score_multiplicative_combination(
    recruiter_response_rate, github_activity_score, last_active_date
):
    """The score equals the independently-computed normalized product, in [0, 1]."""
    evaluator = BehavioralSignalEvaluator()
    candidate = _make_candidate(
        recruiter_response_rate, github_activity_score, last_active_date
    )

    result = evaluator.score(candidate, EVAL_DATE)
    expected = _expected_score(
        recruiter_response_rate, github_activity_score, last_active_date, EVAL_DATE
    )

    assert result == pytest.approx(expected)
    assert 0.0 <= result <= 1.0


# --- Explicit boundary checks -------------------------------------------------


def test_engagement_boundary_at_0_6():
    """response_rate exactly 0.6 is neutral (1.0); strictly above triggers 1.2."""
    evaluator = BehavioralSignalEvaluator()
    assert evaluator.engagement_modifier(0.6) == 1.0
    assert evaluator.engagement_modifier(0.6000001) == 1.2


def test_technical_boundary_at_50_vs_51():
    """github exactly 50 is neutral (1.0); above 50 triggers 1.15."""
    evaluator = BehavioralSignalEvaluator()
    assert evaluator.technical_activity_modifier(50) == 1.0
    assert evaluator.technical_activity_modifier(51) == 1.15
    # The -1 "no GitHub linked" sentinel is neutral, never a penalty.
    assert evaluator.technical_activity_modifier(-1) == 1.0


def test_staleness_boundary_at_180_vs_181_days():
    """last_active exactly 180 days ago is neutral (1.0); 181 days triggers 0.8."""
    evaluator = BehavioralSignalEvaluator()
    assert (
        evaluator.staleness_modifier(EVAL_DATE - timedelta(days=180), EVAL_DATE) == 1.0
    )
    assert (
        evaluator.staleness_modifier(EVAL_DATE - timedelta(days=181), EVAL_DATE) == 0.8
    )
    # None (unknown recency) is neutral, never stale.
    assert evaluator.staleness_modifier(None, EVAL_DATE) == 1.0


def test_score_best_case_is_one():
    """Best case (engagement + technical, recent) normalizes to exactly 1.0."""
    evaluator = BehavioralSignalEvaluator()
    candidate = _make_candidate(
        recruiter_response_rate=1.0,
        github_activity_score=100.0,
        last_active_date=EVAL_DATE,
    )
    assert evaluator.score(candidate, EVAL_DATE) == pytest.approx(1.0)
