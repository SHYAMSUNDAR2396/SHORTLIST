"""Unit tests for :class:`ranking.scorers.career.CareerAnalyzer` boundaries.

Task 8.6 (Req 4.2): the consulting-heavy zeroing rule fires only when the
consulting share is *strictly* greater than 80%. These example-based tests pin
the boundary at exactly 80% (not zeroed) and just above it (zeroed).
"""

from datetime import date

from ranking.scorers.career import CareerAnalyzer
from tests.property._factories import build_candidate, career_entry

_START = date(2020, 1, 1)
_PROD_DESC = "deployed embeddings and vector search in production"


def _role(company: str, months: int, *, company_size: str = "51-200"):
    return career_entry(
        start_date=_START,
        duration_months=months,
        company=company,
        title="ML Engineer",
        company_size=company_size,
        description=_PROD_DESC,
    )


def test_consulting_share_exactly_80_percent_not_zeroed():
    """Exactly 80% consulting duration must NOT zero the score (rule is > 80%)."""
    # 80 months consulting out of 100 total => share == 0.80 exactly.
    candidate = build_candidate(
        career_history=[
            _role("TCS", 80),
            _role("StartupHub", 20),
        ]
    )
    score = CareerAnalyzer().score(candidate)

    assert score > 0.0


def test_consulting_share_above_80_percent_zeroed():
    """Just above 80% (81/100) consulting duration zeroes the score."""
    candidate = build_candidate(
        career_history=[
            _role("TCS", 81),
            _role("StartupHub", 19),
        ]
    )
    score = CareerAnalyzer().score(candidate)

    assert score == 0.0
