# Feature: candidate-ranking-system, Property 18: Education score formula and max selection
"""Property 18: Education score formula and max selection.

**Validates: Requirements 8.6, 8.7**

For any single education entry::

    score_entry(entry) == clamp(
        0.4 * tier_value(entry.tier)
        + 0.3 * degree_to_level(entry.degree)
        + 0.3 * field_relevance(entry.field_of_study)
    )

For a candidate with multiple entries, ``EducationScorer().score`` is the
maximum per-entry score; an empty education list defaults to 0.25. Every result
lies in ``[0.0, 1.0]``.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ranking.constants import (
    EDUCATION_DEGREE_WEIGHT,
    EDUCATION_FIELD_WEIGHT,
    EDUCATION_TIER_WEIGHT,
    degree_to_level,
    field_relevance,
    tier_value,
)
from ranking.models import EducationEntry
from ranking.scorers.education import (
    EMPTY_EDUCATION_DEFAULT,
    EducationScorer,
    score_entry,
)
from tests.property._factories import build_candidate

# Input vocabularies spanning every relevant branch of the mappings.
TIERS = ["tier_1", "tier_2", "tier_3", "tier_4", "unknown"]
DEGREES = ["Ph.D", "M.Tech", "M.Sc", "B.Tech", "B.E.", "B.Sc"]
# Mix of relevant (AI/ML/CS), adjacent (math/stats/etc.), and unrelated fields.
FIELDS = [
    "Computer Science",
    "Machine Learning",
    "Artificial Intelligence",
    "Data Science",
    "Mathematics",
    "Statistics",
    "Electronics",
    "Information Technology",
    "Physics",
    "History",
    "Biology",
    "Fine Arts",
]


def _expected_entry_score(tier: str, degree: str, field: str) -> float:
    """Independently recompute the per-entry formula, clamped to [0, 1]."""
    raw = (
        EDUCATION_TIER_WEIGHT * tier_value(tier)
        + EDUCATION_DEGREE_WEIGHT * degree_to_level(degree)
        + EDUCATION_FIELD_WEIGHT * field_relevance(field)
    )
    return max(0.0, min(1.0, raw))


def _make_entry(tier: str, degree: str, field: str) -> EducationEntry:
    return EducationEntry(
        institution="Some University",
        degree=degree,
        field_of_study=field,
        start_year=2013,
        end_year=2017,
        tier=tier,
    )


entries_strategy = st.lists(
    st.tuples(
        st.sampled_from(TIERS),
        st.sampled_from(DEGREES),
        st.sampled_from(FIELDS),
    ),
    min_size=0,
    max_size=5,
)


@given(tier=st.sampled_from(TIERS), degree=st.sampled_from(DEGREES), field=st.sampled_from(FIELDS))
def test_score_entry_matches_formula(tier, degree, field):
    """score_entry == 0.4*tier + 0.3*degree + 0.3*field, clamped to [0, 1]."""
    entry = _make_entry(tier, degree, field)
    result = score_entry(entry)

    assert result == pytest.approx(_expected_entry_score(tier, degree, field))
    assert 0.0 <= result <= 1.0


@given(specs=entries_strategy)
def test_candidate_score_is_max_over_entries(specs):
    """EducationScorer().score == max per-entry score; empty -> 0.25."""
    scorer = EducationScorer()
    candidate = build_candidate()
    candidate.education = [_make_entry(t, d, f) for (t, d, f) in specs]

    result = scorer.score(candidate)

    if not specs:
        assert result == pytest.approx(EMPTY_EDUCATION_DEFAULT)
    else:
        expected_max = max(_expected_entry_score(t, d, f) for (t, d, f) in specs)
        assert result == pytest.approx(expected_max)
        # The candidate score equals one of the entry scores (the maximum).
        assert result == pytest.approx(
            max(score_entry(e) for e in candidate.education)
        )

    assert 0.0 <= result <= 1.0
