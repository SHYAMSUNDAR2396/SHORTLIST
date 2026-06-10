"""Property-based tests for :class:`ranking.scorers.career.CareerAnalyzer`.

Covers design correctness properties 8, 9, 10, and 11 (tasks 8.2-8.5). Each
test builds candidates with a controlled ``career_history`` via the shared
factory and compares the scorer's output against an independent oracle that
re-implements the requirement's documented formula.
"""

from __future__ import annotations

import re
from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ranking.constants import (
    AI_ML_TITLE_TERMS,
    CONSULTING_FIRMS,
    PRODUCT_COMPANY_SIZES,
    TECHNICAL_TITLE_TERMS,
)
from ranking.scorers.career import CareerAnalyzer
from tests.property._factories import build_candidate, career_entry

# A fixed start date; the production/title/consulting scores depend only on
# company_size, duration_months, description, title — not the actual dates.
_START = date(2020, 1, 1)

# Descriptions whose keyword content drives the per-role relevance weight.
_PROD_DESC = "deployed embeddings and vector search in production"  # production kw
_RESEARCH_DESC = "published a paper at a conference"  # research kw, no production
_NEUTRAL_DESC = "general backend work"  # neither keyword set


# ---------------------------------------------------------------------------
# Independent oracle for production_experience_score (Property 8).
# ---------------------------------------------------------------------------
def _clamp(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _relevance_weight(category: str) -> float:
    if category == "prod":
        return 2.0
    if category == "research":
        return 0.3
    return 1.0


def _expected_production(specs) -> float:
    """Re-implement Req 4.1/4.4/4.5/4.7/4.3 independently from ``specs``.

    ``specs`` is a list of ``(company_size, duration_months, category)``.
    """
    months = [d if d > 0 else 0 for (_s, d, _c) in specs]
    total = sum(months)
    if total <= 0:
        return 0.0

    numerator = 0.0
    for (size, _d, cat), m in zip(specs, months):
        if size in PRODUCT_COMPANY_SIZES:
            numerator += _relevance_weight(cat) * m

    score = _clamp(numerator / total)

    # Req 4.7: cap at 0.2 when every role is at a 10001+ company with no
    # production keyword.
    all_large_no_prod = all(s == "10001+" for (s, _d, _c) in specs) and not any(
        c == "prod" for (_s, _d, c) in specs
    )
    if all_large_no_prod:
        score = min(score, 0.2)

    # Req 4.3: job-hopping penalty when >=3 roles each have <18 months.
    short_roles = sum(1 for m in months if 0 <= m < 18)
    if short_roles >= 3:
        score *= 0.5

    return _clamp(score)


def _entries_from_specs(specs):
    desc_for = {"prod": _PROD_DESC, "research": _RESEARCH_DESC, "neutral": _NEUTRAL_DESC}
    return [
        career_entry(
            start_date=_START,
            duration_months=d,
            company="SomeCo",
            title="Engineer",
            company_size=size,
            description=desc_for[cat],
        )
        for (size, d, cat) in specs
    ]


_SIZE = st.sampled_from(["11-50", "51-200", "201-500", "501-1000", "10001+", "1-10"])
_CATEGORY = st.sampled_from(["prod", "research", "neutral"])
_ROLE_SPEC = st.tuples(_SIZE, st.integers(min_value=0, max_value=120), _CATEGORY)


# Feature: candidate-ranking-system, Property 8: Production experience score composition
@given(specs=st.lists(_ROLE_SPEC, min_size=1, max_size=6))
def test_production_experience_score_composition(specs):
    """Property 8: production_experience_score matches the weighted ratio.

    **Validates: Requirements 4.1, 4.4, 4.5, 4.7**
    """
    candidate = build_candidate(career_history=_entries_from_specs(specs))
    actual = CareerAnalyzer().production_experience_score(candidate)

    assert 0.0 <= actual <= 1.0
    assert actual == pytest.approx(_expected_production(specs), abs=1e-9)


# Feature: candidate-ranking-system, Property 8: Production experience score composition
@given(
    specs=st.lists(
        st.tuples(st.integers(min_value=1, max_value=120), st.sampled_from(["research", "neutral"])),
        min_size=1,
        max_size=5,
    )
)
def test_production_score_all_large_enterprise_capped(specs):
    """All-10001+ roles with no production keyword stay within the 0.2 cap.

    **Validates: Requirements 4.7**
    """
    full_specs = [("10001+", d, cat) for (d, cat) in specs]
    candidate = build_candidate(career_history=_entries_from_specs(full_specs))
    actual = CareerAnalyzer().production_experience_score(candidate)

    assert 0.0 <= actual <= 0.2
    assert actual == pytest.approx(_expected_production(full_specs), abs=1e-9)


# ---------------------------------------------------------------------------
# Property 9: Consulting-heavy career zeroing.
# ---------------------------------------------------------------------------
_CONSULTING_NAMES = sorted(CONSULTING_FIRMS)
_NON_CONSULTING = "StartupHub"  # contains no consulting-firm substring


# Feature: candidate-ranking-system, Property 9: Consulting-heavy career zeroing
@given(
    roles=st.lists(
        st.tuples(st.booleans(), st.integers(min_value=1, max_value=120)),
        min_size=1,
        max_size=6,
    ),
    consulting_pick=st.sampled_from(_CONSULTING_NAMES),
)
def test_consulting_heavy_career_zeroed(roles, consulting_pick):
    """Property 9: > 80% consulting duration forces career score to 0.0.

    **Validates: Requirements 4.2**
    """
    entries = []
    total = 0
    consulting_months = 0
    for is_consulting, months in roles:
        total += months
        if is_consulting:
            consulting_months += months
            company = consulting_pick
        else:
            company = _NON_CONSULTING
        entries.append(
            career_entry(
                start_date=_START,
                duration_months=months,
                company=company,
                title="ML Engineer",
                company_size="51-200",
                description=_PROD_DESC,
            )
        )

    candidate = build_candidate(career_history=entries)
    score = CareerAnalyzer().score(candidate)

    share = consulting_months / total
    if share > 0.80:
        assert score == 0.0


# ---------------------------------------------------------------------------
# Property 10: Job-hopping stability penalty.
# ---------------------------------------------------------------------------
def _product_neutral_role(months: int):
    """A product-company, neutral-description role (per-role relevance == 1.0)."""
    return career_entry(
        start_date=_START,
        duration_months=months,
        company="SomeCo",
        title="Engineer",
        company_size="51-200",
        description=_NEUTRAL_DESC,
    )


# Feature: candidate-ranking-system, Property 10: Job-hopping stability penalty
@given(
    base_long=st.lists(st.integers(min_value=18, max_value=60), min_size=1, max_size=3),
    shorts=st.lists(st.integers(min_value=1, max_value=17), min_size=0, max_size=5),
)
def test_job_hopping_halves_production_score(base_long, shorts):
    """Property 10: >=3 short roles (<18 mo) multiply the score by 0.5.

    All roles are product-company with neutral descriptions, so the base
    weighted ratio is exactly 1.0 regardless of count. The only thing that
    changes the production score is the job-hopping penalty, so we can observe
    the halving directly by comparing the candidate with short roles against
    the same base without them.

    **Validates: Requirements 4.3**
    """
    analyzer = CareerAnalyzer()

    base_roles = [_product_neutral_role(m) for m in base_long]
    base_candidate = build_candidate(career_history=base_roles)
    base_score = analyzer.production_experience_score(base_candidate)
    # Base roles are all >=18 months and product/neutral -> ratio 1.0, no hop.
    assert base_score == pytest.approx(1.0, abs=1e-9)

    with_short = base_roles + [_product_neutral_role(m) for m in shorts]
    candidate = build_candidate(career_history=with_short)
    score = analyzer.production_experience_score(candidate)

    if len(shorts) >= 3:
        assert score == pytest.approx(base_score * 0.5, abs=1e-9)
    else:
        assert score == pytest.approx(base_score, abs=1e-9)


# ---------------------------------------------------------------------------
# Property 11: Title relevance bounded weighted average.
# ---------------------------------------------------------------------------
_AI_ML_PATTERNS = {t: re.compile(r"\b" + re.escape(t) + r"\b") for t in AI_ML_TITLE_TERMS}


def _expected_title_weight(title: str) -> float:
    text = title.lower()
    for term in AI_ML_TITLE_TERMS:
        if len(term) <= 3:
            if _AI_ML_PATTERNS[term].search(text):
                return 1.0
        elif term in text:
            return 1.0
    if any(t in text for t in TECHNICAL_TITLE_TERMS):
        return 0.5
    return 0.0


# Curated titles spanning the three relevance buckets per the Req 4.6 rule.
_TITLE_POOL = [
    "ML Engineer",          # 1.0 (ml)
    "AI Lead",              # 1.0 (ai)
    "NLP Engineer",         # 1.0 (nlp)
    "Machine Learning Researcher",  # 1.0 (machine learning)
    "Software Engineer",    # 0.5 (engineer)
    "Data Analyst",         # 0.5 (data/analyst)
    "Tech Lead",            # 0.5 (tech)
    "Accountant",           # 0.0
    "Marketing Manager",    # 0.0
    "Sales Director",       # 0.0
]


# Feature: candidate-ranking-system, Property 11: Title relevance bounded weighted average
@given(
    roles=st.lists(
        st.tuples(st.sampled_from(_TITLE_POOL), st.integers(min_value=1, max_value=120)),
        min_size=1,
        max_size=6,
    )
)
def test_title_relevance_weighted_average(roles):
    """Property 11: title_relevance_score is the duration-weighted average.

    **Validates: Requirements 4.6**
    """
    entries = [
        career_entry(
            start_date=_START,
            duration_months=months,
            company="SomeCo",
            title=title,
            company_size="51-200",
            description=_NEUTRAL_DESC,
        )
        for (title, months) in roles
    ]
    candidate = build_candidate(career_history=entries)
    actual = CareerAnalyzer().title_relevance_score(candidate)

    total = sum(m for (_t, m) in roles)
    expected = sum(_expected_title_weight(t) * m for (t, m) in roles) / total

    assert 0.0 <= actual <= 1.0
    assert actual == pytest.approx(expected, abs=1e-9)
