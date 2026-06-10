"""Property-based test for :class:`ranking.reasoning.ReasoningGenerator`.

Covers design correctness Property 20 (task 17.2): the generated reasoning
string is well-formed (length, single-line), contains the required content
(current title, years of experience, top-contributing component label), and
references at least one concrete profile attribute. Also asserts determinism.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ranking.models import (
    CandidateProfile,
    CareerEntry,
    EducationEntry,
    LanguageEntry,
    ProfileData,
    RedrobSignals,
    ScoredCandidate,
    SkillEntry,
)
from ranking.reasoning import (
    COMPONENT_LABELS,
    MAX_ATTRIBUTE_CHARS,
    MAX_LENGTH,
    MIN_LENGTH,
    ReasoningGenerator,
    _sanitize,
    top_contributing_component,
)


# ---------------------------------------------------------------------------
# Helper: build a ScoredCandidate with controlled component scores.
# ---------------------------------------------------------------------------
def make_scored(
    *,
    skill_score: float = 0.0,
    career_score: float = 0.0,
    experience_score: float = 0.0,
    behavioral_score: float = 0.0,
    education_score: float = 0.0,
    location_work_mode_score: float = 0.0,
    candidate_id: str = "CAND_0000001",
) -> ScoredCandidate:
    """Construct a :class:`ScoredCandidate` with the given component scores.

    Composite/penalty/final fields use neutral defaults — the reasoning
    generator only consults the six component scores to pick the top factor.
    """
    return ScoredCandidate(
        candidate_id=candidate_id,
        skill_score=skill_score,
        career_score=career_score,
        experience_score=experience_score,
        behavioral_score=behavioral_score,
        education_score=education_score,
        location_work_mode_score=location_work_mode_score,
        composite_score=0.5,
        penalty_multiplier=1.0,
        disqualification_reason=None,
        final_score=0.5,
    )


def _minimal_redrob() -> RedrobSignals:
    return RedrobSignals(
        profile_completeness_score=1.0,
        signup_date=None,  # not consulted by the reasoning generator
        last_active_date=None,
        open_to_work_flag=True,
        profile_views_received_30d=0,
        applications_submitted_30d=0,
        recruiter_response_rate=0.0,
        avg_response_time_hours=0.0,
        skill_assessment_scores={},
        connection_count=0,
        endorsements_received=0,
        notice_period_days=0,
        expected_salary_range_inr_lpa={"min": 0.0, "max": 0.0},
        preferred_work_mode="hybrid",
        willing_to_relocate=True,
        github_activity_score=0.0,
        search_appearance_30d=0,
        saved_by_recruiters_30d=0,
        interview_completion_rate=0.0,
        offer_acceptance_rate=0.0,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )


# ---------------------------------------------------------------------------
# Generators.
# ---------------------------------------------------------------------------
# Skill names: a mix that map to required groups and pure noise. Kept <= 40
# chars so neither truncation nor whitespace-collapse alters the substring we
# assert on.
_MATCHING_SKILLS = ["python", "pinecone", "embeddings", "ndcg", "faiss", "pytorch"]
_NOISE_SKILLS = ["cooking", "excel", "java", "leadership", "welding"]
_SKILL_NAMES = _MATCHING_SKILLS + _NOISE_SKILLS

_PROFICIENCIES = ["beginner", "intermediate", "advanced", "expert"]

# Short company / career-title pools (<= 40 chars), including empty strings to
# exercise the generator's fall-through selection logic.
_COMPANIES = ["ExampleCo", "Acme", "OpenAI", "Globex", ""]
_CAREER_TITLES = ["Engineer", "ML Engineer", "Research Scientist", "Manager", ""]

# current_title pool: normal titles, empty (-> "Candidate" fallback), and very
# long titles to exercise truncation.
_TITLES = [
    "Senior AI Engineer",
    "ML Engineer",
    "Data Scientist",
    "",
    "Senior Principal Distinguished Staff Machine Learning Engineer and Architect Lead",
    "X" * 250,
]


@st.composite
def _skill_entries(draw):
    n = draw(st.integers(min_value=0, max_value=5))
    skills = []
    for _ in range(n):
        skills.append(
            SkillEntry(
                name=draw(st.sampled_from(_SKILL_NAMES)),
                proficiency=draw(st.sampled_from(_PROFICIENCIES)),
                endorsements=draw(st.integers(min_value=0, max_value=50)),
                duration_months=draw(st.integers(min_value=0, max_value=120)),
            )
        )
    return skills


@st.composite
def _career_entries(draw):
    n = draw(st.integers(min_value=0, max_value=4))
    entries = []
    for _ in range(n):
        entries.append(
            CareerEntry(
                company=draw(st.sampled_from(_COMPANIES)),
                title=draw(st.sampled_from(_CAREER_TITLES)),
                start_date=None,
                end_date=None,
                duration_months=draw(st.integers(min_value=0, max_value=120)),
                is_current=False,
                industry="Technology",
                company_size="51-200",
                description="",
            )
        )
    return entries


@st.composite
def _candidates(draw):
    profile = ProfileData(
        anonymized_name="Candidate",
        headline="Engineer",
        summary="Summary.",
        location="Pune",
        country="India",
        years_of_experience=draw(
            st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False)
        ),
        current_title=draw(st.sampled_from(_TITLES)),
        current_company=draw(st.sampled_from(_COMPANIES)),
        current_company_size="51-200",
        current_industry="Technology",
    )
    return CandidateProfile(
        candidate_id="CAND_0000001",
        profile=profile,
        career_history=draw(_career_entries()),
        education=[
            EducationEntry(
                institution="University",
                degree="B.Tech",
                field_of_study="Computer Science",
                start_year=2013,
                end_year=2017,
            )
        ],
        skills=draw(_skill_entries()),
        certifications=[],
        languages=[LanguageEntry(language="English", proficiency="native")],
        redrob_signals=_minimal_redrob(),
    )


# Component scores span [-0.5, 1.5] so clamping is exercised and different
# components can become the top factor across examples.
_COMPONENT_SCORE = st.floats(
    min_value=-0.5, max_value=1.5, allow_nan=False, allow_infinity=False
)


def _candidate_attribute_substrings(candidate: CandidateProfile) -> list:
    """All concrete profile strings the generator may reference (Req 11.3).

    Includes both the full sanitized value and its 60-char truncation (the
    generator caps the chosen attribute at ``MAX_ATTRIBUTE_CHARS``), plus the
    literal generic fallback.
    """
    attrs: list = []

    def add(raw: str) -> None:
        s = _sanitize(raw)
        if s:
            attrs.append(s)
            attrs.append(s[:MAX_ATTRIBUTE_CHARS].rstrip())

    for sk in candidate.skills:
        add(sk.name)
    add(candidate.profile.current_company)
    for entry in candidate.career_history:
        add(entry.company)
        add(entry.title)
    add(candidate.profile.current_title)
    attrs.append("AI/ML profile")
    return attrs


# Feature: candidate-ranking-system, Property 20: Reasoning format and content
@given(
    candidate=_candidates(),
    skill_score=_COMPONENT_SCORE,
    career_score=_COMPONENT_SCORE,
    experience_score=_COMPONENT_SCORE,
    behavioral_score=_COMPONENT_SCORE,
    education_score=_COMPONENT_SCORE,
    location_work_mode_score=_COMPONENT_SCORE,
)
def test_reasoning_format_and_content(
    candidate,
    skill_score,
    career_score,
    experience_score,
    behavioral_score,
    education_score,
    location_work_mode_score,
):
    """Property 20: reasoning is well-formed, complete, and specific.

    **Validates: Requirements 11.1, 11.2, 11.3, 11.5**
    """
    scored = make_scored(
        skill_score=skill_score,
        career_score=career_score,
        experience_score=experience_score,
        behavioral_score=behavioral_score,
        education_score=education_score,
        location_work_mode_score=location_work_mode_score,
    )
    generator = ReasoningGenerator()
    reasoning = generator.generate(candidate, scored)

    # --- Req 11.2: length between 20 and 200 inclusive. ---
    assert MIN_LENGTH <= len(reasoning) <= MAX_LENGTH

    # --- Req 11.5: single line, no line-break characters. ---
    assert "\n" not in reasoning
    assert "\r" not in reasoning

    # --- Req 11.1: includes the (possibly truncated) current title. ---
    expected_title = _sanitize(candidate.profile.current_title) or "Candidate"
    title_part = expected_title[:60].rstrip()
    assert title_part in reasoning

    # --- Req 11.1: includes YOE formatted to one decimal place. ---
    yoe_str = f"{float(candidate.profile.years_of_experience):.1f}"
    assert yoe_str in reasoning

    # --- Req 11.1: names the top-contributing component's human label. ---
    top_component, _ = top_contributing_component(scored)
    expected_label = COMPONENT_LABELS[top_component]
    assert expected_label in reasoning

    # --- Req 11.3: references at least one concrete profile attribute. ---
    candidate_attrs = _candidate_attribute_substrings(candidate)
    assert any(attr in reasoning for attr in candidate_attrs)

    # --- Determinism: same inputs yield identical output. ---
    assert generator.generate(candidate, scored) == reasoning
