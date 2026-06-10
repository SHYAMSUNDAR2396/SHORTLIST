"""Shared test factories for HoneypotDetector tests.

Builds minimal, *valid* :class:`CandidateProfile` objects with sensible defaults
so that each honeypot rule can be exercised in isolation. The defaults are
chosen so that *none* of the four honeypot rules fire unless a test explicitly
provides anomalous data:

- A single career entry starting well before ``EVAL_DATE`` with a
  ``duration_months`` consistent with its date span (no Rule 1 anomaly).
- ``years_of_experience`` consistent with the career span (no Rule 2 anomaly).
- A small skill ``duration_months`` (no Rule 3 anomaly).
- Fewer than 10 expert skills (no Rule 4 anomaly).

Tests override only the fields relevant to the rule under test via
``build_candidate(...)`` and the smaller ``career_entry`` / ``skill`` builders.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from ranking.models import (
    CandidateProfile,
    CareerEntry,
    EducationEntry,
    LanguageEntry,
    ProfileData,
    RedrobSignals,
    SkillEntry,
)

# Fixed evaluation date threaded into the detector for all tests.
EVAL_DATE: date = date(2026, 1, 1)


def make_profile(years_of_experience: float = 5.0) -> ProfileData:
    """A minimal :class:`ProfileData` with an overridable experience value."""
    return ProfileData(
        anonymized_name="Candidate",
        headline="Senior AI Engineer",
        summary="Summary.",
        location="Pune",
        country="India",
        years_of_experience=years_of_experience,
        current_title="Senior AI Engineer",
        current_company="ExampleCo",
        current_company_size="51-200",
        current_industry="Technology",
    )


def make_redrob_signals() -> RedrobSignals:
    """A minimal, valid :class:`RedrobSignals` (values irrelevant to honeypot)."""
    return RedrobSignals(
        profile_completeness_score=1.0,
        signup_date=date(2020, 1, 1),
        last_active_date=EVAL_DATE,
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


def career_entry(
    start_date: Optional[date],
    end_date: Optional[date] = None,
    duration_months: int = 0,
    is_current: bool = False,
    company: str = "ExampleCo",
    title: str = "Engineer",
    company_size: str = "51-200",
    description: str = "",
    industry: str = "Technology",
) -> CareerEntry:
    """Build a :class:`CareerEntry` with explicit dates/duration.

    ``company_size``, ``description``, and ``industry`` are overridable so the
    same builder can drive career-scorer tests (product-company detection,
    production/research keyword weighting) without constructing
    :class:`CareerEntry` by hand. Defaults preserve the prior behavior so the
    honeypot tests that only set dates/duration are unaffected.
    """
    return CareerEntry(
        company=company,
        title=title,
        start_date=start_date,
        end_date=end_date,
        duration_months=duration_months,
        is_current=is_current,
        industry=industry,
        company_size=company_size,
        description=description,
    )


def skill(
    name: str = "python",
    proficiency: str = "intermediate",
    endorsements: int = 0,
    duration_months: int = 0,
) -> SkillEntry:
    """Build a :class:`SkillEntry`."""
    return SkillEntry(
        name=name,
        proficiency=proficiency,
        endorsements=endorsements,
        duration_months=duration_months,
    )


def _default_career_history() -> List[CareerEntry]:
    """A single benign role: ~60 months ending at EVAL_DATE, no Rule 1 anomaly.

    Starts 2021-01 (60 months before 2026-01) with duration_months=60, so
    ``duration_months - months(start->eval) == 0`` (Rule 1 safe), and the span
    is 60 months which equals years_of_experience(5.0)*12 (Rule 2 safe).
    """
    return [
        career_entry(
            start_date=date(2021, 1, 1),
            end_date=None,
            duration_months=60,
            is_current=True,
        )
    ]


def build_candidate(
    candidate_id: str = "CAND_0000001",
    *,
    career_history: Optional[List[CareerEntry]] = None,
    skills: Optional[List[SkillEntry]] = None,
    years_of_experience: float = 5.0,
) -> CandidateProfile:
    """Build a minimal valid :class:`CandidateProfile`.

    With all defaults, none of the four honeypot rules fire. Override
    ``career_history``, ``skills``, and ``years_of_experience`` to target a
    specific rule.
    """
    if career_history is None:
        career_history = _default_career_history()
    if skills is None:
        skills = [skill()]

    return CandidateProfile(
        candidate_id=candidate_id,
        profile=make_profile(years_of_experience=years_of_experience),
        career_history=career_history,
        education=[
            EducationEntry(
                institution="University",
                degree="B.Tech",
                field_of_study="Computer Science",
                start_year=2013,
                end_year=2017,
            )
        ],
        skills=skills,
        certifications=[],
        languages=[LanguageEntry(language="English", proficiency="native")],
        redrob_signals=make_redrob_signals(),
    )
