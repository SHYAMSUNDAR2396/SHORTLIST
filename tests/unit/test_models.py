"""Unit tests for the core data models (Task 2.2).

Verifies dataclass construction, graceful defaults, and field round-tripping.

_Requirements: 1.3, 8.6_
"""

from datetime import date

from ranking.models import (
    CandidateProfile,
    CareerEntry,
    CertificationEntry,
    EducationEntry,
    LanguageEntry,
    LoadStats,
    ProfileData,
    RedrobSignals,
    SkillEntry,
)


def test_career_entry_duration_months_defaults_to_zero():
    entry = CareerEntry(
        company="Acme",
        title="Engineer",
        start_date=date(2020, 1, 1),
        end_date=date(2022, 1, 1),
    )
    assert entry.duration_months == 0
    # Other graceful defaults.
    assert entry.is_current is False
    assert entry.industry == ""
    assert entry.company_size == ""
    assert entry.description == ""


def test_career_entry_end_date_can_be_none():
    entry = CareerEntry(
        company="Acme",
        title="Engineer",
        start_date=date(2020, 1, 1),
        end_date=None,
        is_current=True,
    )
    assert entry.end_date is None
    assert entry.is_current is True


def test_education_entry_tier_and_grade_defaults():
    entry = EducationEntry(
        institution="IIT",
        degree="B.Tech",
        field_of_study="Computer Science",
        start_year=2016,
        end_year=2020,
    )
    assert entry.tier == "unknown"
    assert entry.grade is None


def test_skill_entry_duration_months_defaults_to_zero():
    skill = SkillEntry(name="Python", proficiency="expert", endorsements=10)
    assert skill.duration_months == 0


def test_full_candidate_profile_round_trips():
    profile = ProfileData(
        anonymized_name="Jane Doe",
        headline="Senior AI Engineer",
        summary="Builds retrieval systems.",
        location="Pune",
        country="India",
        years_of_experience=7.5,
        current_title="ML Engineer",
        current_company="Acme",
        current_company_size="51-200",
        current_industry="Software",
    )
    career = [
        CareerEntry(
            company="Acme",
            title="ML Engineer",
            start_date=date(2021, 1, 1),
            end_date=None,
            duration_months=40,
            is_current=True,
            industry="Software",
            company_size="51-200",
            description="Vector search and ranking.",
        )
    ]
    education = [
        EducationEntry(
            institution="IIT",
            degree="M.Tech",
            field_of_study="Computer Science",
            start_year=2014,
            end_year=2016,
            grade="9.1 CGPA",
            tier="tier_1",
        )
    ]
    skills = [
        SkillEntry(name="python", proficiency="expert", endorsements=20, duration_months=60),
        SkillEntry(name="FAISS", proficiency="advanced", endorsements=5),
    ]
    certifications = [CertificationEntry(name="AWS ML", issuer="Amazon", year=2022)]
    languages = [LanguageEntry(language="English", proficiency="native")]
    signals = RedrobSignals(
        profile_completeness_score=0.95,
        signup_date=date(2020, 1, 1),
        last_active_date=date(2024, 5, 1),
        open_to_work_flag=True,
        profile_views_received_30d=120,
        applications_submitted_30d=4,
        recruiter_response_rate=0.7,
        avg_response_time_hours=12.0,
        skill_assessment_scores={"python": 0.9},
        connection_count=500,
        endorsements_received=80,
        notice_period_days=30,
        expected_salary_range_inr_lpa={"min": 30.0, "max": 50.0},
        preferred_work_mode="hybrid",
        willing_to_relocate=True,
        github_activity_score=75.0,
        search_appearance_30d=40,
        saved_by_recruiters_30d=6,
        interview_completion_rate=0.9,
        offer_acceptance_rate=0.5,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )

    candidate = CandidateProfile(
        candidate_id="CAND_0000001",
        profile=profile,
        career_history=career,
        education=education,
        skills=skills,
        certifications=certifications,
        languages=languages,
        redrob_signals=signals,
    )

    assert candidate.candidate_id == "CAND_0000001"
    assert candidate.profile.location == "Pune"
    assert candidate.profile.years_of_experience == 7.5
    assert candidate.career_history[0].duration_months == 40
    assert candidate.career_history[0].end_date is None
    assert candidate.education[0].tier == "tier_1"
    assert candidate.education[0].grade == "9.1 CGPA"
    assert candidate.skills[0].duration_months == 60
    assert candidate.skills[1].duration_months == 0
    assert candidate.certifications[0].issuer == "Amazon"
    assert candidate.languages[0].language == "English"
    assert candidate.redrob_signals.preferred_work_mode == "hybrid"
    assert candidate.redrob_signals.skill_assessment_scores == {"python": 0.9}


def test_load_stats_defaults_to_zero():
    stats = LoadStats()
    assert stats.total_parsed == 0
    assert stats.total_skipped == 0
    assert stats.json_errors == 0
    assert stats.validation_errors == 0
