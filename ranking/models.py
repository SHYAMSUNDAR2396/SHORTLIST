"""Core data models for the Candidate Ranking System.

These dataclasses define the structured representation of candidate profiles
parsed from the input JSONL, the per-candidate scoring results, and the final
ranked output rows. They mirror the candidate schema and the design document's
"Data Models" section.

Notes on graceful defaults:
- ``CareerEntry.end_date`` is ``Optional`` to represent current/ongoing roles.
- ``CareerEntry.duration_months`` defaults to ``0`` (negative/absent durations
  are treated as 0 by the loader).
- ``EducationEntry.tier`` defaults to ``"unknown"`` and ``grade`` to ``None``.
- ``SkillEntry.duration_months`` defaults to ``0``.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


@dataclass
class ProfileData:
    """Top-level profile attributes for a candidate."""

    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str


@dataclass
class CareerEntry:
    """A single role in the candidate's career history."""

    company: str
    title: str
    start_date: date
    end_date: Optional[date]
    duration_months: int = 0
    is_current: bool = False
    industry: str = ""
    company_size: str = ""
    description: str = ""


@dataclass
class EducationEntry:
    """A single education record."""

    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: str = "unknown"


@dataclass
class SkillEntry:
    """A single skill with proficiency and endorsement metadata."""

    name: str
    proficiency: str  # beginner | intermediate | advanced | expert
    endorsements: int
    duration_months: int = 0


@dataclass
class CertificationEntry:
    """A single professional certification."""

    name: str
    issuer: str
    year: int


@dataclass
class LanguageEntry:
    """A single language proficiency record."""

    language: str
    proficiency: str


@dataclass
class RedrobSignals:
    """Simulated platform activity and engagement signals."""

    profile_completeness_score: float
    signup_date: date
    last_active_date: date
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: Dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: Dict[str, float]
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass
class CandidateProfile:
    """A fully parsed candidate profile."""

    candidate_id: str
    profile: ProfileData
    career_history: List[CareerEntry]
    education: List[EducationEntry]
    skills: List[SkillEntry]
    certifications: List[CertificationEntry]
    languages: List[LanguageEntry]
    redrob_signals: RedrobSignals


@dataclass
class LoadStats:
    """Statistics returned by the data loader."""

    total_parsed: int = 0
    total_skipped: int = 0
    json_errors: int = 0
    validation_errors: int = 0


@dataclass
class ScoredCandidate:
    """Per-candidate component and composite scores."""

    candidate_id: str
    skill_score: float
    career_score: float
    experience_score: float
    behavioral_score: float
    education_score: float
    location_work_mode_score: float
    composite_score: float
    penalty_multiplier: float
    disqualification_reason: Optional[str]
    final_score: float  # composite × penalty


@dataclass
class RankedCandidate:
    """A single row in the final ranked submission."""

    candidate_id: str
    rank: int
    score: float  # final_score rounded to 4 decimal places
    reasoning: str
