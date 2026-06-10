"""Unit tests for DataLoader edge cases (Task 4.3).

_Requirements: 1.5_
"""

import json

import pytest

from ranking.loader import DataLoader


def _minimal_valid_candidate(candidate_id: str = "CAND_0000001") -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "Engineer",
            "summary": "Summary.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "ML Engineer",
            "current_company": "Acme",
            "current_company_size": "51-200",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "Acme",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": "2023-01-01",
                "duration_months": 36,
                "is_current": False,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Built retrieval systems.",
            }
        ],
        "education": [
            {
                "institution": "IIT",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "9.0 CGPA",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {
                "name": "python",
                "proficiency": "expert",
                "endorsements": 10,
                "duration_months": 48,
            }
        ],
        "redrob_signals": {
            "profile_completeness_score": 0.9,
            "signup_date": "2020-01-01",
            "last_active_date": "2024-01-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 20,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 60.0,
            "search_appearance_30d": 5,
            "saved_by_recruiters_30d": 1,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


def test_empty_file_parses_zero_skips_zero(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    candidates, stats = DataLoader().load(str(path))

    assert candidates == []
    assert stats.total_parsed == 0
    assert stats.total_skipped == 0
    assert stats.json_errors == 0
    assert stats.validation_errors == 0


def test_single_valid_record_parses_one(tmp_path):
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps(_minimal_valid_candidate()) + "\n", encoding="utf-8")

    candidates, stats = DataLoader().load(str(path))

    assert stats.total_parsed == 1
    assert len(candidates) == 1
    assert candidates[0].candidate_id == "CAND_0000001"
    assert stats.total_skipped == 0


def test_file_not_found_exits_non_zero(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(SystemExit) as exc_info:
        DataLoader().load(str(missing))

    # Exit code must be non-zero (Req 1.5).
    code = exc_info.value.code
    assert code is not None
    assert code != 0


def test_malformed_and_missing_field_records_counted(tmp_path):
    valid = json.dumps(_minimal_valid_candidate("CAND_0000001"))
    malformed = "this is not json {"
    missing_field = _minimal_valid_candidate("CAND_0000002")
    del missing_field["skills"]  # drop a required field
    missing = json.dumps(missing_field)

    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join([valid, malformed, missing]) + "\n", encoding="utf-8")

    candidates, stats = DataLoader().load(str(path))

    assert stats.total_parsed == 1
    assert len(candidates) == 1
    assert candidates[0].candidate_id == "CAND_0000001"
    assert stats.json_errors == 1
    assert stats.validation_errors == 1
    assert stats.total_skipped == 2
