"""Property-based tests for the streaming JSONL DataLoader (Task 4.2).

# Feature: candidate-ranking-system, Property 1: Parser resilience and count integrity

Property 1: Parser resilience and count integrity.
**Validates: Requirements 1.2, 1.3, 1.4**

For any JSONL input composed of a mix of valid candidate records, malformed
JSON lines, and JSON objects missing one or more required fields, the loader
SHALL parse every valid record, skip every invalid record without terminating,
and return counts where ``total_parsed`` equals the number of valid records and
``total_skipped`` equals the sum of JSON-parse failures and validation failures.
"""

import json
import os
import tempfile

from hypothesis import given
from hypothesis import strategies as st

from ranking.loader import REQUIRED_FIELDS, DataLoader

# The 23 keys RedrobSignals requires (loader uses .get() defaults, so presence
# is not strictly required, but we include them to model a realistic record).
REDROB_KEYS = [
    "profile_completeness_score",
    "signup_date",
    "last_active_date",
    "open_to_work_flag",
    "profile_views_received_30d",
    "applications_submitted_30d",
    "recruiter_response_rate",
    "avg_response_time_hours",
    "skill_assessment_scores",
    "connection_count",
    "endorsements_received",
    "notice_period_days",
    "expected_salary_range_inr_lpa",
    "preferred_work_mode",
    "willing_to_relocate",
    "github_activity_score",
    "search_appearance_30d",
    "saved_by_recruiters_30d",
    "interview_completion_rate",
    "offer_acceptance_rate",
    "verified_email",
    "verified_phone",
    "linkedin_connected",
]


def _minimal_valid_candidate(index: int) -> dict:
    """Emit a minimally-valid candidate JSON dict.

    The nested structure is rich enough for ``DataLoader._build_candidate`` to
    succeed: profile has required keys, career_history has >=1 entry with the
    required keys, redrob_signals has all 23 keys, and dates are YYYY-MM-DD.
    """
    return {
        "candidate_id": f"CAND_{index:07d}",
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
        "redrob_signals": {key: _redrob_value(key) for key in REDROB_KEYS},
    }


def _redrob_value(key: str):
    """Provide a plausible value for a redrob signal key."""
    if key in ("signup_date", "last_active_date"):
        return "2023-06-01"
    if key in ("skill_assessment_scores", "expected_salary_range_inr_lpa"):
        return {}
    if key == "preferred_work_mode":
        return "hybrid"
    if key.startswith(("verified_", "linkedin_", "open_to_", "willing_")):
        return True
    if any(token in key for token in ("rate", "score", "hours", "completeness")):
        return 0.5
    return 1


# Strategy producing one of three "kinds" of JSONL lines paired with a tag.
# tag is one of: "valid", "json_error", "validation_error".


@st.composite
def jsonl_lines(draw):
    """Generate a list of (tag, line_text) tuples covering all three kinds."""
    n = draw(st.integers(min_value=0, max_value=12))
    lines = []
    for i in range(n):
        kind = draw(st.sampled_from(["valid", "json_error", "validation_error"]))
        if kind == "valid":
            record = _minimal_valid_candidate(i)
            lines.append(("valid", json.dumps(record)))
        elif kind == "json_error":
            # Non-JSON text that json.loads will reject. Avoid accidentally
            # producing parseable JSON (e.g. a bare number or "null").
            text = draw(
                st.text(
                    alphabet="abcdefghijk {}[]:,xyz",
                    min_size=1,
                    max_size=20,
                ).filter(_is_unparseable_json)
            )
            lines.append(("json_error", text))
        else:  # validation_error: valid JSON object missing >=1 required field
            record = _minimal_valid_candidate(i)
            drop_count = draw(st.integers(min_value=1, max_value=len(REQUIRED_FIELDS)))
            to_drop = draw(
                st.lists(
                    st.sampled_from(list(REQUIRED_FIELDS)),
                    min_size=drop_count,
                    max_size=drop_count,
                    unique=True,
                )
            )
            for fieldname in to_drop:
                record.pop(fieldname, None)
            lines.append(("validation_error", json.dumps(record)))
    return lines


def _is_unparseable_json(text: str) -> bool:
    """Return True only if ``text`` cannot be parsed as JSON (after strip)."""
    stripped = text.strip()
    if not stripped:
        # Blank lines are silently skipped (not counted), so exclude them.
        return False
    try:
        json.loads(stripped)
        return False
    except (json.JSONDecodeError, ValueError):
        return True


@given(lines=jsonl_lines())
def test_property_1_parser_resilience_and_count_integrity(lines):
    # Feature: candidate-ranking-system, Property 1: Parser resilience and count integrity
    # Use a fresh temp file per generated input (a function-scoped pytest
    # fixture is not reset between @given examples, so we manage it here).
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(line_text for _tag, line_text in lines)
                + ("\n" if lines else "")
            )

        expected_valid = sum(1 for tag, _ in lines if tag == "valid")
        expected_json_errors = sum(1 for tag, _ in lines if tag == "json_error")
        expected_validation_errors = sum(
            1 for tag, _ in lines if tag == "validation_error"
        )

        # The loader must never raise on bad records.
        candidates, stats = DataLoader().load(path)

        # Every valid record is parsed.
        assert stats.total_parsed == expected_valid
        assert len(candidates) == expected_valid

        # Every invalid record is skipped, split into the two failure categories.
        assert stats.json_errors == expected_json_errors
        assert stats.validation_errors == expected_validation_errors
        assert stats.total_skipped == expected_json_errors + expected_validation_errors

        # The parsed candidates carry the ids of the valid records.
        valid_ids = {
            json.loads(line_text)["candidate_id"]
            for tag, line_text in lines
            if tag == "valid"
        }
        assert {c.candidate_id for c in candidates} == valid_ids
    finally:
        os.remove(path)
