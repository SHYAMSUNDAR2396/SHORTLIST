"""Unit tests for HoneypotDetector boundary thresholds and detect() partition.

Covers Requirements 2.1, 2.2, 2.3, 2.4 (boundary values at exactly the +24 /
+36 / +12 month thresholds and the 10-expert-skill threshold) and the detect()
clean/flagged partition behavior (Req 2.5).
"""

from datetime import date

from ranking.honeypot import HoneypotDetector, months_between
from tests.property._factories import (
    EVAL_DATE,
    build_candidate,
    career_entry,
    skill,
)


def _date_months_before(n: int) -> date:
    total = EVAL_DATE.year * 12 + (EVAL_DATE.month - 1) - n
    year, month = divmod(total, 12)
    return date(year, month + 1, 1)


# ---------------------------------------------------------------------------
# Rule 1 (Req 2.1): duration vs date span, threshold = +24 months
# ---------------------------------------------------------------------------
def test_duration_exactly_plus_24_not_flagged():
    detector = HoneypotDetector()
    start = _date_months_before(60)
    span = months_between(start, EVAL_DATE)
    candidate = build_candidate(
        career_history=[career_entry(start_date=start, duration_months=span + 24)],
        years_of_experience=span / 12.0,
    )
    assert detector._violates_duration_rule(candidate, EVAL_DATE) is False


def test_duration_plus_25_flagged():
    detector = HoneypotDetector()
    start = _date_months_before(60)
    span = months_between(start, EVAL_DATE)
    candidate = build_candidate(
        career_history=[career_entry(start_date=start, duration_months=span + 25)],
        years_of_experience=span / 12.0,
    )
    assert detector._violates_duration_rule(candidate, EVAL_DATE) is True


# ---------------------------------------------------------------------------
# Rule 2 (Req 2.2): experience-span mismatch, threshold = 36 months
# ---------------------------------------------------------------------------
def test_experience_span_exactly_36_not_flagged():
    detector = HoneypotDetector()
    # span = 60 months; experience = 96 months (8.0 yrs) => diff = 36 exactly.
    start = _date_months_before(60)
    entry = career_entry(start_date=start, end_date=None, is_current=True)
    candidate = build_candidate(career_history=[entry], years_of_experience=8.0)
    assert detector._violates_experience_span_rule(candidate, EVAL_DATE) is False


def test_experience_span_37_flagged():
    detector = HoneypotDetector()
    # span = 60 months; experience = 97 months => diff = 37 (> 36) => flagged.
    start = _date_months_before(60)
    entry = career_entry(start_date=start, end_date=None, is_current=True)
    candidate = build_candidate(
        career_history=[entry], years_of_experience=97.0 / 12.0
    )
    assert detector._violates_experience_span_rule(candidate, EVAL_DATE) is True


# ---------------------------------------------------------------------------
# Rule 3 (Req 2.3): skill duration vs career span, threshold = +12 months
# ---------------------------------------------------------------------------
def test_skill_duration_exactly_plus_12_not_flagged():
    detector = HoneypotDetector()
    start = _date_months_before(60)
    span = months_between(start, EVAL_DATE)
    entry = career_entry(
        start_date=start, end_date=None, duration_months=span, is_current=True
    )
    candidate = build_candidate(
        career_history=[entry],
        skills=[skill(duration_months=span + 12)],
        years_of_experience=span / 12.0,
    )
    assert detector._violates_skill_duration_rule(candidate, EVAL_DATE) is False


def test_skill_duration_plus_13_flagged():
    detector = HoneypotDetector()
    start = _date_months_before(60)
    span = months_between(start, EVAL_DATE)
    entry = career_entry(
        start_date=start, end_date=None, duration_months=span, is_current=True
    )
    candidate = build_candidate(
        career_history=[entry],
        skills=[skill(duration_months=span + 13)],
        years_of_experience=span / 12.0,
    )
    assert detector._violates_skill_duration_rule(candidate, EVAL_DATE) is True


# ---------------------------------------------------------------------------
# Rule 4 (Req 2.4): expert skills without endorsements
# ---------------------------------------------------------------------------
def test_exactly_10_expert_zero_endorsements_flagged():
    detector = HoneypotDetector()
    skills = [skill(proficiency="expert", endorsements=0) for _ in range(10)]
    candidate = build_candidate(skills=skills)
    assert detector._violates_expert_endorsement_rule(candidate) is True


def test_10_expert_with_one_endorsement_not_flagged():
    detector = HoneypotDetector()
    skills = [skill(proficiency="expert", endorsements=0) for _ in range(9)]
    skills.append(skill(proficiency="expert", endorsements=1))
    candidate = build_candidate(skills=skills)
    assert detector._violates_expert_endorsement_rule(candidate) is False


def test_9_expert_zero_endorsements_not_flagged():
    detector = HoneypotDetector()
    skills = [skill(proficiency="expert", endorsements=0) for _ in range(9)]
    candidate = build_candidate(skills=skills)
    assert detector._violates_expert_endorsement_rule(candidate) is False


# ---------------------------------------------------------------------------
# detect() partitioning (Req 2.5)
# ---------------------------------------------------------------------------
def test_clean_candidate_with_defaults_not_flagged():
    detector = HoneypotDetector()
    candidate = build_candidate(candidate_id="CAND_0000001")
    assert detector.is_honeypot(candidate, EVAL_DATE) is False


def test_detect_partitions_clean_and_flagged():
    detector = HoneypotDetector()

    clean = build_candidate(candidate_id="CAND_0000001")

    # Flagged via Rule 4 (10 expert skills, 0 endorsements).
    flagged = build_candidate(
        candidate_id="CAND_0000002",
        skills=[skill(proficiency="expert", endorsements=0) for _ in range(10)],
    )

    clean_candidates, flagged_ids = detector.detect([clean, flagged], EVAL_DATE)

    assert [c.candidate_id for c in clean_candidates] == ["CAND_0000001"]
    assert flagged_ids == {"CAND_0000002"}


def test_detect_preserves_input_order_for_clean():
    detector = HoneypotDetector()
    c1 = build_candidate(candidate_id="CAND_0000001")
    c2 = build_candidate(candidate_id="CAND_0000002")
    c3 = build_candidate(candidate_id="CAND_0000003")

    clean_candidates, flagged_ids = detector.detect([c1, c2, c3], EVAL_DATE)

    assert [c.candidate_id for c in clean_candidates] == [
        "CAND_0000001",
        "CAND_0000002",
        "CAND_0000003",
    ]
    assert flagged_ids == set()
