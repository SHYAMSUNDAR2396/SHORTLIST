"""Unit tests for the CV/speech/robotics disqualifier boundary (Req 6.2).

Task 14.3: exercise the exact ">70% of total skill duration" boundary of the
``cv_speech_robotics`` criterion in :class:`ranking.disqualifier.DisqualifierFilter`.

The rule (Req 6.2) triggers only when CV/speech/robotics skill duration is
*strictly greater than* 70% of total skill duration AND there is no NLP/IR
skill with ``duration_months >= 1``. These tests pin three behaviors:

- Exactly 70% does NOT trigger (boundary is ``>``, not ``>=``).
- Just above 70% (71%) WITH no NLP/IR skill DOES trigger.
- A high CV share is suppressed when an NLP/IR skill (>= 1 month) is present.
"""

from __future__ import annotations

from ranking.disqualifier import CRITERION_CV_SPEECH_ROBOTICS, DisqualifierFilter
from tests.property._factories import EVAL_DATE, build_candidate, skill


def _triggers_cv(candidate) -> bool:
    """True iff the cv_speech_robotics criterion fires for the candidate."""
    triggered = DisqualifierFilter().triggered_criteria(candidate, EVAL_DATE)
    return CRITERION_CV_SPEECH_ROBOTICS in {name for name, _ in triggered}


def test_cv_share_exactly_70_percent_does_not_trigger():
    """Exactly 70% CV/speech duration must NOT trigger (rule is strictly > 70%)."""
    # 70 CV months out of 100 total => share == 0.70 (not > 0.70).
    candidate = build_candidate(
        skills=[
            skill(name="computer vision", duration_months=70),
            skill(name="python", duration_months=30),
        ]
    )
    assert _triggers_cv(candidate) is False


def test_cv_share_just_above_70_percent_triggers():
    """71% CV/speech duration with no NLP/IR skill triggers the criterion."""
    # 71 CV months out of 100 total => share == 0.71 (> 0.70).
    candidate = build_candidate(
        skills=[
            skill(name="computer vision", duration_months=71),
            skill(name="python", duration_months=29),
        ]
    )
    assert _triggers_cv(candidate) is True


def test_nlp_skill_prevents_trigger_even_with_high_cv_share():
    """An NLP/IR skill (>= 1 month) suppresses the trigger despite a high CV share."""
    # 90% CV share, but a 10-month NLP skill is present => no trigger.
    candidate = build_candidate(
        skills=[
            skill(name="computer vision", duration_months=90),
            skill(name="nlp", duration_months=10),
        ]
    )
    assert _triggers_cv(candidate) is False
