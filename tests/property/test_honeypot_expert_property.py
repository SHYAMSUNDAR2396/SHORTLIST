# Feature: candidate-ranking-system, Property 5: Honeypot expert-skills-without-endorsements
"""Property 5: Honeypot expert-skills-without-endorsements.

**Validates: Requirements 2.4**

The detector SHALL flag the candidate if and only if the candidate has 10 or
more skills at "expert" proficiency AND the sum of endorsements across those
expert-level skills equals 0.
"""

from hypothesis import given
from hypothesis import strategies as st

from ranking.honeypot import (
    EXPERT_SKILL_COUNT_THRESHOLD,
    HoneypotDetector,
)
from tests.property._factories import build_candidate, skill

# Proficiency labels: include "expert" plus a few non-expert ones (noise).
_proficiency = st.sampled_from(
    ["expert", "Expert", "EXPERT", "advanced", "intermediate", "beginner"]
)


@given(
    specs=st.lists(
        st.tuples(_proficiency, st.integers(min_value=0, max_value=50)),
        min_size=0,
        max_size=20,
    )
)
def test_expert_endorsement_rule_matches_predicate(specs):
    detector = HoneypotDetector()

    skills = [
        skill(name="s", proficiency=prof, endorsements=endo)
        for prof, endo in specs
    ]

    expert = [
        (prof, endo) for prof, endo in specs if prof.strip().lower() == "expert"
    ]
    expert_count = len(expert)
    expert_endorsements = sum(endo for _, endo in expert)
    expected = (
        expert_count >= EXPERT_SKILL_COUNT_THRESHOLD and expert_endorsements == 0
    )

    candidate = build_candidate(skills=skills)

    assert detector._violates_expert_endorsement_rule(candidate) is expected


@given(
    expert_count=st.integers(min_value=8, max_value=14),
    endorsement_total=st.integers(min_value=0, max_value=5),
)
def test_expert_endorsement_count_and_endorsement_boundary(
    expert_count, endorsement_total
):
    """Flagged iff count >= 10 AND total endorsements == 0."""
    detector = HoneypotDetector()

    skills = [skill(proficiency="expert", endorsements=0) for _ in range(expert_count)]
    # Distribute the endorsement total onto the first expert skill (if any).
    if expert_count > 0 and endorsement_total > 0:
        skills[0] = skill(proficiency="expert", endorsements=endorsement_total)

    candidate = build_candidate(skills=skills)

    expected = (
        expert_count >= EXPERT_SKILL_COUNT_THRESHOLD and endorsement_total == 0
    )
    assert detector._violates_expert_endorsement_rule(candidate) is expected
