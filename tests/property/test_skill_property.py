"""Property-based tests for :class:`ranking.scorers.skill.SkillScorer`.

Covers design correctness properties 6 and 7 (tasks 7.2 and 7.3):

- Property 6 (Req 3.1, 3.5): the skill score is always in [0.0, 1.0] and is
  exactly 0.0 when no skill maps to a required group.
- Property 7 (Req 3.2, 3.3): a relevant skill whose ``duration_months`` exceeds
  the candidate's total career duration (sum of career_history duration_months)
  by more than 6 months has its contribution reduced to 0.25 of normal; within
  the 6-month threshold the normal contribution applies.

Each test builds candidates with controlled skills/career via the shared
factory and (for Property 7) targets ``SkillScorer._skill_contribution``
directly to assert the exact 0.25 multiplier at the boundary.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ranking.constants import (
    PROFICIENCY_FACTOR,
    SKILL_GROUPS,
    group_weight,
    skill_to_group,
)
from ranking.scorers.skill import (
    TRUST_REDUCTION_FACTOR,
    TRUST_REDUCTION_THRESHOLD_MONTHS,
    SkillScorer,
)
from tests.property._factories import build_candidate, career_entry, skill

# ---------------------------------------------------------------------------
# Shared generators.
# ---------------------------------------------------------------------------

# Every known variant across all groups maps to a required group (must-have or
# nice-to-have), so any of these is guaranteed-relevant.
_KNOWN_VARIANTS = sorted({v for variants in SKILL_GROUPS.values() for v in variants})

_PROFICIENCIES = st.sampled_from(
    sorted(PROFICIENCY_FACTOR.keys()) + ["unknown", ""]
)
_ENDORSEMENTS = st.integers(min_value=-5, max_value=200)
_DURATION = st.integers(min_value=0, max_value=240)

# Random noise names: short alphabetic tokens. These are filtered at runtime so
# that only names that genuinely map to no group are used.
_NOISE_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=12
)


def _make_skill(name, proficiency, endorsements, duration_months):
    return skill(
        name=name,
        proficiency=proficiency,
        endorsements=endorsements,
        duration_months=duration_months,
    )


# A skill spec is (name, proficiency, endorsements, duration_months).
_KNOWN_SKILL_SPEC = st.tuples(
    st.sampled_from(_KNOWN_VARIANTS), _PROFICIENCIES, _ENDORSEMENTS, _DURATION
)
_NOISE_SKILL_SPEC = st.tuples(_NOISE_NAME, _PROFICIENCIES, _ENDORSEMENTS, _DURATION)


# ---------------------------------------------------------------------------
# Property 6: Skill score bounded output.
# ---------------------------------------------------------------------------


# Feature: candidate-ranking-system, Property 6: Skill score bounded output
@given(
    known=st.lists(_KNOWN_SKILL_SPEC, min_size=0, max_size=8),
    noise=st.lists(_NOISE_SKILL_SPEC, min_size=0, max_size=8),
)
def test_skill_score_bounded_and_zero_when_no_match(known, noise):
    """Property 6: score is in [0.0, 1.0]; 0.0 when nothing maps to a group.

    Skills are drawn from (a) known group variants (always relevant) and (b)
    random noise names that are filtered so ``skill_to_group`` returns ``None``.
    When every skill is non-matching noise (or the list is empty) the score must
    be exactly 0.0; in all cases the score must lie in [0.0, 1.0].

    **Validates: Requirements 3.1, 3.5**
    """
    # Keep only genuinely non-matching noise names.
    noise = [spec for spec in noise if skill_to_group(spec[0]) is None]

    skills = [_make_skill(*spec) for spec in known + noise]
    candidate = build_candidate(skills=skills)

    score = SkillScorer().score(candidate)

    # Always bounded (Req 3.1).
    assert 0.0 <= score <= 1.0

    # Exactly 0.0 when none of the candidate's skills map to a required group
    # (Req 3.5), including the empty-skills case.
    if not known:
        assert score == 0.0


# Feature: candidate-ranking-system, Property 6: Skill score bounded output
@given(
    noise=st.lists(_NOISE_SKILL_SPEC, min_size=0, max_size=10),
)
def test_skill_score_zero_for_only_noise(noise):
    """A candidate whose skills are entirely non-matching scores exactly 0.0.

    **Validates: Requirements 3.5**
    """
    noise = [spec for spec in noise if skill_to_group(spec[0]) is None]
    skills = [_make_skill(*spec) for spec in noise]
    candidate = build_candidate(skills=skills)

    score = SkillScorer().score(candidate)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Property 7: Skill trust-penalty threshold.
# ---------------------------------------------------------------------------


def _unreduced_expected(weight, proficiency, endorsements, duration_months):
    """Independently compute the unreduced contribution for the given factors."""
    prof = PROFICIENCY_FACTOR.get(proficiency, 0.25)
    endorsement_factor = SkillScorer._endorsement_factor(endorsements)
    duration_factor = SkillScorer._duration_factor(duration_months)
    return weight * prof * endorsement_factor * duration_factor


# Feature: candidate-ranking-system, Property 7: Skill trust-penalty threshold
@given(
    weight=st.sampled_from([1, 2]),
    proficiency=st.sampled_from(sorted(PROFICIENCY_FACTOR.keys())),
    endorsements=st.integers(min_value=0, max_value=200),
    duration_months=st.integers(min_value=0, max_value=240),
    career_total=st.integers(min_value=0, max_value=240),
)
def test_skill_contribution_trust_penalty_boundary(
    weight, proficiency, endorsements, duration_months, career_total
):
    """Property 7: contribution is x0.25 exactly when duration exceeds career
    total by more than 6 months, and unreduced within the threshold.

    Targets ``SkillScorer._skill_contribution`` directly. We compare the
    contribution at ``(duration - career_total) == 6`` (no reduction) against
    ``== 7`` (x0.25), holding all factors fixed by recomputing career_total so
    the difference hits exactly the boundary.

    **Validates: Requirements 3.2, 3.3**
    """
    contribution = SkillScorer._skill_contribution(
        weight=weight,
        proficiency=proficiency,
        endorsements=endorsements,
        duration_months=duration_months,
        career_total_months=career_total,
    )

    expected = _unreduced_expected(weight, proficiency, endorsements, duration_months)
    if duration_months - career_total > TRUST_REDUCTION_THRESHOLD_MONTHS:
        expected *= TRUST_REDUCTION_FACTOR

    assert contribution == pytest.approx(expected, abs=1e-12)


# Feature: candidate-ranking-system, Property 7: Skill trust-penalty threshold
@given(
    weight=st.sampled_from([1, 2]),
    proficiency=st.sampled_from(sorted(PROFICIENCY_FACTOR.keys())),
    endorsements=st.integers(min_value=0, max_value=200),
    duration=st.integers(min_value=12, max_value=200),
)
def test_skill_contribution_exact_quarter_at_plus_seven(
    weight, proficiency, endorsements, duration
):
    """At the +7 boundary the contribution equals exactly 0.25 x the +6 value.

    With ``duration`` fixed, the only factor that changes between a career_total
    that yields ``duration - career_total == 6`` and one that yields ``== 7`` is
    the trust multiplier. So the +7 contribution must be exactly 0.25 x the +6
    contribution (which is itself the unreduced value).

    **Validates: Requirements 3.2, 3.3**
    """
    career_total_plus6 = duration - 6  # difference == 6 -> no reduction
    career_total_plus7 = duration - 7  # difference == 7 -> x0.25

    contribution_plus6 = SkillScorer._skill_contribution(
        weight=weight,
        proficiency=proficiency,
        endorsements=endorsements,
        duration_months=duration,
        career_total_months=career_total_plus6,
    )
    contribution_plus7 = SkillScorer._skill_contribution(
        weight=weight,
        proficiency=proficiency,
        endorsements=endorsements,
        duration_months=duration,
        career_total_months=career_total_plus7,
    )

    # +6 is the unreduced value.
    assert contribution_plus6 == pytest.approx(
        _unreduced_expected(weight, proficiency, endorsements, duration), abs=1e-12
    )
    # +7 is exactly 0.25 x the unreduced value.
    assert contribution_plus7 == pytest.approx(
        TRUST_REDUCTION_FACTOR * contribution_plus6, abs=1e-12
    )


# Feature: candidate-ranking-system, Property 7: Skill trust-penalty threshold
@given(
    variant=st.sampled_from(_KNOWN_VARIANTS),
    proficiency=st.sampled_from(sorted(PROFICIENCY_FACTOR.keys())),
    endorsements=st.integers(min_value=0, max_value=200),
    career_total=st.integers(min_value=12, max_value=180),
)
def test_score_level_penalty_relationship(
    variant, proficiency, endorsements, career_total
):
    """End-to-end: penalized candidate (duration = career_total + 7) scores no
    higher than the non-penalized one (duration = career_total + 6), and the
    relationship reflects the 0.25 factor on that skill's contribution.

    Both candidates are identical except for the single relevant skill's
    ``duration_months``. A longer duration would normally *increase* the
    duration_factor, so any score reduction is attributable to the trust
    penalty.

    **Validates: Requirements 3.2, 3.3**
    """
    # Single career entry summing to ``career_total`` months.
    career = [career_entry(start_date=None, duration_months=career_total)]

    dur_not_penalized = career_total + TRUST_REDUCTION_THRESHOLD_MONTHS  # +6
    dur_penalized = career_total + TRUST_REDUCTION_THRESHOLD_MONTHS + 1  # +7

    not_penalized = build_candidate(
        career_history=career,
        skills=[_make_skill(variant, proficiency, endorsements, dur_not_penalized)],
    )
    penalized = build_candidate(
        career_history=career,
        skills=[_make_skill(variant, proficiency, endorsements, dur_penalized)],
    )

    score_not_penalized = SkillScorer().score(not_penalized)
    score_penalized = SkillScorer().score(penalized)

    # Penalized score must not exceed the non-penalized score.
    assert score_penalized <= score_not_penalized + 1e-12

    # The relationship reflects the 0.25 factor on the single skill's
    # contribution. The penalized (+7) contribution must equal exactly 0.25 x
    # the *unreduced* contribution computed at the same penalized duration
    # (computed via a large career_total so no reduction applies). We compare at
    # equal duration so the only difference is the trust multiplier itself.
    group = skill_to_group(variant)
    weight = group_weight(group)

    contribution_plus7_penalized = SkillScorer._skill_contribution(
        weight=weight,
        proficiency=proficiency,
        endorsements=endorsements,
        duration_months=dur_penalized,
        career_total_months=career_total,
    )
    contribution_plus7_unreduced = SkillScorer._skill_contribution(
        weight=weight,
        proficiency=proficiency,
        endorsements=endorsements,
        duration_months=dur_penalized,
        # Large career_total -> difference <= 6 -> no trust reduction applies.
        career_total_months=dur_penalized,
    )
    assert contribution_plus7_penalized == pytest.approx(
        TRUST_REDUCTION_FACTOR * contribution_plus7_unreduced, abs=1e-12
    )
