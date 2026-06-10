"""Skill relevance scoring for the Candidate Ranking System.

The :class:`SkillScorer` evaluates how well a candidate's skills match the
"Senior AI Engineer — Founding Team" job requirements (Requirement 3). It is a
pure function over a :class:`CandidateProfile`: no global state, no clock reads,
and no randomness, so the score is deterministic and property-testable.

Design references (see design.md "Component: SkillScorer" and Requirement 3):

- Each candidate skill is mapped to a canonical skill group via
  :func:`ranking.constants.skill_to_group`. Only skills that resolve to a known
  *required* group (must-have or nice-to-have) contribute; everything else is
  ignored (Req 3.4).
- Per matched skill the raw contribution is::

      contribution = category_weight
                     × proficiency_factor
                     × endorsement_factor
                     × duration_factor

  where (Req 3.1, 3.2):
    * ``category_weight`` is 2 for must-have groups and 1 for nice-to-have
      groups (via :func:`ranking.constants.group_weight`).
    * ``proficiency_factor`` comes from
      :data:`ranking.constants.PROFICIENCY_FACTOR` (expert=1.0, advanced=0.75,
      intermediate=0.5, beginner=0.25), defaulting to 0.25 for unknown values.
    * ``endorsement_factor`` and ``duration_factor`` are documented below.

- Trust reduction (Req 3.3): when a relevant skill's ``duration_months`` exceeds
  the candidate's total career duration (the *sum* of ``career_history``
  ``duration_months``) by more than 6 months, that skill's contribution is
  multiplied by 0.25.

- The summed contributions are normalized into ``[0.0, 1.0]`` by dividing by a
  fixed "strong candidate" constant and clamping at 1.0 (see
  :data:`NORMALIZATION_CONSTANT`). The score is exactly ``0.0`` when none of the
  candidate's skills map to any required group (Req 3.5), including the
  empty-skills case.

Factor formulas (both chosen to be saturating with a positive floor so that a
matched skill always contributes something, while extra endorsements / tenure
yield diminishing returns; both are bounded in ``(0.0, 1.0]``):

- ``endorsement_factor = 0.5 + 0.5 × min(1.0, endorsements / 50.0)``
  → 0.5 at 0 endorsements, rising linearly to 1.0 at 50+ endorsements.

- ``duration_factor    = 0.5 + 0.5 × min(1.0, duration_months / 36.0)``
  → 0.5 at 0 months, rising linearly to 1.0 at 36+ months (3 years).

Deduplication (design "choose an approach that keeps groups from being double
counted"): when several skills map to the same group, only that group's *best*
(maximum) post-trust contribution is kept, and the final raw total is the sum of
those best-per-group contributions across distinct matched groups. This prevents
a candidate from inflating their score by listing many synonyms of the same
underlying competency.
"""

from typing import Dict

from ranking.constants import (
    PROFICIENCY_FACTOR,
    group_weight,
    skill_to_group,
)
from ranking.models import CandidateProfile

# Default proficiency factor for unrecognized proficiency strings (Req 3.2).
DEFAULT_PROFICIENCY_FACTOR: float = 0.25

# Saturating-factor parameters (documented in the module docstring).
ENDORSEMENT_SATURATION: float = 50.0
DURATION_SATURATION_MONTHS: float = 36.0
FACTOR_FLOOR: float = 0.5

# Trust-reduction parameters (Req 3.3).
TRUST_REDUCTION_THRESHOLD_MONTHS: int = 6
TRUST_REDUCTION_FACTOR: float = 0.25

# Normalization constant representing a "strong candidate" (design guidance).
#
# The maximum possible per-group contribution equals that group's
# ``category_weight`` (all other factors saturate at 1.0): 2.0 for a must-have
# group and 1.0 for a nice-to-have group. A strong candidate is modeled as one
# who fully covers all four must-have groups plus two nice-to-have groups:
#
#     4 must-have × 2.0  +  2 nice-to-have × 1.0  =  8.0 + 2.0  =  10.0
#
# Dividing the summed contributions by this constant (then clamping to 1.0)
# keeps the result in [0.0, 1.0]: a candidate strong across all must-haves
# scores ~0.8 and additional nice-to-have coverage pushes toward 1.0.
MUST_HAVE_TARGET_GROUPS: int = 4
MUST_HAVE_MAX_CONTRIBUTION: float = 2.0
NICE_TO_HAVE_TARGET_GROUPS: int = 2
NICE_TO_HAVE_MAX_CONTRIBUTION: float = 1.0
NORMALIZATION_CONSTANT: float = (
    MUST_HAVE_TARGET_GROUPS * MUST_HAVE_MAX_CONTRIBUTION
    + NICE_TO_HAVE_TARGET_GROUPS * NICE_TO_HAVE_MAX_CONTRIBUTION
)


class SkillScorer:
    """Scores a candidate's skill relevance, returning a value in [0.0, 1.0]."""

    def score(self, candidate: CandidateProfile) -> float:
        """Return the normalized skill relevance score in ``[0.0, 1.0]``.

        Returns exactly ``0.0`` when none of the candidate's skills map to a
        required skill group (Req 3.5), including when the skills list is empty.

        Args:
            candidate: The candidate profile to score.

        Returns:
            A float in ``[0.0, 1.0]``.
        """
        career_total_months = self._career_total_months(candidate)

        # Keep only the best contribution per distinct required group so that
        # multiple synonyms of one competency are not double counted.
        best_per_group: Dict[str, float] = {}

        for skill in candidate.skills:
            group = skill_to_group(skill.name)
            if group is None:
                continue
            weight = group_weight(group)
            if weight <= 0:
                # Defensive: skill_to_group only returns known groups, but a
                # group with no category weight contributes nothing.
                continue

            contribution = self._skill_contribution(
                weight=weight,
                proficiency=skill.proficiency,
                endorsements=skill.endorsements,
                duration_months=skill.duration_months,
                career_total_months=career_total_months,
            )

            current_best = best_per_group.get(group)
            if current_best is None or contribution > current_best:
                best_per_group[group] = contribution

        if not best_per_group:
            # No skills matched any required group (Req 3.5).
            return 0.0

        raw_total = sum(best_per_group.values())
        normalized = raw_total / NORMALIZATION_CONSTANT

        # Clamp into [0.0, 1.0] (Req 3.1).
        if normalized < 0.0:
            return 0.0
        if normalized > 1.0:
            return 1.0
        return normalized

    # ------------------------------------------------------------------
    # Per-skill contribution
    # ------------------------------------------------------------------
    @classmethod
    def _skill_contribution(
        cls,
        weight: int,
        proficiency: str,
        endorsements: int,
        duration_months: int,
        career_total_months: int,
    ) -> float:
        """Compute a single matched skill's contribution (post trust reduction).

        contribution = category_weight × proficiency_factor
                       × endorsement_factor × duration_factor,
        then multiplied by 0.25 when the skill's duration_months exceeds the
        candidate's total career duration by more than 6 months (Req 3.3).
        """
        proficiency_factor = cls._proficiency_factor(proficiency)
        endorsement_factor = cls._endorsement_factor(endorsements)
        duration_factor = cls._duration_factor(duration_months)

        contribution = (
            weight * proficiency_factor * endorsement_factor * duration_factor
        )

        if duration_months - career_total_months > TRUST_REDUCTION_THRESHOLD_MONTHS:
            contribution *= TRUST_REDUCTION_FACTOR

        return contribution

    # ------------------------------------------------------------------
    # Factor helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _proficiency_factor(proficiency: str) -> float:
        """Map a proficiency string to its factor; unknown values default to
        0.25 (Req 3.2)."""
        if not proficiency:
            return DEFAULT_PROFICIENCY_FACTOR
        return PROFICIENCY_FACTOR.get(
            proficiency.strip().lower(), DEFAULT_PROFICIENCY_FACTOR
        )

    @staticmethod
    def _endorsement_factor(endorsements: int) -> float:
        """Saturating endorsement factor in ``(0.0, 1.0]`` with a 0.5 floor.

        ``0.5 + 0.5 × min(1.0, endorsements / 50.0)``. Negative endorsement
        counts are treated as 0 (yielding the 0.5 floor).
        """
        safe_endorsements = max(0, endorsements)
        saturated = min(1.0, safe_endorsements / ENDORSEMENT_SATURATION)
        return FACTOR_FLOOR + FACTOR_FLOOR * saturated

    @staticmethod
    def _duration_factor(duration_months: int) -> float:
        """Saturating duration factor in ``(0.0, 1.0]`` with a 0.5 floor.

        ``0.5 + 0.5 × min(1.0, duration_months / 36.0)``. Negative durations are
        treated as 0 (yielding the 0.5 floor).
        """
        safe_months = max(0, duration_months)
        saturated = min(1.0, safe_months / DURATION_SATURATION_MONTHS)
        return FACTOR_FLOOR + FACTOR_FLOOR * saturated

    # ------------------------------------------------------------------
    # Career-duration helper
    # ------------------------------------------------------------------
    @staticmethod
    def _career_total_months(candidate: CandidateProfile) -> int:
        """Total career duration as the sum of career_history duration_months.

        Per Req 3.3 the trust-reduction comparison uses the *sum* of
        ``duration_months`` across all career entries (not a date-range span).
        Negative durations are treated as 0.
        """
        return sum(
            max(0, entry.duration_months) for entry in candidate.career_history
        )
