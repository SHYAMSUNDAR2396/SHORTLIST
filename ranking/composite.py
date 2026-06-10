"""Composite scoring aggregation for the Candidate Ranking System.

This module combines the six per-candidate component scores into a single
composite score using the fixed weights defined in
:data:`ranking.constants.WEIGHTS` (Req 8.1). Aggregation proceeds in three
deterministic, pure steps:

1. **Clamp** each component to ``[0.0, 1.0]`` (Req 8.3). A scoring component is
   responsible for producing its own value in range based on its internal
   rules; clamping here is a defensive normalization, never cross-candidate
   min-max scaling.
2. **Weighted sum** of the clamped components using ``WEIGHTS``
   (0.35 skill + 0.25 career + 0.15 experience + 0.10 behavioral +
   0.10 education + 0.05 location_work_mode). Because the weights sum to 1.0
   and every component is clamped to ``[0, 1]``, the weighted sum also lies in
   ``[0.0, 1.0]``.
3. **Penalty** multiplication, applied *after* aggregation (Req 8.4). The
   penalty multiplier comes from the Disqualifier_Filter and is in ``(0.0, 1.0]``.

Behavioral contribution cap (Req 7.9): the behavioral weight is 0.10, which is
strictly less than the 15% ceiling, and a clamped behavioral component is at
most 1.0, so the behavioral term can contribute at most 0.10 (10%) of the total
weight. The constraint therefore holds by construction.

The clamping (:func:`clamp01`) and weighted-sum (:func:`weighted_sum`) steps are
exposed as standalone module-level functions so that property tests can target
each piece independently.
"""

from typing import Dict

from .constants import WEIGHTS

# Canonical component keys, in weight order. The ``scores`` dict passed to
# CompositeScorer.compute is keyed by these names; any missing key defaults to
# 0.0.
COMPONENT_KEYS = (
    "skill",
    "career",
    "experience",
    "behavioral",
    "education",
    "location_work_mode",
)


def clamp01(x: float) -> float:
    """Clamp a value to the closed interval ``[0.0, 1.0]`` (Req 8.3).

    Values below 0.0 become 0.0 and values above 1.0 become 1.0. This is a
    pure, deterministic helper used to normalize every component score before
    it is weighted.
    """
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def weighted_sum(scores: Dict[str, float]) -> float:
    """Return the clamped, weighted sum of component scores (Req 8.1, 8.3).

    Each component is clamped to ``[0.0, 1.0]`` and multiplied by its weight in
    :data:`ranking.constants.WEIGHTS`. Missing keys default to 0.0. Because the
    weights sum to 1.0 and each clamped component lies in ``[0, 1]``, the result
    lies in ``[0.0, 1.0]``.
    """
    total = 0.0
    for key, weight in WEIGHTS.items():
        component = scores.get(key, 0.0)
        total += weight * clamp01(component)
    return total


class CompositeScorer:
    """Aggregate component scores into a single composite score.

    The weights are sourced from :data:`ranking.constants.WEIGHTS`:

    - skill: 0.35
    - career: 0.25
    - experience: 0.15
    - behavioral: 0.10
    - education: 0.10
    - location_work_mode: 0.05

    The weights sum to 1.0, so a fully-clamped input yields a composite in
    ``[0.0, 1.0]`` before any penalty is applied.
    """

    WEIGHTS = WEIGHTS

    def compute(self, scores: Dict[str, float], penalty_multiplier: float = 1.0) -> float:
        """Compute the composite score: clamp -> weighted sum -> penalty.

        Steps (all pure and deterministic):

        1. Each component in ``scores`` is clamped to ``[0.0, 1.0]`` (Req 8.3).
           Missing keys default to 0.0.
        2. The clamped components are combined as a weighted sum using
           ``WEIGHTS`` (Req 8.1).
        3. The aggregated weighted sum is multiplied by ``penalty_multiplier``
           (Req 8.4). The penalty is applied *after* weight aggregation.

        Args:
            scores: Mapping of component name -> raw score. Expected keys are
                'skill', 'career', 'experience', 'behavioral', 'education', and
                'location_work_mode'. Unknown keys are ignored; missing keys
                default to 0.0.
            penalty_multiplier: Disqualifier penalty in ``(0.0, 1.0]``. Defaults
                to 1.0 (no penalty).

        Returns:
            The penalized composite score. With ``penalty_multiplier`` in
            ``(0.0, 1.0]`` the result lies in ``[0.0, 1.0]``.
        """
        return weighted_sum(scores) * penalty_multiplier

    def apply_penalty(self, composite_score: float, penalty_multiplier: float) -> float:
        """Apply a disqualifier penalty to an already-aggregated score (Req 8.4).

        Kept separable from :meth:`compute` so the weighted-sum-then-penalty
        formula can be exercised piece by piece. Multiplies the post-aggregation
        composite score by the penalty multiplier.
        """
        return composite_score * penalty_multiplier
