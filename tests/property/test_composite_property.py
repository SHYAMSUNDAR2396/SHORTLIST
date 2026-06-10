# Feature: candidate-ranking-system, Property 16: Composite score formula and clamping
# Feature: candidate-ranking-system, Property 17: All component scores bounded in [0.0, 1.0]
"""Composite scorer property tests (design Properties 16 and 17, composite-side).

Property 16: Composite score formula and clamping.
    **Validates: Requirements 8.1, 8.3, 8.4, 7.9**

    For any set of component scores (including out-of-range values) and any
    penalty multiplier in ``(0.0, 1.0]``, ``CompositeScorer.compute`` equals
    ``(sum over WEIGHTS of weight * clamp01(component)) * penalty_multiplier``.
    Missing keys default to 0.0. Because the behavioral weight is 0.10 and a
    clamped component is at most 1.0, the behavioral contribution can never
    exceed 15% of the total weight.

Property 17 (composite-side): All component scores bounded in [0.0, 1.0].
    **Validates: Requirements 8.3, 3.1**

    ``clamp01`` maps any float into ``[0.0, 1.0]``: it is the identity inside
    the range, returns 0.0 below the range, and 1.0 above it.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ranking.composite import CompositeScorer, clamp01, weighted_sum
from ranking.constants import WEIGHTS

# Component-score floats span well outside [0, 1] so clamping is exercised.
_component_scores = st.floats(
    min_value=-2.0, max_value=3.0, allow_nan=False, allow_infinity=False
)

# A dict of component scores; keys are a (possibly partial) subset of the
# canonical component names so the "missing keys default to 0.0" path is hit.
_component_dicts = st.dictionaries(
    keys=st.sampled_from(list(WEIGHTS.keys())),
    values=_component_scores,
)

# Penalty multiplier strictly in (0.0, 1.0].
_penalty_multipliers = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    exclude_min=True,
)


def _expected_composite(scores, penalty):
    """Independently recompute the weighted, clamped sum times the penalty."""
    total = 0.0
    for key, weight in WEIGHTS.items():
        component = scores.get(key, 0.0)
        clamped = min(1.0, max(0.0, component))
        total += weight * clamped
    return total * penalty


# --- Property 16: Composite score formula and clamping ------------------------


@given(scores=_component_dicts, penalty=_penalty_multipliers)
def test_composite_formula_and_clamping(scores, penalty):
    """compute() equals the clamped weighted sum scaled by the penalty, in [0, 1]."""
    scorer = CompositeScorer()
    result = scorer.compute(scores, penalty_multiplier=penalty)
    expected = _expected_composite(scores, penalty)

    assert result == pytest.approx(expected)
    # With a penalty in (0, 1] and clamped components, the result is in [0, 1].
    assert 0.0 <= result <= 1.0 + 1e-9


@given(scores=_component_dicts)
def test_composite_default_penalty_is_weighted_sum(scores):
    """With the default penalty (1.0), compute() equals weighted_sum()."""
    scorer = CompositeScorer()
    assert scorer.compute(scores) == pytest.approx(weighted_sum(scores))


def test_behavioral_contribution_never_exceeds_15_percent():
    """The behavioral term is capped by its 0.10 weight, below the 15% ceiling.

    The maximal behavioral contribution occurs at a clamped behavioral score of
    1.0, which yields exactly the behavioral weight (0.10). 0.10 <= 0.15.
    """
    assert WEIGHTS["behavioral"] == 0.10
    behavioral_only = weighted_sum({"behavioral": 1.0})
    assert behavioral_only == pytest.approx(0.10)
    assert behavioral_only <= 0.15
    # Even an out-of-range behavioral input is clamped to 1.0 first.
    assert weighted_sum({"behavioral": 99.0}) == pytest.approx(0.10)


def test_missing_keys_default_to_zero():
    """Absent component keys contribute 0.0 to the weighted sum."""
    # Only skill present: composite is exactly the skill weight times the value.
    assert weighted_sum({"skill": 1.0}) == pytest.approx(WEIGHTS["skill"])
    # Empty dict yields 0.0.
    assert weighted_sum({}) == 0.0


def test_all_components_one_sums_to_one():
    """All clamped components at 1.0 yield a composite of 1.0 (weights sum to 1)."""
    full = {key: 1.0 for key in WEIGHTS}
    assert weighted_sum(full) == pytest.approx(1.0)
    assert CompositeScorer().compute(full, penalty_multiplier=0.5) == pytest.approx(0.5)


# --- Property 17 (composite-side): component scores bounded in [0, 1] ---------

# A wide range of floats including deep negatives and well above 1.0.
_wide_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)


@given(x=_wide_floats)
def test_clamp01_is_bounded(x):
    """clamp01 maps any float into [0.0, 1.0]."""
    assert 0.0 <= clamp01(x) <= 1.0


@given(x=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_clamp01_identity_in_range(x):
    """clamp01 is the identity for values already within [0.0, 1.0]."""
    assert clamp01(x) == pytest.approx(x)


@given(x=st.floats(min_value=-1e6, max_value=0.0, allow_nan=False, allow_infinity=False, exclude_max=True))
def test_clamp01_floors_negatives(x):
    """clamp01 returns 0.0 for any value strictly below 0.0."""
    assert clamp01(x) == 0.0


@given(x=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False, exclude_min=True))
def test_clamp01_caps_above_one(x):
    """clamp01 returns 1.0 for any value strictly above 1.0."""
    assert clamp01(x) == 1.0
