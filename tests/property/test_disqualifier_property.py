"""Property-based test for :class:`ranking.disqualifier.DisqualifierFilter`.

Covers design correctness Property 14 (task 14.2): the disqualifier
minimum-penalty rule. The filter must apply exactly the single lowest triggered
multiplier (never the product of multipliers), record the corresponding
criterion, and return ``(composite, None)`` when nothing triggers (Req 6.5/6.6).

Strategy
--------
Rather than predict which criteria fire from the inputs (which would re-encode
the implementation), we *construct* candidates engineered to trigger known
subsets of criteria via independent Hypothesis flags, then independently call
``triggered_criteria()`` to learn the actual triggered set and derive the
expected ``(multiplier, reason)`` from it. This avoids over-constraining while
still asserting the core invariant: ``apply()`` returns
``composite x min(multipliers)`` with the lowest-by-:data:`CRITERION_ORDER`
criterion as the reason.

Three independent, composable injection flags each map to a distinct criterion:

- ``inject_consulting``  -> ``all_consulting`` (0.1)
- ``inject_cv``          -> ``cv_speech_robotics`` (0.3)
- ``inject_no_validation`` -> ``no_external_validation`` (0.5)

``recent_ai_only`` never fires: no constructed role carries an AI-related title
or description, so there is never an "AI role" to evaluate for recency.
"""

from __future__ import annotations

from functools import reduce
from typing import List, Optional, Tuple

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ranking.disqualifier import (
    CRITERION_ALL_CONSULTING,
    CRITERION_CV_SPEECH_ROBOTICS,
    CRITERION_NO_EXTERNAL_VALIDATION,
    DisqualifierFilter,
)
from tests.property._factories import EVAL_DATE, build_candidate, career_entry, skill

# A start date well before EVAL_DATE; the disqualifier criteria exercised here
# depend on duration_months / company / description / skills, not exact dates.
_START = EVAL_DATE.replace(year=EVAL_DATE.year - 10)

# Composite scores spanning 0, a fraction, and 1.0 so the multiplier is applied
# against representative magnitudes.
_COMPOSITE_VALUES = st.sampled_from([0.0, 0.5, 0.8, 1.0])


def _build_for_flags(
    inject_consulting: bool,
    inject_cv: bool,
    inject_no_validation: bool,
):
    """Construct a candidate engineered to trigger a chosen subset of criteria.

    Career history drives ``all_consulting`` (company names) and
    ``no_external_validation`` (total months > 60 with no validation signals);
    skills drive ``cv_speech_robotics``. All titles/descriptions are AI-free so
    ``recent_ai_only`` never fires.
    """
    # --- Career history: two roles -----------------------------------------
    # >60 total months when injecting "no external validation"; otherwise <=60
    # so that criterion provably does not fire.
    per_role_months = 40 if inject_no_validation else 20  # 80 vs 40 total
    if inject_consulting:
        companies = ["TCS", "Infosys"]
        title = "Consultant"
    else:
        companies = ["StartupHub", "ProductCo"]
        title = "Engineer"

    career_history = [
        career_entry(
            start_date=_START,
            duration_months=per_role_months,
            company=company,
            title=title,
            company_size="51-200",
            description="",  # AI-free and validation-signal-free
        )
        for company in companies
    ]

    # --- Skills: CV-dominant (no NLP/IR) when injecting cv_speech_robotics ---
    if inject_cv:
        skills = [skill(name="computer vision", duration_months=100)]
    else:
        skills = [skill(name="python", duration_months=10)]

    # github_activity_score defaults to 0.0 and certifications default to []
    # in the factory, satisfying the no-external-validation preconditions.
    return build_candidate(career_history=career_history, skills=skills)


def _expected_min(
    triggered: List[Tuple[str, float]],
) -> Tuple[float, Optional[str]]:
    """Independent oracle: lowest multiplier, ties broken by list order.

    ``triggered`` is returned in ``CRITERION_ORDER``, so the first occurrence of
    the minimum multiplier is the lowest-by-order criterion (Req 6.5).
    """
    if not triggered:
        return 1.0, None
    best_name, best_mult = triggered[0]
    for name, mult in triggered[1:]:
        if mult < best_mult:
            best_name, best_mult = name, mult
    return best_mult, best_name


# Feature: candidate-ranking-system, Property 14: Disqualifier minimum-penalty rule
@given(
    inject_consulting=st.booleans(),
    inject_cv=st.booleans(),
    inject_no_validation=st.booleans(),
    composite_score=_COMPOSITE_VALUES,
)
def test_disqualifier_minimum_penalty_rule(
    inject_consulting,
    inject_cv,
    inject_no_validation,
    composite_score,
):
    """Property 14: apply() uses the single lowest triggered multiplier.

    For any constructed candidate, ``apply()`` returns
    ``composite_score x min(multipliers of triggered)`` with the lowest-by-order
    criterion as the reason (or ``(composite, None)`` when nothing triggers),
    and never the product of multipliers.

    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6**
    """
    candidate = _build_for_flags(inject_consulting, inject_cv, inject_no_validation)
    filt = DisqualifierFilter()

    triggered = filt.triggered_criteria(candidate, EVAL_DATE)
    names = {name for name, _ in triggered}

    # The injected conditions map deterministically to their criteria (validates
    # the individual criteria definitions, Req 6.2/6.3/6.4).
    assert (CRITERION_ALL_CONSULTING in names) is inject_consulting
    assert (CRITERION_CV_SPEECH_ROBOTICS in names) is inject_cv
    assert (CRITERION_NO_EXTERNAL_VALIDATION in names) is inject_no_validation

    expected_mult, expected_reason = _expected_min(triggered)
    penalized, reason = filt.apply(candidate, composite_score, EVAL_DATE)

    if not triggered:
        # Req 6.6: nothing triggered -> multiplier 1.0, no recorded reason.
        assert reason is None
        assert penalized == pytest.approx(composite_score, abs=1e-12)
    else:
        # Req 6.5: exactly the single lowest multiplier is applied.
        assert reason == expected_reason
        assert penalized == pytest.approx(composite_score * expected_mult, abs=1e-12)

        # The applied multiplier is one of the triggered multipliers and is the
        # minimum among them.
        triggered_mults = [m for _, m in triggered]
        assert expected_mult == min(triggered_mults)

        # Explicitly assert the penalty is NOT the compounded product when more
        # than one criterion triggers and the score is non-zero (Req 6.5).
        if len(triggered) > 1 and composite_score > 0:
            product = reduce(lambda a, b: a * b, triggered_mults, 1.0)
            assert penalized != pytest.approx(composite_score * product, abs=1e-12)
