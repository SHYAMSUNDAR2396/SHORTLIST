"""Property-based tests for :mod:`ranking.ranker`.

Covers design correctness properties 17 and 22 (tasks 16.2 and 16.3):

- **Property 17: Ranking order invariant** — the final ranking has scores
  non-increasing by rank position, ties broken by ``candidate_id`` ascending,
  and ranks form a contiguous ``1..N`` sequence.
- **Property 22: Excluded candidates absent from output** — clean candidates
  (no ``disqualification_reason``) are preferred; penalized candidates appear
  only when fewer than ``top_n`` clean candidates exist, filling remaining
  slots in sorted order.

Each test builds :class:`ScoredCandidate` objects via a small local helper and
compares the ranker's output against an independent oracle.
"""

from __future__ import annotations

from typing import List, Optional

from hypothesis import given
from hypothesis import strategies as st

from ranking.models import ScoredCandidate
from ranking.ranker import select_top, to_ranked


# ---------------------------------------------------------------------------
# Local helper: build a ScoredCandidate with the relevant fields populated and
# zeros for everything else.
# ---------------------------------------------------------------------------
def make_scored(
    candidate_id: str,
    final_score: float,
    disqualification_reason: Optional[str] = None,
) -> ScoredCandidate:
    """Build a :class:`ScoredCandidate` for ranking tests.

    Only ``candidate_id``, ``final_score`` and ``disqualification_reason`` matter
    for the ranking stage; all component scores are set to ``0.0`` and the
    penalty multiplier to ``1.0``.
    """
    return ScoredCandidate(
        candidate_id=candidate_id,
        skill_score=0.0,
        career_score=0.0,
        experience_score=0.0,
        behavioral_score=0.0,
        education_score=0.0,
        location_work_mode_score=0.0,
        composite_score=final_score,
        penalty_multiplier=1.0,
        disqualification_reason=disqualification_reason,
        final_score=final_score,
    )


# A pool of unique candidate_ids in the CAND_ + 7-digit format.
def _cand_id(n: int) -> str:
    return f"CAND_{n:07d}"


# Scores drawn from a small set so ties occur frequently.
_TIE_SCORES = st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0])


# A list of (candidate_index, score) with unique indices so candidate_ids are
# unique. Hypothesis draws a set of indices, then a score per index.
@st.composite
def _scored_lists(draw, min_size: int = 0, max_size: int = 30):
    indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=999_9999),
            min_size=min_size,
            max_size=max_size,
            unique=True,
        )
    )
    result: List[ScoredCandidate] = []
    for idx in indices:
        score = draw(_TIE_SCORES)
        result.append(make_scored(_cand_id(idx), score))
    return result


# Feature: candidate-ranking-system, Property 17: Ranking order invariant
@given(scored=_scored_lists(min_size=0, max_size=30), top_n=st.integers(min_value=1, max_value=50))
def test_ranking_order_invariant(scored, top_n):
    """Property 17: scores non-increasing by rank, ties by candidate_id asc.

    For any list of scored candidates, ``select_top`` then ``to_ranked``
    produces a ranking whose scores are non-increasing by rank position, where
    candidates with identical ``final_score`` appear in ascending
    ``candidate_id`` order, and whose ranks are a contiguous ``1..N`` sequence.

    **Validates: Requirements 8.2, 8.5, 9.4, 9.5**
    """
    selected = select_top(scored, top_n=top_n)
    ranked = to_ranked(selected)

    # Ranks are contiguous 1..N.
    assert [r.rank for r in ranked] == list(range(1, len(ranked) + 1))

    # Length is min(top_n, total) since all candidates are clean here.
    assert len(ranked) == min(top_n, len(scored))

    # Consecutive entries: scores non-increasing; ties by candidate_id asc.
    for i in range(len(ranked) - 1):
        assert ranked[i].score >= ranked[i + 1].score
        if ranked[i].score == ranked[i + 1].score:
            assert ranked[i].candidate_id < ranked[i + 1].candidate_id

    # Verify against an independent oracle: sort by (-final_score, candidate_id).
    oracle = sorted(scored, key=lambda c: (-c.final_score, c.candidate_id))[:top_n]
    assert [r.candidate_id for r in ranked] == [c.candidate_id for c in oracle]


# ---------------------------------------------------------------------------
# Property 22: build mixes of clean and penalized candidates.
# ---------------------------------------------------------------------------
# A spec is (score, is_penalized). Unique candidate_ids assigned by position.
@st.composite
def _mixed_lists(draw, max_size: int = 20):
    specs = draw(
        st.lists(
            st.tuples(_TIE_SCORES, st.booleans()),
            min_size=0,
            max_size=max_size,
        )
    )
    result: List[ScoredCandidate] = []
    for i, (score, is_penalized) in enumerate(specs):
        reason = "disqualified" if is_penalized else None
        result.append(make_scored(_cand_id(i), score, disqualification_reason=reason))
    return result


# Feature: candidate-ranking-system, Property 22: Excluded candidates absent from output
@given(scored=_mixed_lists(max_size=20), top_n=st.integers(min_value=1, max_value=10))
def test_excluded_candidates_absent_from_output(scored, top_n):
    """Property 22: penalized candidates fill slots only after clean ones.

    Clean candidates (``disqualification_reason is None``) are selected first.
    Penalized candidates appear only when fewer than ``top_n`` clean candidates
    exist, filling the remaining slots in sorted order. While unused clean
    candidates remain, no penalized candidate appears in the output.

    **Validates: Requirements 2.5, 11.4**
    """
    selected = select_top(scored, top_n=top_n)

    clean = [c for c in scored if c.disqualification_reason is None]
    penalized = [c for c in scored if c.disqualification_reason is not None]
    num_clean = len(clean)
    total = len(scored)

    # Output length is min(top_n, total).
    assert len(selected) == min(top_n, total)

    num_penalized_out = sum(1 for c in selected if c.disqualification_reason is not None)

    # Expected number of penalized in output: only fills remaining slots after
    # clean candidates, capped by how many penalized exist and by top_n.
    expected_penalized = max(0, min(top_n, total) - num_clean)
    assert num_penalized_out == expected_penalized

    # If there are at least top_n clean candidates, no penalized appear.
    if num_clean >= top_n:
        assert num_penalized_out == 0

    # No penalized appears while unused clean candidates remain: i.e. if any
    # penalized is selected, then every clean candidate must also be selected.
    selected_ids = {c.candidate_id for c in selected}
    if num_penalized_out > 0:
        for c in clean:
            assert c.candidate_id in selected_ids

    # The selected clean candidates are exactly the top-scoring clean ones.
    clean_oracle = sorted(clean, key=lambda c: (-c.final_score, c.candidate_id))[:top_n]
    selected_clean = [c for c in selected if c.disqualification_reason is None]
    assert [c.candidate_id for c in selected_clean] == [c.candidate_id for c in clean_oracle]

    # The selected penalized candidates are exactly the top-scoring penalized
    # ones needed to fill remaining slots.
    penalized_oracle = sorted(
        penalized, key=lambda c: (-c.final_score, c.candidate_id)
    )[:expected_penalized]
    selected_penalized = [c for c in selected if c.disqualification_reason is not None]
    assert [c.candidate_id for c in selected_penalized] == [
        c.candidate_id for c in penalized_oracle
    ]
