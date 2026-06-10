"""Ranking and selection for the Candidate Ranking System.

This module turns a list of :class:`ranking.models.ScoredCandidate` records
into a final ranked ordering, applying the two ranking-stage rules from the
design:

1. **Deterministic ordering** (Req 8.2, 8.5, 9.4, 9.5) — candidates are sorted
   by a stable key ``(-final_score, candidate_id)`` so that scores are
   non-increasing by rank position and any two candidates with identical scores
   appear in ascending lexicographic ``candidate_id`` order. Python's built-in
   :func:`sorted` is stable, and the explicit key makes the ordering fully
   deterministic with no reliance on input order, hashing, or randomness
   (design "Determinism Guarantees", Req 10.5).

2. **Clean-first selection** (Req 11.4, 2.5) — candidates whose
   ``disqualification_reason`` is ``None`` ("clean") are preferred. The top
   ``top_n`` slots are filled from clean candidates first; only when fewer than
   ``top_n`` clean candidates exist are the remaining slots filled from
   penalized candidates (those with a non-``None`` ``disqualification_reason``),
   in their own sorted order.

Honeypot-flagged candidates are excluded *upstream* by the HoneypotDetector and
never appear in the ``scored`` list passed here, satisfying the honeypot half of
Req 2.5. This module therefore concerns itself only with separating clean from
penalized candidates.

Reasoning strings (Req 11) are produced by the ReasoningGenerator and attached
by the main orchestrator. To keep this module focused and pure, :func:`to_ranked`
accepts an optional ``reasoning_fn`` callback; when omitted, the ``reasoning``
field is left empty for the orchestrator to fill. Scores are carried through as
the raw ``final_score``; rounding to 4 decimal places happens at the formatting
boundary (Req 8.8), not here.
"""

from typing import Callable, List, Optional, Tuple

from .models import RankedCandidate, ScoredCandidate

# Default number of candidates to select for the final submission (Req 8.5).
DEFAULT_TOP_N: int = 100

# A reasoning callback maps (rank, ScoredCandidate) -> reasoning string. It is
# optional; when not provided, reasoning is left empty for the orchestrator.
ReasoningFn = Callable[[int, ScoredCandidate], str]


def ranking_sort_key(scored: ScoredCandidate) -> Tuple[float, str]:
    """Return the deterministic sort key for a scored candidate.

    The key is ``(-final_score, candidate_id)`` so that, under ascending sort:

    - higher ``final_score`` sorts first (scores non-increasing by rank), and
    - ties on ``final_score`` break by ``candidate_id`` ascending lexicographic
      order (Req 8.2, 8.5, 9.4, 9.5).

    Exposed at module level so property tests (16.2) can verify the ordering
    invariant against the exact key the ranker uses.
    """
    return (-scored.final_score, scored.candidate_id)


def sort_scored(scored: List[ScoredCandidate]) -> List[ScoredCandidate]:
    """Return ``scored`` sorted deterministically by :func:`ranking_sort_key`.

    Uses Python's stable :func:`sorted` with an explicit key, so the result is
    independent of the input ordering and contains no source of nondeterminism
    (Req 10.5). The input list is not mutated.
    """
    return sorted(scored, key=ranking_sort_key)


def is_clean(scored: ScoredCandidate) -> bool:
    """True when a candidate carries no disqualification penalty (Req 11.4).

    A "clean" candidate has ``disqualification_reason is None``. Penalized
    candidates have a non-``None`` reason recorded by the Disqualifier_Filter.
    """
    return scored.disqualification_reason is None


def partition_clean_penalized(
    scored: List[ScoredCandidate],
) -> Tuple[List[ScoredCandidate], List[ScoredCandidate]]:
    """Split candidates into ``(clean, penalized)``, each sorted deterministically.

    Both partitions are sorted with :func:`ranking_sort_key`. Separating first
    and sorting each partition independently lets :func:`select_top` prefer
    clean candidates while still honoring the score/tie ordering within each
    group (Req 11.4, 2.5).
    """
    clean: List[ScoredCandidate] = []
    penalized: List[ScoredCandidate] = []
    for candidate in scored:
        if is_clean(candidate):
            clean.append(candidate)
        else:
            penalized.append(candidate)
    return sort_scored(clean), sort_scored(penalized)


def select_top(
    scored: List[ScoredCandidate], top_n: int = DEFAULT_TOP_N
) -> List[ScoredCandidate]:
    """Select the top ``top_n`` candidates in ranked order (Req 8.5, 11.4, 2.5).

    Selection rules:

    - Clean candidates (no ``disqualification_reason``) are preferred and fill
      the slots first, in ``(-final_score, candidate_id)`` order.
    - Penalized candidates fill any remaining slots *only* when fewer than
      ``top_n`` clean candidates exist, also in sorted order (Req 11.4).
    - Honeypot-flagged candidates are assumed already excluded upstream and are
      not expected in ``scored`` (Req 2.5).

    If the total number of available candidates is fewer than ``top_n`` (e.g. a
    very small dataset), as many as are available are returned; producing
    exactly ``top_n`` rows is the responsibility of the formatter / main
    pipeline.

    Args:
        scored: Scored candidates to rank (non-honeypot). Not mutated.
        top_n: Maximum number of candidates to select. Non-positive values
            yield an empty list.

    Returns:
        A new list of at most ``top_n`` ScoredCandidate objects in final ranked
        order (clean-first, then penalized fill, each by sort key).
    """
    if top_n <= 0:
        return []

    clean, penalized = partition_clean_penalized(scored)

    selected = clean[:top_n]
    if len(selected) < top_n:
        remaining = top_n - len(selected)
        selected = selected + penalized[:remaining]
    return selected


def to_ranked(
    ordered: List[ScoredCandidate],
    reasoning_fn: Optional[ReasoningFn] = None,
) -> List[RankedCandidate]:
    """Attach ranks 1..N to an already-ordered list of candidates.

    The input ``ordered`` is taken to be in final ranked order (e.g. the output
    of :func:`select_top`); ranks are assigned by position starting at 1.

    The ``score`` field carries the raw ``final_score`` unchanged — rounding to
    4 decimal places is deferred to the formatting boundary (Req 8.8). The
    ``reasoning`` field is populated from ``reasoning_fn(rank, candidate)`` when
    a callback is supplied, otherwise left as an empty string for the
    orchestrator to fill via the ReasoningGenerator (Req 11).

    Args:
        ordered: Candidates already in final ranked order.
        reasoning_fn: Optional callback producing a reasoning string for each
            ``(rank, ScoredCandidate)``.

    Returns:
        A list of RankedCandidate objects with ``rank`` 1..len(ordered).
    """
    ranked: List[RankedCandidate] = []
    for index, candidate in enumerate(ordered):
        rank = index + 1
        reasoning = reasoning_fn(rank, candidate) if reasoning_fn is not None else ""
        ranked.append(
            RankedCandidate(
                candidate_id=candidate.candidate_id,
                rank=rank,
                score=candidate.final_score,
                reasoning=reasoning,
            )
        )
    return ranked


def rank_candidates(
    scored: List[ScoredCandidate],
    top_n: int = DEFAULT_TOP_N,
    reasoning_fn: Optional[ReasoningFn] = None,
) -> List[RankedCandidate]:
    """Select the top ``top_n`` candidates and assign ranks 1..N.

    Convenience composition of :func:`select_top` (clean-first selection in
    deterministic sort order) and :func:`to_ranked` (rank assignment). Honeypot
    exclusion is handled upstream; penalized candidates only appear when fewer
    than ``top_n`` clean candidates exist (Req 11.4, 2.5).

    Args:
        scored: Scored, non-honeypot candidates. Not mutated.
        top_n: Maximum number of candidates to select (default 100).
        reasoning_fn: Optional reasoning callback forwarded to :func:`to_ranked`.

    Returns:
        Ranked candidates (at most ``top_n``) with scores non-increasing by rank
        and ties broken by ``candidate_id`` ascending.
    """
    selected = select_top(scored, top_n)
    return to_ranked(selected, reasoning_fn)
