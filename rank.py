"""Candidate Ranking System - CLI entry point and pipeline orchestrator.

This module wires together the full ranking pipeline and exposes the single
reproduce command required by the challenge::

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pipeline stages (design "Pipeline Flow"):

1. **Load** candidate profiles from the JSONL input (``DataLoader``). A file
   access failure causes a non-zero exit inside the loader (Req 1.5).
2. **Determine eval_date** deterministically from the loaded data (see below).
3. **Honeypot detection** — flag and exclude temporally-impossible profiles
   (Req 2.5).
4. **Score** each clean candidate across the six weighted dimensions, aggregate
   the composite score, then apply the disqualifier penalty (Req 8.1, 8.4).
5. **Rank & select** the top 100 (clean-first, deterministic order; Req 8.5,
   11.4, 2.5).
6. **Reasoning** strings for each selected candidate (Req 11).
7. **Format** the validated submission CSV (Req 9).

Determinism (Req 10.5)
----------------------
Bit-identical output across runs is a hard requirement. Two design choices make
this hold:

- **No randomness / no hash-order dependence.** Selection and ordering rely on
  the ranker's explicit ``(-final_score, candidate_id)`` sort key.
- **A single eval_date captured once.** Crucially, the eval_date is *not* read
  from the system clock (``date.today()`` would make output depend on the run
  date, breaking reproducibility across days). Instead it is derived
  deterministically from the dataset itself: the maximum of all candidate dates
  present (last_active / signup signal dates and career start/end dates). This
  fixes the "180 days", "last 12 months", and "24/36/12 month" thresholds to the
  data, so the same input always yields the same output regardless of when the
  pipeline runs. If no dates are present at all, a fixed constant date is used.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Dict, List, Optional

from ranking.composite import CompositeScorer
from ranking.disqualifier import DisqualifierFilter
from ranking.formatter import SubmissionFormatter
from ranking.honeypot import HoneypotDetector
from ranking.loader import DataLoader
from ranking.models import CandidateProfile, RankedCandidate, ScoredCandidate
from ranking.ranker import select_top, to_ranked
from ranking.reasoning import ReasoningGenerator
from ranking.scorers.behavioral import BehavioralSignalEvaluator
from ranking.scorers.career import CareerAnalyzer
from ranking.scorers.education import EducationScorer
from ranking.scorers.experience import ExperienceScorer
from ranking.scorers.location import LocationWorkModeScorer
from ranking.scorers.skill import SkillScorer

# Number of candidates in the final submission (Req 8.5).
TOP_N = 100

# Deterministic fallback eval_date used only when the dataset carries no usable
# dates at all. Chosen as a fixed constant so output never depends on the wall
# clock (Req 10.5).
FALLBACK_EVAL_DATE = date(2026, 1, 1)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the ``--candidates`` and ``--out`` CLI arguments (Req 10.5)."""
    parser = argparse.ArgumentParser(
        description="Rank candidate profiles and emit a top-100 submission CSV."
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to the candidates JSONL input file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the submission CSV file.",
    )
    return parser.parse_args(argv)


def determine_eval_date(candidates: List[CandidateProfile]) -> date:
    """Derive a single, deterministic evaluation date from the dataset.

    The eval_date is the maximum date observed across all candidates, considering
    each candidate's Redrob signal dates (``last_active_date``, ``signup_date``)
    and every career-history ``start_date`` / ``end_date``. Using the dataset's
    own latest date (rather than ``date.today()``) keeps every recency/age
    threshold reproducible across runs and across days (Req 10.5).

    Returns :data:`FALLBACK_EVAL_DATE` when no usable date is present anywhere in
    the dataset.
    """
    latest: Optional[date] = None

    def consider(d: Optional[date]) -> None:
        nonlocal latest
        if d is not None and (latest is None or d > latest):
            latest = d

    for candidate in candidates:
        signals = candidate.redrob_signals
        consider(signals.last_active_date)
        consider(signals.signup_date)
        for entry in candidate.career_history:
            consider(entry.start_date)
            consider(entry.end_date)

    return latest if latest is not None else FALLBACK_EVAL_DATE


def score_candidates(
    candidates: List[CandidateProfile], eval_date: date
) -> List[ScoredCandidate]:
    """Score each (clean) candidate and apply the disqualifier penalty.

    For every candidate this computes the six component scores, aggregates the
    unpenalized composite (Req 8.1), then applies the disqualifier to obtain the
    penalized ``final_score`` and the recorded reason (Req 6, 8.4). The scorer
    instances are created once and reused so the per-candidate loop stays a
    single efficient pass (Req 10.1, 10.6).
    """
    skill_scorer = SkillScorer()
    career_scorer = CareerAnalyzer()
    experience_scorer = ExperienceScorer()
    behavioral_scorer = BehavioralSignalEvaluator()
    education_scorer = EducationScorer()
    location_scorer = LocationWorkModeScorer()
    composite_scorer = CompositeScorer()
    disqualifier = DisqualifierFilter()

    scored: List[ScoredCandidate] = []
    for candidate in candidates:
        skill = skill_scorer.score(candidate)
        career = career_scorer.score(candidate)
        experience = experience_scorer.score(candidate)
        behavioral = behavioral_scorer.score(candidate, eval_date)
        education = education_scorer.score(candidate)
        location = location_scorer.score(candidate)

        scores: Dict[str, float] = {
            "skill": skill,
            "career": career,
            "experience": experience,
            "behavioral": behavioral,
            "education": education,
            "location_work_mode": location,
        }

        # Unpenalized composite (penalty applied separately below, Req 8.4).
        composite = composite_scorer.compute(scores, penalty_multiplier=1.0)

        penalized_score, reason = disqualifier.apply(candidate, composite, eval_date)

        # Recover the effective penalty multiplier for the record. Guard against
        # a zero composite so the division stays well-defined.
        if reason is None:
            penalty_multiplier = 1.0
        elif composite > 0.0:
            penalty_multiplier = penalized_score / composite
        else:
            penalty_multiplier = 1.0

        # Round the final_score to the 4-decimal output precision *before*
        # ranking (Req 8.8). The submission validator compares the emitted
        # 4-decimal scores and requires equal scores to be ordered by
        # candidate_id ascending (Req 9.4, 9.5). The ranker breaks ties by
        # candidate_id on whatever final_score it is given, so two candidates
        # whose raw scores differ but round to the same value must already carry
        # the rounded value here for the tie-break to match the emitted column.
        final_score = round(penalized_score, 4)

        scored.append(
            ScoredCandidate(
                candidate_id=candidate.candidate_id,
                skill_score=skill,
                career_score=career,
                experience_score=experience,
                behavioral_score=behavioral,
                education_score=education,
                location_work_mode_score=location,
                composite_score=composite,
                penalty_multiplier=penalty_multiplier,
                disqualification_reason=reason,
                final_score=final_score,
            )
        )

    return scored


def build_ranked(
    ordered: List[ScoredCandidate],
    profiles_by_id: Dict[str, CandidateProfile],
) -> List[RankedCandidate]:
    """Assign ranks 1..N and attach a reasoning string to each selected row.

    ``ordered`` is the ranker's selected, already-sorted list. The reasoning
    generator needs the original :class:`CandidateProfile`, so a callback looks
    it up by ``candidate_id`` from the precomputed ``profiles_by_id`` map
    (built once, O(1) per lookup; Req 10.1).
    """
    reasoning_generator = ReasoningGenerator()

    def reasoning_fn(_rank: int, scored: ScoredCandidate) -> str:
        profile = profiles_by_id.get(scored.candidate_id)
        if profile is None:
            return ""
        return reasoning_generator.generate(profile, scored)

    return to_ranked(ordered, reasoning_fn=reasoning_fn)


def run_pipeline(candidates_path: str, output_path: str) -> int:
    """Execute the full ranking pipeline and write the submission CSV.

    Returns an integer exit code (0 on success). File-access failures during
    load cause a non-zero exit inside the loader (Req 1.5).
    """
    # 1. Load candidates (loader exits non-zero on file access failure, Req 1.5).
    candidates, _stats = DataLoader().load(candidates_path)

    # Build candidate_id -> profile map once for reasoning + valid-id filtering.
    profiles_by_id: Dict[str, CandidateProfile] = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    valid_candidate_ids = set(profiles_by_id.keys())

    # 2. Single deterministic eval_date for the whole run (Req 10.5).
    eval_date = determine_eval_date(candidates)
    print(f"[rank] Using deterministic eval_date={eval_date.isoformat()}", file=sys.stdout)

    # 3. Honeypot detection — flagged candidates are excluded (Req 2.5).
    clean_candidates, flagged_ids = HoneypotDetector().detect(candidates, eval_date)
    print(
        f"[rank] Honeypot detection flagged {len(flagged_ids)} candidate(s); "
        f"{len(clean_candidates)} remain.",
        file=sys.stdout,
    )

    # 4. Score + disqualify each clean candidate.
    scored = score_candidates(clean_candidates, eval_date)

    # 5. Select the top 100 (clean-first, deterministic order; Req 8.5, 11.4).
    ordered = select_top(scored, TOP_N)

    # 6. Assign ranks and generate reasoning (Req 11).
    ranked = build_ranked(ordered, profiles_by_id)

    # 7. Write the submission CSV (Req 9); unknown ids are filtered/reported.
    SubmissionFormatter().write(ranked, output_path, valid_candidate_ids=valid_candidate_ids)
    print(f"[rank] Wrote {len(ranked)} ranked candidate(s) to {output_path}", file=sys.stdout)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns an integer exit code (0 on success)."""
    args = parse_args(argv)
    return run_pipeline(args.candidates, args.out)


if __name__ == "__main__":
    sys.exit(main())
