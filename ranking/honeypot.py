"""Honeypot detection for the Candidate Ranking System.

The :class:`HoneypotDetector` applies four temporal-consistency rules to flag
synthetically-injected "honeypot" candidates whose profiles contain impossible
attributes (e.g. 8 years tenure at a company founded 3 years ago). A candidate
is flagged if *any* of the rules match, and flagged candidates are excluded
from the final ranked output (Req 2.5).

Design references (see design.md "Component: HoneypotDetector" and the
"Determinism Guarantees" section):

- A single ``eval_date`` is captured once at pipeline start and threaded into
  ``detect``. The detector never reads the system clock per-candidate, which is
  required for bit-identical output across runs (Req 10.5).
- All date math goes through :func:`months_between` so the +24 / +36 / +12
  month thresholds are computed consistently.
- Missing dates are handled gracefully: an entry/skill whose required date is
  absent is skipped for the rule that needs it rather than crashing.
- The detector is deterministic and side-effect free.

Rules (Requirement 2):
1. (Req 2.1) Any career_history entry has ``duration_months`` exceeding
   ``months(start_date → eval_date)`` by more than 24 months.
2. (Req 2.2) ``|years_of_experience (in months) − total career span (months)|``
   exceeds 36, where the span runs from the earliest career start_date to the
   latest end_date (or ``eval_date`` when a role is current / has no end_date).
3. (Req 2.3) Any skill's ``duration_months`` exceeds the candidate's total
   career span in months (earliest start_date → ``eval_date``) by more than 12.
4. (Req 2.4) The candidate has 10 or more "expert"-proficiency skills AND the
   sum of endorsements across those expert skills is 0.
"""

from datetime import date
from typing import List, Optional, Set, Tuple

from ranking.models import CandidateProfile

# Rule thresholds (in months) from Requirement 2.
DURATION_ANOMALY_THRESHOLD_MONTHS: int = 24
EXPERIENCE_SPAN_MISMATCH_THRESHOLD_MONTHS: int = 36
SKILL_DURATION_ANOMALY_THRESHOLD_MONTHS: int = 12

# Rule 4 thresholds (Req 2.4).
EXPERT_SKILL_COUNT_THRESHOLD: int = 10
EXPERT_PROFICIENCY: str = "expert"


def months_between(d1: date, d2: date) -> int:
    """Return the number of whole months from ``d1`` to ``d2``.

    Computed as ``(d2.year - d1.year) * 12 + (d2.month - d1.month)`` so the same
    convention is used everywhere a month-span is needed. The result can be
    negative when ``d2`` precedes ``d1``.
    """
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


class HoneypotDetector:
    """Flags candidates with temporally-impossible profiles."""

    def detect(
        self, candidates: List[CandidateProfile], eval_date: date
    ) -> Tuple[List[CandidateProfile], Set[str]]:
        """Partition ``candidates`` into clean and flagged sets.

        Each candidate is evaluated against the four temporal-consistency rules
        using the single threaded ``eval_date``. A candidate is flagged if any
        rule matches.

        Args:
            candidates: The clean candidate list (already loaded/validated).
            eval_date: The single evaluation date captured once at pipeline
                start. Used for all date math; the system clock is never read
                per-candidate.

        Returns:
            A tuple ``(clean_candidates, flagged_ids)`` where
            ``clean_candidates`` preserves input order and excludes every
            flagged candidate, and ``flagged_ids`` is the set of flagged
            ``candidate_id`` values.
        """
        clean_candidates: List[CandidateProfile] = []
        flagged_ids: Set[str] = set()

        for candidate in candidates:
            if self.is_honeypot(candidate, eval_date):
                flagged_ids.add(candidate.candidate_id)
            else:
                clean_candidates.append(candidate)

        return clean_candidates, flagged_ids

    def is_honeypot(self, candidate: CandidateProfile, eval_date: date) -> bool:
        """Return ``True`` when any of the four honeypot rules match."""
        return (
            self._violates_duration_rule(candidate, eval_date)
            or self._violates_experience_span_rule(candidate, eval_date)
            or self._violates_skill_duration_rule(candidate, eval_date)
            or self._violates_expert_endorsement_rule(candidate)
        )

    # ------------------------------------------------------------------
    # Rule 1 (Req 2.1): per-entry duration vs date span
    # ------------------------------------------------------------------
    @staticmethod
    def _violates_duration_rule(candidate: CandidateProfile, eval_date: date) -> bool:
        """Flag if any career entry's duration_months exceeds the months from
        its start_date to ``eval_date`` by more than 24 months.

        Entries with a missing ``start_date`` are skipped for this rule.
        """
        for entry in candidate.career_history:
            if entry.start_date is None:
                continue
            span_months = months_between(entry.start_date, eval_date)
            if entry.duration_months - span_months > DURATION_ANOMALY_THRESHOLD_MONTHS:
                return True
        return False

    # ------------------------------------------------------------------
    # Rule 2 (Req 2.2): years_of_experience vs total career span
    # ------------------------------------------------------------------
    @classmethod
    def _violates_experience_span_rule(
        cls, candidate: CandidateProfile, eval_date: date
    ) -> bool:
        """Flag if |years_of_experience (months) − total career span (months)|
        exceeds 36 months.

        The career span runs from the earliest start_date to the latest end_date
        (or ``eval_date`` when a role is current or has no end_date). When there
        are no valid start dates the span cannot be computed, so this span-based
        rule is skipped.
        """
        earliest_start = cls._earliest_start_date(candidate)
        if earliest_start is None:
            return False

        latest_end = cls._latest_end_date(candidate, eval_date)
        if latest_end is None:
            return False

        span_months = months_between(earliest_start, latest_end)
        experience_months = candidate.profile.years_of_experience * 12
        return abs(experience_months - span_months) > EXPERIENCE_SPAN_MISMATCH_THRESHOLD_MONTHS

    # ------------------------------------------------------------------
    # Rule 3 (Req 2.3): per-skill duration vs total career span
    # ------------------------------------------------------------------
    @classmethod
    def _violates_skill_duration_rule(
        cls, candidate: CandidateProfile, eval_date: date
    ) -> bool:
        """Flag if any skill's duration_months exceeds the candidate's total
        career span (earliest start_date → ``eval_date``, in months) by more
        than 12 months.

        When there are no valid start dates the span cannot be computed, so this
        span-based rule is skipped.
        """
        earliest_start = cls._earliest_start_date(candidate)
        if earliest_start is None:
            return False

        span_months = months_between(earliest_start, eval_date)
        for skill in candidate.skills:
            if skill.duration_months - span_months > SKILL_DURATION_ANOMALY_THRESHOLD_MONTHS:
                return True
        return False

    # ------------------------------------------------------------------
    # Rule 4 (Req 2.4): expert skills without endorsements
    # ------------------------------------------------------------------
    @staticmethod
    def _violates_expert_endorsement_rule(candidate: CandidateProfile) -> bool:
        """Flag if the candidate has 10 or more "expert" skills AND the sum of
        endorsements across those expert skills is 0.
        """
        expert_skills = [
            skill
            for skill in candidate.skills
            if skill.proficiency.strip().lower() == EXPERT_PROFICIENCY
        ]
        if len(expert_skills) < EXPERT_SKILL_COUNT_THRESHOLD:
            return False
        total_endorsements = sum(skill.endorsements for skill in expert_skills)
        return total_endorsements == 0

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _earliest_start_date(candidate: CandidateProfile) -> Optional[date]:
        """Return the earliest non-None career start_date, or ``None``."""
        starts = [
            entry.start_date
            for entry in candidate.career_history
            if entry.start_date is not None
        ]
        if not starts:
            return None
        return min(starts)

    @staticmethod
    def _latest_end_date(candidate: CandidateProfile, eval_date: date) -> Optional[date]:
        """Return the latest career end_date, treating current / missing
        end_dates as ``eval_date``.

        Only entries that have a usable start_date contribute, so that a stray
        entry with no dates at all does not affect the span. Returns ``None``
        when no entry has a start_date.
        """
        ends: List[date] = []
        for entry in candidate.career_history:
            if entry.start_date is None:
                continue
            if entry.is_current or entry.end_date is None:
                ends.append(eval_date)
            else:
                ends.append(entry.end_date)
        if not ends:
            return None
        return max(ends)
