"""Disqualification filtering for the Candidate Ranking System.

The :class:`DisqualifierFilter` applies the explicit disqualification rules
from the job description (Requirement 6). Each rule, when triggered, carries a
penalty *multiplier* that is applied to the candidate's composite score. When
several rules trigger simultaneously, only the single most-severe (lowest)
multiplier is applied — multipliers are never compounded (Req 6.5).

Criteria (Requirement 6):

- ``recent_ai_only`` (Req 6.1, multiplier 0.1) — every AI-related role started
  within the last 12 months (relative to ``eval_date``) AND the candidate has
  no traditional-ML skill with ``duration_months > 6``. Only triggers when the
  candidate has at least one AI-related role.
- ``cv_speech_robotics`` (Req 6.2, multiplier 0.3) — more than 70% of total
  skill ``duration_months`` is attributed to CV/speech/robotics-category skills
  AND the candidate has no NLP/IR-category skill with ``duration_months >= 1``.
- ``no_external_validation`` (Req 6.3, multiplier 0.5) — total career duration
  exceeds 60 months, ``github_activity_score <= 0``, ``certifications`` is
  empty, and no role description references open-source / public repositories.
- ``all_consulting`` (Req 6.4, multiplier 0.1) — every career_history entry's
  company is a consulting firm (career_history must be non-empty).

Determinism (design "Determinism Guarantees", Req 10.5)
-------------------------------------------------------
This module is a pure function of its inputs. The evaluation date is threaded
in via ``eval_date`` (used only by ``recent_ai_only``); the system clock is
never read. Criteria are evaluated in a fixed order (:data:`CRITERION_ORDER`)
so that when multiple criteria share the same lowest multiplier the tie is
broken deterministically by that order.
"""

import re
from datetime import date
from typing import List, Optional, Tuple

from ranking.models import CandidateProfile
from ranking.constants import (
    CONSULTING_FIRMS,
    CV_SPEECH_ROBOTICS_CATEGORIES,
    NLP_IR_CATEGORIES,
)

# ---------------------------------------------------------------------------
# Criterion names and their penalty multipliers (Req 6.1-6.4).
# ---------------------------------------------------------------------------
CRITERION_RECENT_AI_ONLY: str = "recent_ai_only"
CRITERION_CV_SPEECH_ROBOTICS: str = "cv_speech_robotics"
CRITERION_NO_EXTERNAL_VALIDATION: str = "no_external_validation"
CRITERION_ALL_CONSULTING: str = "all_consulting"

MULTIPLIER_RECENT_AI_ONLY: float = 0.1
MULTIPLIER_CV_SPEECH_ROBOTICS: float = 0.3
MULTIPLIER_NO_EXTERNAL_VALIDATION: float = 0.5
MULTIPLIER_ALL_CONSULTING: float = 0.1

# Fixed evaluation order. Used both to evaluate criteria and to break ties when
# multiple triggered criteria share the same lowest multiplier (Req 6.5): the
# earliest entry in this list wins. ``recent_ai_only`` precedes
# ``all_consulting`` so a 0.1/0.1 tie resolves to ``recent_ai_only``.
CRITERION_ORDER: List[str] = [
    CRITERION_RECENT_AI_ONLY,
    CRITERION_CV_SPEECH_ROBOTICS,
    CRITERION_NO_EXTERNAL_VALIDATION,
    CRITERION_ALL_CONSULTING,
]

# ---------------------------------------------------------------------------
# Keyword tables (pragmatic, lowercase).
# ---------------------------------------------------------------------------
# AI-related keywords used to identify "recent AI" roles by title/description
# (Req 6.1). Short ambiguous tokens are matched on word boundaries (below) so
# that "ai" does not match "email"/"training" and "ml" does not match "html".
AI_KEYWORDS: List[str] = [
    "ai", "ml", "machine learning", "llm", "deep learning",
    "nlp", "neural", "embeddings", "transformer",
]

# Traditional / classical ML skill names (Req 6.1). A candidate with one of
# these skills held for more than 6 months is NOT considered "recent AI only".
TRADITIONAL_ML_SKILLS: List[str] = [
    "statistical modeling", "feature engineering", "regression",
    "classical ml", "scikit-learn", "xgboost", "random forest",
]
TRADITIONAL_ML_MIN_MONTHS: int = 6

# Keywords in a role description indicating external validation via
# open-source / public repositories (Req 6.3).
EXTERNAL_VALIDATION_KEYWORDS: List[str] = [
    "open source", "open-source", "github", "public repo", "oss", "contributor",
]

# Req 6.1 recency window and Req 6.3 tenure threshold (in months).
RECENT_AI_WINDOW_MONTHS: int = 12
NO_VALIDATION_MIN_CAREER_MONTHS: int = 60

# Word-boundary patterns for the ambiguous short AI tokens.
_SHORT_AI_TOKENS = {"ai", "ml", "nlp"}
_AI_WORD_PATTERNS = {
    token: re.compile(r"\b" + re.escape(token) + r"\b") for token in _SHORT_AI_TOKENS
}


def _months_between(d1: date, d2: date) -> int:
    """Return whole months from ``d1`` to ``d2`` (negative when d2 precedes d1).

    Uses the same ``(year, month)`` convention as the rest of the pipeline so
    month-span math is consistent across modules.
    """
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def _safe_months(months: Optional[int]) -> int:
    """Treat negative/absent durations as 0."""
    if not months or months < 0:
        return 0
    return months


def _text_has_keyword(text: str, keywords: List[str]) -> bool:
    """Case-insensitive substring match of any keyword within ``text``.

    Short ambiguous AI tokens (``ai``/``ml``/``nlp``) are matched on word
    boundaries to avoid false positives inside larger words.
    """
    if not text:
        return False
    lowered = text.lower()
    for keyword in keywords:
        if keyword in _SHORT_AI_TOKENS:
            if _AI_WORD_PATTERNS[keyword].search(lowered):
                return True
        elif keyword in lowered:
            return True
    return False


def _is_ai_role(title: str, description: str) -> bool:
    """True when a role's title or description references an AI keyword."""
    return _text_has_keyword(title or "", AI_KEYWORDS) or _text_has_keyword(
        description or "", AI_KEYWORDS
    )


def _skill_matches_category(skill_name: str, categories) -> bool:
    """True when a skill name matches any category (case-insensitive).

    A match is an exact equality or a substring containment of the category
    within the skill name (e.g. "computer vision engineer" matches the
    "computer vision" category).
    """
    if not skill_name:
        return False
    normalized = skill_name.strip().lower()
    if not normalized:
        return False
    for category in categories:
        if normalized == category or category in normalized:
            return True
    return False


def _is_consulting_company(company: str) -> bool:
    """True when a company name matches a known consulting firm (Req 6.4).

    Matching is case-insensitive and accepts both exact membership and
    substring containment so "TCS", "Infosys Ltd", or "Tata Consultancy (TCS)"
    all resolve as consulting firms. Implemented locally to keep this module
    decoupled from the career scorer.
    """
    if not company:
        return False
    normalized = company.strip().lower()
    if not normalized:
        return False
    if normalized in CONSULTING_FIRMS:
        return True
    return any(firm in normalized for firm in CONSULTING_FIRMS)


class DisqualifierFilter:
    """Apply the most-severe triggered disqualification penalty (Req 6)."""

    def apply(
        self,
        candidate: CandidateProfile,
        composite_score: float,
        eval_date: date,
    ) -> Tuple[float, Optional[str]]:
        """Apply the single lowest triggered penalty multiplier (Req 6.5/6.6).

        Args:
            candidate: The candidate to evaluate.
            composite_score: The aggregated composite score before penalties.
            eval_date: The single evaluation date threaded through the pipeline
                (used by the ``recent_ai_only`` recency check).

        Returns:
            ``(penalized_score, triggered_criterion_name)``. When no criterion
            triggers, returns ``(composite_score, None)`` (multiplier 1.0).
            When one or more trigger, the lowest (most severe) multiplier is
            applied and its criterion name is returned; ties on the multiplier
            are broken by :data:`CRITERION_ORDER`.
        """
        triggered = self.triggered_criteria(candidate, eval_date)
        if not triggered:
            return composite_score, None

        # Pick the lowest multiplier; iterating in CRITERION_ORDER with a strict
        # ``<`` comparison makes the earliest-ordered criterion win on ties.
        best_name, best_multiplier = triggered[0]
        for name, multiplier in triggered[1:]:
            if multiplier < best_multiplier:
                best_name, best_multiplier = name, multiplier

        return composite_score * best_multiplier, best_name

    def triggered_criteria(
        self, candidate: CandidateProfile, eval_date: date
    ) -> List[Tuple[str, float]]:
        """Return all triggered ``(criterion_name, multiplier)`` pairs.

        Pairs are returned in the fixed :data:`CRITERION_ORDER`. Exposed so the
        minimum-penalty rule (Req 6.5) can be independently verified.
        """
        results: List[Tuple[str, float]] = []
        if self._triggers_recent_ai_only(candidate, eval_date):
            results.append((CRITERION_RECENT_AI_ONLY, MULTIPLIER_RECENT_AI_ONLY))
        if self._triggers_cv_speech_robotics(candidate):
            results.append(
                (CRITERION_CV_SPEECH_ROBOTICS, MULTIPLIER_CV_SPEECH_ROBOTICS)
            )
        if self._triggers_no_external_validation(candidate):
            results.append(
                (CRITERION_NO_EXTERNAL_VALIDATION, MULTIPLIER_NO_EXTERNAL_VALIDATION)
            )
        if self._triggers_all_consulting(candidate):
            results.append((CRITERION_ALL_CONSULTING, MULTIPLIER_ALL_CONSULTING))
        return results

    # ------------------------------------------------------------------
    # Criterion 1 (Req 6.1): recent AI-only career
    # ------------------------------------------------------------------
    @staticmethod
    def _triggers_recent_ai_only(
        candidate: CandidateProfile, eval_date: date
    ) -> bool:
        """True when every AI role is recent (<= 12 months) and there is no
        traditional-ML skill held for more than 6 months.

        Only triggers when at least one AI-related role exists; a role with a
        missing ``start_date`` cannot be confirmed recent, so its presence
        prevents the criterion from firing.
        """
        ai_roles = [
            entry
            for entry in (candidate.career_history or [])
            if _is_ai_role(entry.title or "", entry.description or "")
        ]
        if not ai_roles:
            return False

        for entry in ai_roles:
            if entry.start_date is None:
                return False
            age_months = _months_between(entry.start_date, eval_date)
            if age_months > RECENT_AI_WINDOW_MONTHS:
                return False

        # No traditional-ML skill with duration_months > 6.
        for skill in candidate.skills or []:
            if _skill_matches_category(skill.name or "", TRADITIONAL_ML_SKILLS):
                if _safe_months(skill.duration_months) > TRADITIONAL_ML_MIN_MONTHS:
                    return False

        return True

    # ------------------------------------------------------------------
    # Criterion 2 (Req 6.2): CV/speech/robotics-dominant with no NLP/IR
    # ------------------------------------------------------------------
    @staticmethod
    def _triggers_cv_speech_robotics(candidate: CandidateProfile) -> bool:
        """True when >70% of total skill duration is in CV/speech/robotics
        categories AND there is no NLP/IR skill with duration_months >= 1.

        Guards divide-by-zero: with no total skill duration the criterion does
        not trigger.
        """
        skills = candidate.skills or []
        total_months = sum(_safe_months(skill.duration_months) for skill in skills)
        if total_months <= 0:
            return False

        cv_months = sum(
            _safe_months(skill.duration_months)
            for skill in skills
            if _skill_matches_category(skill.name or "", CV_SPEECH_ROBOTICS_CATEGORIES)
        )
        if cv_months / total_months <= 0.70:
            return False

        # Disqualified only if the candidate has no NLP/IR skill (>= 1 month).
        for skill in skills:
            if _skill_matches_category(skill.name or "", NLP_IR_CATEGORIES):
                if _safe_months(skill.duration_months) >= 1:
                    return False

        return True

    # ------------------------------------------------------------------
    # Criterion 3 (Req 6.3): no external validation over 5+ years
    # ------------------------------------------------------------------
    @staticmethod
    def _triggers_no_external_validation(candidate: CandidateProfile) -> bool:
        """True when total career > 60 months, github_activity_score <= 0,
        certifications is empty, and no role description references
        open-source / public repositories.
        """
        roles = candidate.career_history or []
        total_months = sum(_safe_months(entry.duration_months) for entry in roles)
        if total_months <= NO_VALIDATION_MIN_CAREER_MONTHS:
            return False

        if candidate.redrob_signals.github_activity_score > 0:
            return False

        if candidate.certifications:
            return False

        for entry in roles:
            if _text_has_keyword(entry.description or "", EXTERNAL_VALIDATION_KEYWORDS):
                return False

        return True

    # ------------------------------------------------------------------
    # Criterion 4 (Req 6.4): all-consulting career
    # ------------------------------------------------------------------
    @staticmethod
    def _triggers_all_consulting(candidate: CandidateProfile) -> bool:
        """True when career_history is non-empty and every entry's company is
        a consulting firm.
        """
        roles = candidate.career_history or []
        if not roles:
            return False
        return all(_is_consulting_company(entry.company or "") for entry in roles)
