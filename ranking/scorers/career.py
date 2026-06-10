"""Career-history analysis scorer for the Candidate Ranking System.

Implements :class:`CareerAnalyzer`, which derives a ``career_score`` in
``[0.0, 1.0]`` from a candidate's ``career_history``. The score blends two
sub-signals:

- **production_experience_score** (Req 4.1, 4.4, 4.5, 4.7) — the share of the
  candidate's total career duration spent in genuine product-company roles,
  with each role's months weighted by how production-oriented (vs. purely
  research-oriented) its description is. Job-hopping (Req 4.3) and
  large-enterprise-only (Req 4.7) adjustments apply.
- **title_relevance_score** (Req 4.6) — a duration-weighted average of
  per-role title relevance (AI/ML titles = 1.0, other technical titles = 0.5,
  non-technical titles = 0.0).

A consulting-heavy career (Req 4.2) forces the final ``career_score`` to 0.0.

Design choices
--------------
- **Blend weights**: ``career_score = 0.6 * production_experience_score +
  0.4 * title_relevance_score``. Production experience is the dominant signal
  for a hands-on founding-team engineer, so it is weighted higher; title
  relevance is a meaningful but secondary indicator (titles are noisier than
  demonstrated production work).
- **Adjustment order for production_experience_score**: compute the
  weighted product-company ratio and clamp it to ``[0.0, 1.0]`` (Req 4.1
  defines this score as a bounded ratio; relevance weights can push the raw
  numerator above the denominator), then apply the Req 4.7 large-enterprise
  cap (ceiling of 0.2), then apply the Req 4.3 job-hopping multiplier (×0.5).
  Clamping the base before the multiplier keeps the ×0.5 penalty observable
  rather than absorbed by a saturated raw ratio.
- **Consulting zeroing is applied last** as an override on the blended score.

This module is a pure function of its inputs (no clock, no randomness).
"""

import re
from typing import List

from ranking.models import CandidateProfile, CareerEntry
from ranking.constants import (
    CONSULTING_FIRMS,
    PRODUCTION_KEYWORDS,
    RESEARCH_KEYWORDS,
    PRODUCT_COMPANY_SIZES,
    LARGE_ENTERPRISE_SIZE,
    AI_ML_TITLE_TERMS,
    TECHNICAL_TITLE_TERMS,
)

# Blend weights for the final career_score (documented above).
PRODUCTION_BLEND_WEIGHT: float = 0.6
TITLE_BLEND_WEIGHT: float = 0.4

# Relevance multipliers applied to a role's duration when computing the
# production_experience_score numerator (Req 4.4, 4.5).
PRODUCTION_RELEVANCE_WEIGHT: float = 2.0
RESEARCH_RELEVANCE_WEIGHT: float = 0.3
NEUTRAL_RELEVANCE_WEIGHT: float = 1.0

# Req 4.7 cap and Req 4.3 penalty.
LARGE_ENTERPRISE_CAP: float = 0.2
JOB_HOPPING_MULTIPLIER: float = 0.5
JOB_HOPPING_MIN_SHORT_ROLES: int = 3
JOB_HOPPING_SHORT_MONTHS: int = 18

# Req 4.2 consulting-share threshold.
CONSULTING_SHARE_THRESHOLD: float = 0.80

# Pre-compiled word-boundary patterns for the ambiguous short AI/ML terms so
# that "ai" does not match "training"/"email" and "ml" does not match "html".
_AI_ML_WORD_PATTERNS = {
    term: re.compile(r"\b" + re.escape(term) + r"\b")
    for term in AI_ML_TITLE_TERMS
}


def _clamp(value: float) -> float:
    """Clamp a score into the closed range [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _safe_months(entry: CareerEntry) -> int:
    """Return a non-negative duration in months for a role (0 if absent)."""
    months = entry.duration_months or 0
    return months if months > 0 else 0


def _description_has_production_keyword(description: str) -> bool:
    """True when the role description mentions any production-deployment term."""
    if not description:
        return False
    text = description.lower()
    return any(keyword in text for keyword in PRODUCTION_KEYWORDS)


def _description_has_research_keyword(description: str) -> bool:
    """True when the role description mentions any research-indicating term."""
    if not description:
        return False
    text = description.lower()
    return any(keyword in text for keyword in RESEARCH_KEYWORDS)


def _role_relevance_weight(entry: CareerEntry) -> float:
    """Per-role relevance multiplier for the production numerator (Req 4.4/4.5).

    - 2.0 when the description contains a production-deployment keyword.
    - 0.3 when it contains a research keyword and NO production keyword.
    - 1.0 otherwise.
    """
    description = entry.description or ""
    if _description_has_production_keyword(description):
        return PRODUCTION_RELEVANCE_WEIGHT
    if _description_has_research_keyword(description):
        return RESEARCH_RELEVANCE_WEIGHT
    return NEUTRAL_RELEVANCE_WEIGHT


def _is_product_company(entry: CareerEntry) -> bool:
    """True when the role's company_size falls in the product-company set."""
    size = (entry.company_size or "").strip()
    return size in PRODUCT_COMPANY_SIZES


def _is_consulting_company(company: str) -> bool:
    """True when a company name matches a known consulting firm.

    Matching is case-insensitive; both exact membership and substring
    containment are accepted so that "TCS", "Infosys Ltd", or
    "Tata Consultancy (TCS)" all resolve as consulting firms.
    """
    if not company:
        return False
    normalized = company.strip().lower()
    if not normalized:
        return False
    if normalized in CONSULTING_FIRMS:
        return True
    return any(firm in normalized for firm in CONSULTING_FIRMS)


def _title_weight(title: str) -> float:
    """Per-role title relevance weight (Req 4.6).

    - 1.0 when the title contains an AI/ML term (word-boundary aware for the
      ambiguous short tokens "ai" and "ml").
    - 0.5 when the title contains a general technical term but no AI/ML term.
    - 0.0 when the title has no technical relevance at all.
    """
    if not title:
        return 0.0
    text = title.lower()

    for term, pattern in _AI_ML_WORD_PATTERNS.items():
        if len(term) <= 3:
            # Ambiguous short tokens ("ai", "ml", "nlp"): require word boundary.
            if pattern.search(text):
                return 1.0
        else:
            # Multi-word phrases ("machine learning", "data science"): substring.
            if term in text:
                return 1.0

    if any(term in text for term in TECHNICAL_TITLE_TERMS):
        return 0.5

    return 0.0


class CareerAnalyzer:
    """Analyze career history into a normalized career_score in [0.0, 1.0]."""

    def production_experience_score(self, candidate: CandidateProfile) -> float:
        """Compute the production_experience_score (Req 4.1, 4.3, 4.4, 4.5, 4.7).

        = sum(relevance_weight * months) over product-company roles
          / sum(months) over all roles,
        clamped to [0.0, 1.0], then capped at 0.2 when all roles are at
        10001+ companies with no production keywords, then multiplied by 0.5
        when a job-hopping pattern is present. Result is in [0.0, 1.0].
        """
        roles = candidate.career_history or []
        if not roles:
            return 0.0

        total_months = sum(_safe_months(role) for role in roles)
        if total_months <= 0:
            return 0.0

        weighted_product_months = 0.0
        for role in roles:
            if _is_product_company(role):
                months = _safe_months(role)
                weighted_product_months += _role_relevance_weight(role) * months

        score = weighted_product_months / total_months

        # Req 4.1 defines this score as a bounded ratio. Relevance weights
        # (×2.0) can push the numerator above the denominator, so clamp the
        # base ratio into [0.0, 1.0] before applying the cap/penalty so that
        # those adjustments act on a properly bounded value.
        score = _clamp(score)

        # Req 4.7: cap at 0.2 when every role is at a 10001+ company and none
        # of the descriptions contain a production keyword.
        if self._all_large_enterprise_no_production(roles):
            score = min(score, LARGE_ENTERPRISE_CAP)

        # Req 4.3: job-hopping stability penalty.
        if self._is_job_hopping(roles):
            score *= JOB_HOPPING_MULTIPLIER

        return _clamp(score)

    def title_relevance_score(self, candidate: CandidateProfile) -> float:
        """Duration-weighted average of per-role title weights (Req 4.6)."""
        roles = candidate.career_history or []
        if not roles:
            return 0.0

        total_months = sum(_safe_months(role) for role in roles)
        if total_months <= 0:
            return 0.0

        weighted = 0.0
        for role in roles:
            weighted += _title_weight(role.title or "") * _safe_months(role)

        return _clamp(weighted / total_months)

    def consulting_share(self, candidate: CandidateProfile) -> float:
        """Fraction of total career months spent at consulting firms (Req 4.2).

        Returns 0.0 when there is no measurable career duration.
        """
        roles = candidate.career_history or []
        if not roles:
            return 0.0

        total_months = sum(_safe_months(role) for role in roles)
        if total_months <= 0:
            return 0.0

        consulting_months = sum(
            _safe_months(role)
            for role in roles
            if _is_consulting_company(role.company or "")
        )
        return consulting_months / total_months

    def score(self, candidate: CandidateProfile) -> float:
        """Return the blended career_score in [0.0, 1.0].

        career_score = 0.6 * production_experience_score
                     + 0.4 * title_relevance_score,
        overridden to 0.0 when >80% of career months are at consulting firms.
        """
        roles = candidate.career_history or []
        if not roles:
            return 0.0

        production = self.production_experience_score(candidate)
        title = self.title_relevance_score(candidate)

        career_score = (
            PRODUCTION_BLEND_WEIGHT * production + TITLE_BLEND_WEIGHT * title
        )

        # Req 4.2: consulting-heavy career zeroing (applied last as override).
        if self.consulting_share(candidate) > CONSULTING_SHARE_THRESHOLD:
            return 0.0

        return _clamp(career_score)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _all_large_enterprise_no_production(roles: List[CareerEntry]) -> bool:
        """True when every role is at a 10001+ company with no prod keyword."""
        if not roles:
            return False
        for role in roles:
            if (role.company_size or "").strip() != LARGE_ENTERPRISE_SIZE:
                return False
            if _description_has_production_keyword(role.description or ""):
                return False
        return True

    @staticmethod
    def _is_job_hopping(roles: List[CareerEntry]) -> bool:
        """True when >=3 roles each have duration_months < 18 (Req 4.3)."""
        short_roles = sum(
            1
            for role in roles
            if 0 <= _safe_months(role) < JOB_HOPPING_SHORT_MONTHS
        )
        return short_roles >= JOB_HOPPING_MIN_SHORT_ROLES
