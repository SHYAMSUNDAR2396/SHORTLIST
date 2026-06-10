"""Behavioral signal scoring for the Candidate Ranking System.

The :class:`BehavioralSignalEvaluator` translates a candidate's Redrob platform
engagement signals into a single normalized ``behavioral_score`` in ``[0.0, 1.0]``.

The score is built by multiplicatively combining three independent modifiers,
starting from a neutral base of ``1.0`` (Requirement 7.1):

* **Engagement** — recruiter responsiveness (Req 7.2, 7.3)
* **Technical activity** — public GitHub activity (Req 7.4, 7.5, 7.6)
* **Staleness/recency** — how recently the candidate was active (Req 7.7, 7.8)

Normalization
-------------
The raw product ranges from ``0.8`` (a stale candidate with no positive
modifiers) up to ``1.2 * 1.15 = 1.38`` (a best-case candidate). To map this
achievable range into ``[0.0, 1.0]`` we divide the raw product by the maximum
achievable product (``MAX_RAW_PRODUCT = 1.2 * 1.15 = 1.38``). This keeps the
mapping monotonic (more positive signals never lower the score) and guarantees a
best-case candidate maps to exactly ``1.0``. The result is then clamped to
``[0.0, 1.0]`` as a defensive bound.

With this normalization:

* best case (response > 0.6, github > 50, recent) → ``1.38 / 1.38 = 1.0``
* worst case (low response, github ``-1`` neutral, stale) → ``0.8 / 1.38 ≈ 0.5797``

Determinism
-----------
The evaluation date is threaded in explicitly (``eval_date``) rather than read
from the system clock, so the score is a pure function of its inputs and is
reproducible across runs.

Edge cases
----------
* ``github_activity_score == -1`` (no GitHub linked) is treated as **neutral**
  (modifier ``1.0``), never a penalty (Req 7.4).
* ``last_active_date is None`` is treated as **neutral** (modifier ``1.0``)
  rather than stale, since we cannot establish that the candidate is inactive.
"""

from datetime import date
from typing import Optional

from ranking.models import CandidateProfile

# Maximum achievable raw product: best-case engagement (1.2) × best-case
# technical activity (1.15). The staleness modifier's best case is 1.0, so it
# does not contribute to the maximum. Used to normalize the raw product into
# [0.0, 1.0] such that a best-case candidate maps to exactly 1.0.
MAX_RAW_PRODUCT = 1.2 * 1.15  # == 1.38

# Threshold (in days) beyond which a candidate is considered stale.
STALENESS_THRESHOLD_DAYS = 180


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


class BehavioralSignalEvaluator:
    """Scores a candidate's Redrob behavioral signals into ``[0.0, 1.0]``.

    Each modifier is exposed as a small helper method so that the multiplicative
    combination can be targeted directly by property-based tests.
    """

    def engagement_modifier(self, recruiter_response_rate: float) -> float:
        """Engagement modifier from recruiter responsiveness.

        * ``1.2`` when ``recruiter_response_rate > 0.6`` (Req 7.2)
        * ``1.0`` (neutral) when ``recruiter_response_rate <= 0.6`` (Req 7.3)
        """
        return 1.2 if recruiter_response_rate > 0.6 else 1.0

    def technical_activity_modifier(self, github_activity_score: float) -> float:
        """Technical-activity modifier from GitHub activity.

        * ``1.15`` when ``github_activity_score > 50`` (Req 7.5)
        * ``1.0`` (neutral) when ``github_activity_score`` is in ``[-1, 50]``;
          this covers the "no GitHub linked" sentinel ``-1`` (Req 7.4) and the
          ``[0, 50]`` inclusive band (Req 7.6).

        Values below ``-1`` are also treated as neutral (``1.0``) — the scorer
        never penalizes for low/absent GitHub activity.
        """
        return 1.15 if github_activity_score > 50 else 1.0

    def staleness_modifier(
        self, last_active_date: Optional[date], eval_date: date
    ) -> float:
        """Staleness/recency modifier from last-active recency.

        * ``0.8`` when ``last_active_date`` is more than 180 days before
          ``eval_date`` (Req 7.7)
        * ``1.0`` (neutral) when within 180 days, inclusive (Req 7.8)
        * ``1.0`` (neutral) when ``last_active_date`` is ``None`` — recency
          cannot be established, so we do not penalize.
        """
        if last_active_date is None:
            return 1.0
        days_inactive = (eval_date - last_active_date).days
        return 0.8 if days_inactive > STALENESS_THRESHOLD_DAYS else 1.0

    def score(self, candidate: CandidateProfile, eval_date: date) -> float:
        """Return the candidate's behavioral score in ``[0.0, 1.0]``.

        Combines the three modifiers multiplicatively from a base of ``1.0``
        (Req 7.1), then normalizes the raw product by ``MAX_RAW_PRODUCT`` and
        clamps the result into ``[0.0, 1.0]``.
        """
        signals = candidate.redrob_signals

        engagement = self.engagement_modifier(signals.recruiter_response_rate)
        technical = self.technical_activity_modifier(signals.github_activity_score)
        staleness = self.staleness_modifier(signals.last_active_date, eval_date)

        raw_product = 1.0 * engagement * technical * staleness
        normalized = raw_product / MAX_RAW_PRODUCT
        return _clamp(normalized)
