"""Experience-fit scoring for the Candidate Ranking System.

This module implements the :class:`ExperienceScorer`, which evaluates how well a
candidate's years of experience align with the 5-9 year target range for the
"Senior AI Engineer — Founding Team" role.

Scoring proceeds in two stages (see Requirement 5):

1. **Experience validation (Req 5.5, 5.6).** The stated
   ``profile.years_of_experience`` is cross-checked against a value derived from
   the candidate's ``career_history``. The career-derived total is the length of
   the *union* of each role's date interval (overlapping roles counted once),
   expressed in years. If the absolute difference between the stated value and
   the career-derived value exceeds 2.0 years, the career-derived value is used
   in its place. When ``career_history`` is empty or every role has
   ``duration_months == 0`` (so no span can be derived), the stated value is
   used unchanged with no validation.

2. **Piecewise experience-fit function (Req 5.1-5.4).** The effective years
   value ``v`` is mapped to a score in ``[0.0, 1.0]``:

   ===========================  ==========================================
   ``v`` range                  score
   ===========================  ==========================================
   ``[5.0, 9.0]``               ``1.0``
   ``[4.0, 5.0)``               ``0.5 + 0.5 * ((v - 4.0) / 1.0)``
   ``(9.0, 11.0]``              ``1.0 - 0.5 * ((v - 9.0) / 2.0)``
   ``v < 4.0`` or ``v > 11.0``  ``0.2``
   ===========================  ==========================================

Career-derived-years approach (the "union of date ranges"):

Each career entry is converted to a month interval ``[start, end)`` where
``start`` is the entry's ``start_date`` and ``end = start + duration_months``
(computed with :func:`add_months`). We deliberately derive ``end`` from
``duration_months`` rather than from ``end_date`` so the computation does not
depend on an evaluation date and stays deterministic. Entries without a
``start_date`` or with ``duration_months <= 0`` contribute no interval. The
intervals are then sorted and merged (overlapping or adjacent intervals are
combined), the merged lengths are summed in months, and the total is divided by
12 to yield years. Merging guarantees overlapping roles are counted only once.

All functions here are pure: they read only their inputs (no clock, no
randomness), so repeated calls on the same candidate yield identical results.
"""

from datetime import date, timedelta
from typing import List, Optional, Tuple

from ranking.models import CandidateProfile, CareerEntry

_ONE_DAY = timedelta(days=1)

# Validation override threshold in years (Req 5.5).
VALIDATION_THRESHOLD_YEARS: float = 2.0

# Out-of-band experience score (Req 5.4).
OUT_OF_RANGE_SCORE: float = 0.2


def add_months(d: date, n: int) -> date:
    """Return the date ``n`` months after ``d`` (``n`` may be 0 or negative).

    Month arithmetic is performed safely: the resulting day is clamped to the
    last valid day of the target month (e.g. Jan 31 + 1 month -> Feb 28/29).
    """
    # Zero-based month index from year 0 makes the carry arithmetic simple.
    month_index = (d.year * 12 + (d.month - 1)) + n
    year, month0 = divmod(month_index, 12)
    month = month0 + 1

    # Clamp the day to the last day of the target month.
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    last_day = (next_month_first - _ONE_DAY).day
    day = min(d.day, last_day)
    return date(year, month, day)


def _months_between(start: date, end: date) -> float:
    """Return the number of months between ``start`` and ``end`` (end >= start).

    Computed as whole-month difference plus a fractional day component so that
    partial months contribute proportionally. Returns 0.0 when ``end`` is not
    after ``start``.
    """
    if end <= start:
        return 0.0
    whole = (end.year - start.year) * 12 + (end.month - start.month)
    # Adjust the whole-month count down if the end day precedes the start day,
    # then add the leftover days as a fraction of ~30.44 days/month.
    anchor = add_months(start, whole)
    if anchor > end:
        whole -= 1
        anchor = add_months(start, whole)
    leftover_days = (end - anchor).days
    return whole + (leftover_days / 30.44)


def _entry_interval(entry: CareerEntry) -> Optional[Tuple[date, date]]:
    """Convert a career entry to a ``(start, end)`` interval, or ``None``.

    ``end`` is derived from ``duration_months`` (``start + duration_months``).
    Entries lacking a ``start_date`` or with non-positive ``duration_months``
    yield no interval and are skipped.
    """
    start = entry.start_date
    if start is None:
        return None
    duration = entry.duration_months
    if duration is None or duration <= 0:
        return None
    return (start, add_months(start, duration))


def career_derived_years(career_history: List[CareerEntry]) -> Optional[float]:
    """Compute total experience in years from the union of career intervals.

    Each role becomes a ``[start, start + duration_months)`` month interval;
    overlapping or adjacent intervals are merged so shared time is counted only
    once. The merged span (in months) is summed and divided by 12.

    Returns ``None`` when no span can be derived (empty history, or every entry
    missing a start_date / having ``duration_months <= 0``). A ``None`` result
    signals callers to skip validation (Req 5.6).
    """
    intervals: List[Tuple[date, date]] = []
    for entry in career_history:
        interval = _entry_interval(entry)
        if interval is not None:
            intervals.append(interval)

    if not intervals:
        return None

    # Sort by start date, then merge overlapping/adjacent intervals.
    intervals.sort(key=lambda iv: iv[0])
    merged: List[Tuple[date, date]] = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            # Overlapping or adjacent: extend the current interval.
            if end > cur_end:
                cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    total_months = sum(_months_between(s, e) for s, e in merged)
    return total_months / 12.0


def experience_fit(v: float) -> float:
    """Piecewise experience-fit function over the effective years value ``v``.

    See the module docstring for the full mapping (Req 5.1-5.4). Always returns
    a value in ``[0.0, 1.0]``.
    """
    if 5.0 <= v <= 9.0:
        return 1.0
    if 4.0 <= v < 5.0:
        return 0.5 + 0.5 * ((v - 4.0) / 1.0)
    if 9.0 < v <= 11.0:
        return 1.0 - 0.5 * ((v - 9.0) / 2.0)
    return OUT_OF_RANGE_SCORE


def effective_years(candidate: CandidateProfile) -> float:
    """Resolve the years value to score after applying validation (Req 5.5/5.6).

    Uses the stated ``years_of_experience`` unless a career-derived total can be
    computed and differs from it by more than ``VALIDATION_THRESHOLD_YEARS``, in
    which case the career-derived total is used.
    """
    stated = candidate.profile.years_of_experience
    derived = career_derived_years(candidate.career_history)
    if derived is None:
        # Req 5.6: no derivable span -> use stated value unchanged.
        return stated
    if abs(stated - derived) > VALIDATION_THRESHOLD_YEARS:
        # Req 5.5: stated value disagrees materially -> trust career history.
        return derived
    return stated


class ExperienceScorer:
    """Scores how well a candidate's experience matches the 5-9 year target."""

    def score(self, candidate: CandidateProfile) -> float:
        """Return the experience_fit_score in ``[0.0, 1.0]`` (Req 5.1-5.6)."""
        v = effective_years(candidate)
        return experience_fit(v)
