# Feature: candidate-ranking-system, Property 19: Location and work-mode scoring
"""Property 19: Location and work-mode scoring.

**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**

location_fit (case-insensitive):
- India + Pune/Noida -> 1.0
- India elsewhere -> 0.8 (relocate) / 0.6 (no relocate)
- non-India -> 0.4 (relocate) / 0.2 (no relocate)

work_mode_fit:
- hybrid/onsite/flexible -> 1.0
- remote -> 0.7
- other/missing -> 0.7 (conservative remote fallback)

score == location_fit * work_mode_fit, and every value lies in [0.0, 1.0].
"""

from hypothesis import given
from hypothesis import strategies as st

from ranking.scorers.location import LocationWorkModeScorer
from tests.property._factories import build_candidate

COUNTRIES = ["India", "india", "INDIA", "USA", "usa", "Germany", "Canada", ""]
# Location strings: some contain Pune/Noida in varied case, some do not.
LOCATIONS = [
    "Pune",
    "pune",
    "PUNE",
    "Noida",
    "noida",
    "NOIDA",
    "Pune, Maharashtra",
    "Greater Noida",
    "Mumbai",
    "Bangalore",
    "Berlin",
    "New York",
    "",
]
WORK_MODES = ["hybrid", "onsite", "flexible", "remote", "", "Hybrid", "REMOTE", "weird"]

ONSITE_LEANING = {"hybrid", "onsite", "flexible"}


def _expected_location_fit(country: str, location: str, relocate: bool) -> float:
    country_norm = (country or "").strip().lower()
    location_norm = (location or "").lower()
    if country_norm == "india":
        if "pune" in location_norm or "noida" in location_norm:
            return 1.0
        return 0.8 if relocate else 0.6
    return 0.4 if relocate else 0.2


def _expected_work_mode_fit(mode: str) -> float:
    normalized = (mode or "").strip().lower()
    if normalized in ONSITE_LEANING:
        return 1.0
    return 0.7  # remote and any other/missing value


@given(
    country=st.sampled_from(COUNTRIES),
    location=st.sampled_from(LOCATIONS),
    relocate=st.booleans(),
    work_mode=st.sampled_from(WORK_MODES),
)
def test_location_work_mode_scoring(country, location, relocate, work_mode):
    scorer = LocationWorkModeScorer()
    candidate = build_candidate()
    candidate.profile.country = country
    candidate.profile.location = location
    candidate.redrob_signals.willing_to_relocate = relocate
    candidate.redrob_signals.preferred_work_mode = work_mode

    location_fit = scorer.location_fit(candidate)
    work_mode_fit = scorer.work_mode_fit(candidate)
    score = scorer.score(candidate)

    expected_location = _expected_location_fit(country, location, relocate)
    expected_work_mode = _expected_work_mode_fit(work_mode)

    assert location_fit == expected_location
    assert work_mode_fit == expected_work_mode
    assert score == expected_location * expected_work_mode

    assert 0.0 <= location_fit <= 1.0
    assert 0.0 <= work_mode_fit <= 1.0
    assert 0.0 <= score <= 1.0
