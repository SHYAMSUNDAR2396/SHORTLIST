"""Location and work-mode preference scoring for the Candidate Ranking System.

The :class:`LocationWorkModeScorer` evaluates how well a candidate's location
and preferred work mode align with the "Senior AI Engineer — Founding Team"
role, which is based in Pune/Noida, India (Requirement 12). It is a pure
function over a :class:`CandidateProfile`: no global state, no clock reads, and
no randomness, so the score is deterministic and property-testable.

Design references (see design.md "Component: LocationWorkModeScorer" and
Requirement 12):

The final ``location_work_mode_score`` is the product of two independent
sub-scores, each exposed as a public helper method so they can be targeted in
isolation by property test 12.2:

- :meth:`LocationWorkModeScorer.location_fit` (Req 12.1, 12.2, 12.3) — all
  comparisons are case-insensitive:
    * country == "India" and location contains "Pune" or "Noida" → 1.0
    * country == "India" and location does NOT contain Pune/Noida →
      0.8 if ``willing_to_relocate`` else 0.6
    * country != "India" → 0.4 if ``willing_to_relocate`` else 0.2

- :meth:`LocationWorkModeScorer.work_mode_fit` (Req 12.4, 12.5) — the
  ``preferred_work_mode`` string is normalized (stripped + lowercased):
    * "hybrid", "onsite", or "flexible" → 1.0
    * "remote" → 0.7
    * **Fallback:** any other value (including empty/missing or an
      unrecognized mode) is treated as "remote" and scored 0.7. Remote is the
      least-aligned recognized mode for this onsite-leaning role, so defaulting
      unknown modes to it is the conservative choice and avoids rewarding
      candidates whose work-mode signal is absent or malformed.

Both sub-scores lie in ``[0.0, 1.0]``, so their product (the composite
``location_work_mode_score``) is also in ``[0.0, 1.0]``.

The inputs are drawn as follows:
- ``country`` and ``location`` from ``candidate.profile``.
- ``preferred_work_mode`` and ``willing_to_relocate`` from
  ``candidate.redrob_signals``.
"""

from ranking.models import CandidateProfile

# Location-fit constants (Req 12.1, 12.2, 12.3).
TARGET_COUNTRY: str = "india"
PREFERRED_CITIES: tuple = ("pune", "noida")

LOCATION_FIT_TARGET_CITY: float = 1.0
LOCATION_FIT_INDIA_RELOCATE: float = 0.8
LOCATION_FIT_INDIA_NO_RELOCATE: float = 0.6
LOCATION_FIT_FOREIGN_RELOCATE: float = 0.4
LOCATION_FIT_FOREIGN_NO_RELOCATE: float = 0.2

# Work-mode-fit constants (Req 12.4, 12.5).
ONSITE_LEANING_MODES: frozenset = frozenset({"hybrid", "onsite", "flexible"})
REMOTE_MODE: str = "remote"

WORK_MODE_FIT_ONSITE_LEANING: float = 1.0
WORK_MODE_FIT_REMOTE: float = 0.7


class LocationWorkModeScorer:
    """Scores location/work-mode fit, returning a value in [0.0, 1.0]."""

    def score(self, candidate: CandidateProfile) -> float:
        """Return ``location_work_mode_score = location_fit × work_mode_fit``.

        The result lies in ``[0.0, 1.0]`` because both sub-scores do
        (Requirement 12).

        Args:
            candidate: The candidate profile to score.

        Returns:
            A float in ``[0.0, 1.0]``.
        """
        return self.location_fit(candidate) * self.work_mode_fit(candidate)

    # ------------------------------------------------------------------
    # Sub-score helpers (public so property test 12.2 can target them)
    # ------------------------------------------------------------------
    def location_fit(self, candidate: CandidateProfile) -> float:
        """Return the location-fit sub-score in ``[0.0, 1.0]`` (Req 12.1–12.3).

        Case-insensitive comparisons on country and location:
        - India + Pune/Noida → 1.0
        - India elsewhere → 0.8 (relocate) / 0.6 (no relocate)
        - non-India → 0.4 (relocate) / 0.2 (no relocate)
        """
        country = (candidate.profile.country or "").strip().lower()
        location = (candidate.profile.location or "").lower()
        willing_to_relocate = bool(candidate.redrob_signals.willing_to_relocate)

        if country == TARGET_COUNTRY:
            if any(city in location for city in PREFERRED_CITIES):
                return LOCATION_FIT_TARGET_CITY
            return (
                LOCATION_FIT_INDIA_RELOCATE
                if willing_to_relocate
                else LOCATION_FIT_INDIA_NO_RELOCATE
            )

        return (
            LOCATION_FIT_FOREIGN_RELOCATE
            if willing_to_relocate
            else LOCATION_FIT_FOREIGN_NO_RELOCATE
        )

    def work_mode_fit(self, candidate: CandidateProfile) -> float:
        """Return the work-mode-fit sub-score in ``[0.0, 1.0]`` (Req 12.4–12.5).

        - hybrid/onsite/flexible → 1.0
        - remote → 0.7
        - any other/missing value falls back to remote (0.7); see module
          docstring for the rationale behind this conservative default.
        """
        mode = (candidate.redrob_signals.preferred_work_mode or "").strip().lower()
        if mode in ONSITE_LEANING_MODES:
            return WORK_MODE_FIT_ONSITE_LEANING
        # "remote" and every unrecognized/missing mode score as remote.
        return WORK_MODE_FIT_REMOTE
