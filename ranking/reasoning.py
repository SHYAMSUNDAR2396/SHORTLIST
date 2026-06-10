"""Reasoning string generation for the Candidate Ranking System.

The :class:`ReasoningGenerator` produces the human-readable ``reasoning`` field
emitted in the final submission CSV for each ranked candidate (Requirement 11).
It is a pure, deterministic function over a :class:`CandidateProfile` and the
candidate's :class:`ScoredCandidate` record — no clock reads, no randomness — so
output is bit-identical across runs.

Each generated string satisfies the following guarantees (Req 11.1, 11.2, 11.3,
11.5):

- **Content (Req 11.1):** includes the candidate's current title, years of
  experience, and the highest-contributing scoring component (the component
  whose ``weight × clamped_component_value`` is largest).
- **Specificity (Req 11.3):** references at least one concrete attribute drawn
  from the candidate's profile data — a matched required-skill name, the current
  company, or a real career-history title. Only data actually present in the
  profile is referenced (never a hallucinated skill).
- **Length (Req 11.2):** between 20 and 200 characters inclusive. The title and
  the specific attribute are truncated so the assembled template always fits;
  unusually short data is padded with a truthful clause.
- **CSV safety (Req 11.5):** contains no line-break characters and collapses all
  internal whitespace, so the value is a valid single CSV field when quoted.

The format mirrors the sample submission style, using semicolons as separators::

    "{current_title} with {yoe:.1f} yrs; top factor {label}; {specific_attribute}."

The helper :func:`top_contributing_component` computes the highest-contributing
component from a :class:`ScoredCandidate` and is exposed at module level so the
property test (Task 17.2) can target it directly.
"""

from typing import Dict, Optional, Tuple

from .constants import WEIGHTS, skill_to_group
from .models import CandidateProfile, ScoredCandidate

# Minimum and maximum allowed reasoning length (Req 11.2), inclusive.
MIN_LENGTH: int = 20
MAX_LENGTH: int = 200

# Per-component budgets so the assembled template always fits within MAX_LENGTH
# while still including every required element. These are generous; the final
# string is still hard-clamped to MAX_LENGTH as a defensive backstop.
MAX_TITLE_CHARS: int = 60
MAX_ATTRIBUTE_CHARS: int = 60

# Human-readable label for each scoring component (Req 11.1).
COMPONENT_LABELS: Dict[str, str] = {
    "skill": "skill match",
    "career": "career fit",
    "experience": "experience fit",
    "behavioral": "engagement",
    "education": "education",
    "location_work_mode": "location fit",
}

# Map each composite component to the corresponding ScoredCandidate attribute.
# Ordered to match WEIGHTS so iteration is deterministic and ties resolve in a
# fixed, documented order (skill first, then career, ...).
_COMPONENT_SCORE_ATTRS: Tuple[Tuple[str, str], ...] = (
    ("skill", "skill_score"),
    ("career", "career_score"),
    ("experience", "experience_score"),
    ("behavioral", "behavioral_score"),
    ("education", "education_score"),
    ("location_work_mode", "location_work_mode_score"),
)


def _clamp01(x: float) -> float:
    """Clamp a value to ``[0.0, 1.0]`` (mirrors composite clamping, Req 8.3)."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def component_contributions(scored: ScoredCandidate) -> Dict[str, float]:
    """Return each component's weighted contribution ``weight × clamp(value)``.

    The contribution of a component is the same quantity that the composite
    score sums over (Req 8.1): its weight from :data:`ranking.constants.WEIGHTS`
    multiplied by its clamped component score. This is what determines which
    component contributed the highest absolute points (Req 11.1).
    """
    contributions: Dict[str, float] = {}
    for component, attr in _COMPONENT_SCORE_ATTRS:
        weight = WEIGHTS.get(component, 0.0)
        value = getattr(scored, attr, 0.0)
        contributions[component] = weight * _clamp01(value)
    return contributions


def top_contributing_component(scored: ScoredCandidate) -> Tuple[str, float]:
    """Return the highest-contributing component as ``(name, contribution)``.

    The "top factor" is the component whose ``weight × clamped_component_value``
    is largest (Req 11.1). Ties are broken deterministically by the fixed
    component order in :data:`_COMPONENT_SCORE_ATTRS` (skill, career,
    experience, behavioral, education, location_work_mode): the first component
    achieving the maximum wins.

    Exposed at module level so the property test can target the selection logic
    independently of string formatting.
    """
    best_component = _COMPONENT_SCORE_ATTRS[0][0]
    best_value = -1.0
    for component, attr in _COMPONENT_SCORE_ATTRS:
        weight = WEIGHTS.get(component, 0.0)
        contribution = weight * _clamp01(getattr(scored, attr, 0.0))
        if contribution > best_value:
            best_value = contribution
            best_component = component
    return best_component, best_value


def _sanitize(text: str) -> str:
    """Collapse all whitespace (including newlines) into single spaces (Req 11.5).

    Splitting on arbitrary whitespace and re-joining removes line breaks, tabs,
    and runs of spaces, guaranteeing the result is a single line suitable as a
    quoted CSV field.
    """
    if not text:
        return ""
    return " ".join(text.split())


class ReasoningGenerator:
    """Generate a 20–200 character single-line reasoning string per candidate."""

    def generate(
        self,
        candidate: CandidateProfile,
        scored: ScoredCandidate,
        scores: Optional[Dict[str, float]] = None,
    ) -> str:
        """Return the reasoning string for a ranked candidate (Req 11.1–11.5).

        Args:
            candidate: The candidate profile (source of title, YOE, and the
                specific skill/career attribute that is referenced).
            scored: The candidate's component and composite scores; used to
                determine the highest-contributing scoring component.
            scores: Optional raw component-score mapping. Accepted for
                interface flexibility but unused — the authoritative component
                scores live on ``scored``.

        Returns:
            A single-line string of 20–200 characters (inclusive) containing the
            current title, years of experience, the top scoring factor, and at
            least one concrete profile attribute.
        """
        # ``scores`` is accepted for caller flexibility; ScoredCandidate is the
        # authoritative source, so the parameter is intentionally not consulted.
        _ = scores

        title = _sanitize(candidate.profile.current_title) or "Candidate"
        yoe = candidate.profile.years_of_experience
        try:
            yoe_str = f"{float(yoe):.1f}"
        except (TypeError, ValueError):
            yoe_str = "0.0"

        component, _contribution = top_contributing_component(scored)
        label = COMPONENT_LABELS.get(component, component)

        attribute = self._select_attribute(candidate)

        # Truncate the variable-length pieces so the assembled template fits.
        title_part = title[:MAX_TITLE_CHARS].rstrip()
        attribute_part = attribute[:MAX_ATTRIBUTE_CHARS].rstrip()

        reasoning = (
            f"{title_part} with {yoe_str} yrs; "
            f"top factor {label}; {attribute_part}."
        )
        reasoning = _sanitize(reasoning)

        reasoning = self._enforce_length(reasoning, label)
        return reasoning

    # ------------------------------------------------------------------
    # Attribute selection (Req 11.3)
    # ------------------------------------------------------------------
    @staticmethod
    def _select_attribute(candidate: CandidateProfile) -> str:
        """Pick a concrete, real profile attribute to reference (Req 11.3).

        Preference order, choosing only data actually present in the profile so
        nothing is hallucinated:

        1. Up to two of the candidate's skills that map to a required skill
           group (the ones that positively influenced the skill score).
        2. The current company from the profile.
        3. The most recent career-history company, then title.
        4. The first listed skill name (any skill), then the current title.

        Always returns a non-empty, sanitized string.
        """
        # 1. Matched required-skill names, in listed order, de-duplicated.
        matched: list = []
        for skill in candidate.skills:
            name = _sanitize(skill.name)
            if not name:
                continue
            if skill_to_group(skill.name) is not None and name not in matched:
                matched.append(name)
            if len(matched) >= 2:
                break
        if matched:
            return ", ".join(matched)

        # 2. Current company (a real career attribute).
        company = _sanitize(candidate.profile.current_company)
        if company:
            return company

        # 3. Most recent career-history company, then title.
        for entry in candidate.career_history:
            entry_company = _sanitize(entry.company)
            if entry_company:
                return entry_company
        for entry in candidate.career_history:
            entry_title = _sanitize(entry.title)
            if entry_title:
                return entry_title

        # 4. Any listed skill name, then the current title as a final fallback.
        for skill in candidate.skills:
            name = _sanitize(skill.name)
            if name:
                return name

        current_title = _sanitize(candidate.profile.current_title)
        if current_title:
            return current_title

        # Absolute last resort: a truthful generic descriptor.
        return "AI/ML profile"

    # ------------------------------------------------------------------
    # Length enforcement (Req 11.2)
    # ------------------------------------------------------------------
    @staticmethod
    def _enforce_length(reasoning: str, label: str) -> str:
        """Clamp ``reasoning`` to the inclusive ``[MIN_LENGTH, MAX_LENGTH]`` range.

        Over-long strings are truncated to ``MAX_LENGTH`` (with any trailing
        whitespace stripped). Unusually short strings are padded with a truthful
        clause referencing the top scoring factor until they reach
        ``MIN_LENGTH``, without ever exceeding ``MAX_LENGTH``.
        """
        # Truncate if too long (Req 11.2 upper bound).
        if len(reasoning) > MAX_LENGTH:
            reasoning = reasoning[:MAX_LENGTH].rstrip()

        # Pad if too short (Req 11.2 lower bound). A strong-factor clause is
        # truthful because ``label`` names the candidate's top scoring factor.
        if len(reasoning) < MIN_LENGTH:
            pad_clause = f" Strong {label}."
            while len(reasoning) < MIN_LENGTH and len(reasoning) + len(pad_clause) <= MAX_LENGTH:
                reasoning = reasoning + pad_clause
            # If still short (e.g., MAX_LENGTH is tiny), pad with periods.
            while len(reasoning) < MIN_LENGTH:
                reasoning = reasoning + "."

        return reasoning
