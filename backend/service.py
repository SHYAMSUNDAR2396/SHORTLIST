"""Pipeline service: runs the ranker in-process and enriches candidates.

The frontend renders an "AI panel" view of each candidate (multiple reviewer
perspectives, a verdict, strengths/concerns, a fairness/bias audit, trajectory
signals). Our ranking system is a single deterministic rule-based scorer, so
this module *adapts* the real component scores it produces into that richer
shape. Every derived field is a deterministic, explainable function of the
candidate's actual component scores — no randomness, no fabricated data.

Design notes:
- The pipeline is executed in-process via the same components used by
  ``rank.py`` (load -> honeypot -> score -> disqualify -> rank), so the API and
  the CLI produce identical rankings.
- Results are cached in memory after the first run; the frontend's "run
  pipeline" action recomputes and refreshes the cache.
- Component scores are in [0, 1]; the UI shows scores on a 0-10 scale, so we
  multiply by 10 at the adapter boundary.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from ranking.composite import CompositeScorer
from ranking.disqualifier import DisqualifierFilter
from ranking.honeypot import HoneypotDetector
from ranking.loader import DataLoader
from ranking.models import CandidateProfile, ScoredCandidate
from ranking.ranker import select_top, to_ranked
from ranking.reasoning import ReasoningGenerator
from ranking.scorers.behavioral import BehavioralSignalEvaluator
from ranking.scorers.career import CareerAnalyzer
from ranking.scorers.education import EducationScorer
from ranking.scorers.experience import ExperienceScorer
from ranking.scorers.location import LocationWorkModeScorer
from ranking.scorers.skill import SkillScorer
from ranking.constants import skill_to_group

import rank as pipeline

# Repository root (this file lives in backend/).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = REPO_ROOT / "candidates.jsonl"
SAMPLE_CANDIDATES = REPO_ROOT / "sample_candidates.json"
UPLOAD_DIR = REPO_ROOT / "uploads"

TOP_N = 100
# Counterfactual delta above which a candidate is flagged for potential bias.
BIAS_FLAG_THRESHOLD = 0.15


@dataclass
class EnrichedCandidate:
    """The UI-facing candidate view derived from real component scores."""

    rank: int
    candidate_id: str
    name: str
    composite_score: float          # 0-10
    trajectory_score: float         # 0-10
    hiring_manager_score: float     # 0-10
    peer_interviewer_score: float   # 0-10
    devils_advocate_score: float    # 0-10
    panel_variance: float
    requires_human_review: bool
    verdict_consensus: str
    strengths: List[str]
    concerns: List[str]
    narrative: str
    bias_flag: bool
    counterfactual_delta: Optional[float]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "name": self.name,
            "composite_score": round(self.composite_score, 2),
            "trajectory_score": round(self.trajectory_score, 2),
            "hiring_manager_score": round(self.hiring_manager_score, 2),
            "peer_interviewer_score": round(self.peer_interviewer_score, 2),
            "devils_advocate_score": round(self.devils_advocate_score, 2),
            "panel_variance": round(self.panel_variance, 2),
            "requires_human_review": self.requires_human_review,
            "verdict_consensus": self.verdict_consensus,
            "strengths": self.strengths,
            "concerns": self.concerns,
            "narrative": self.narrative,
            "bias_flag": self.bias_flag,
            "counterfactual_delta": (
                round(self.counterfactual_delta, 4)
                if self.counterfactual_delta is not None
                else None
            ),
        }


@dataclass
class PipelineState:
    """In-memory state shared across requests."""

    candidates: List[EnrichedCandidate] = field(default_factory=list)
    by_id: Dict[str, EnrichedCandidate] = field(default_factory=dict)
    eval_date: Optional[date] = None
    source_path: Optional[str] = None
    running: bool = False
    last_result: Optional[dict] = None
    has_run: bool = False


# ---------------------------------------------------------------------------
# Verdict / score-band helpers (deterministic).
# ---------------------------------------------------------------------------
def _verdict(composite_0_10: float) -> str:
    if composite_0_10 >= 7.5:
        return "strong_yes"
    if composite_0_10 >= 6.0:
        return "yes"
    if composite_0_10 >= 4.0:
        return "maybe"
    return "no"


def _clamp10(x: float) -> float:
    return max(0.0, min(10.0, x))


class PipelineService:
    """Runs the ranking pipeline and serves enriched results to the frontend."""

    def __init__(self) -> None:
        self._state = PipelineState()
        self._lock = threading.Lock()

        # Reusable scorer instances (stateless, pure).
        self._skill = SkillScorer()
        self._career = CareerAnalyzer()
        self._experience = ExperienceScorer()
        self._behavioral = BehavioralSignalEvaluator()
        self._education = EducationScorer()
        self._location = LocationWorkModeScorer()
        self._composite = CompositeScorer()
        self._disqualifier = DisqualifierFilter()
        self._reasoning = ReasoningGenerator()
        self._honeypot = HoneypotDetector()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    @property
    def has_run(self) -> bool:
        return self._state.has_run

    @property
    def running(self) -> bool:
        return self._state.running

    @property
    def last_result(self) -> Optional[dict]:
        return self._state.last_result

    def candidates(self) -> List[EnrichedCandidate]:
        return self._state.candidates

    def candidate(self, candidate_id: str) -> Optional[EnrichedCandidate]:
        return self._state.by_id.get(candidate_id)

    def output_csv_path(self) -> Path:
        return REPO_ROOT / "submission.csv"

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def resolve_source(self) -> Path:
        """Pick the candidate source: full dataset if present, else sample."""
        if DEFAULT_CANDIDATES.exists():
            return DEFAULT_CANDIDATES
        if SAMPLE_CANDIDATES.exists():
            return SAMPLE_CANDIDATES
        # Check uploads/ for any .jsonl or .json file
        for f in sorted(UPLOAD_DIR.glob("*.jsonl")) + sorted(UPLOAD_DIR.glob("*.json")):
            return f
        raise FileNotFoundError(
            "No candidate data found. Upload a candidates.jsonl or "
            "sample_candidates.json file via the pipeline setup page."
        )

    def run(self, candidates_path: Optional[str] = None) -> dict:
        """Run the full pipeline and cache enriched results.

        Returns a result dict (returncode + tails) mirroring what the frontend's
        PipelineStatus expects. Thread-safe: only one run proceeds at a time.
        """
        with self._lock:
            self._state.running = True

        try:
            source = Path(candidates_path) if candidates_path else self.resolve_source()
            if not source.exists():
                raise FileNotFoundError(f"Candidate source not found: {source}")

            # Load (handles .jsonl; sample is a JSON array, so branch on suffix).
            candidates = self._load_candidates(source)
            eval_date = pipeline.determine_eval_date(candidates)

            clean, _flagged = self._honeypot.detect(candidates, eval_date)
            scored = self._score_all(clean, eval_date)
            ordered = select_top(scored, TOP_N)

            profiles_by_id = {c.candidate_id: c for c in candidates}
            enriched = self._enrich(ordered, profiles_by_id, eval_date)

            # Also write the official submission.csv so the export endpoint and
            # the CLI stay in sync.
            ranked = to_ranked(
                ordered,
                reasoning_fn=lambda _r, sc: self._reasoning.generate(
                    profiles_by_id[sc.candidate_id], sc
                ),
            )
            from ranking.formatter import SubmissionFormatter

            SubmissionFormatter().write(
                ranked,
                str(self.output_csv_path()),
                valid_candidate_ids=set(profiles_by_id.keys()),
            )

            with self._lock:
                self._state.candidates = enriched
                self._state.by_id = {c.candidate_id: c for c in enriched}
                self._state.eval_date = eval_date
                self._state.source_path = str(source)
                self._state.has_run = True
                self._state.last_result = {
                    "returncode": 0,
                    "stdout_tail": (
                        f"Ranked {len(enriched)} candidates from "
                        f"{source.name} (eval_date={eval_date.isoformat()})."
                    ),
                }
            return self._state.last_result
        except Exception as exc:  # surface failures to the frontend
            with self._lock:
                self._state.last_result = {"returncode": 1, "error": str(exc)}
            return self._state.last_result
        finally:
            with self._lock:
                self._state.running = False

    def ensure_loaded(self) -> None:
        """Run the pipeline once on first access if it hasn't run yet."""
        if not self._state.has_run and not self._state.running:
            try:
                self.run()
            except Exception:
                # No data available yet — return empty state, don't crash.
                pass

    # ------------------------------------------------------------------
    # Internal: loading + scoring
    # ------------------------------------------------------------------
    def _load_candidates(self, source: Path) -> List[CandidateProfile]:
        """Load candidates from a .jsonl file or a .json array of profiles."""
        if source.suffix.lower() == ".jsonl":
            candidates, _stats = DataLoader().load(str(source))
            return candidates

        # sample_candidates.json is a JSON array; reuse the loader's row mapping
        # by streaming each element as a JSONL line into a temp-free path.
        import json
        import tempfile
        import os

        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
        fd, tmp = tempfile.mkstemp(suffix=".jsonl")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                for row in data:
                    out.write(json.dumps(row) + "\n")
            candidates, _stats = DataLoader().load(tmp)
        finally:
            os.remove(tmp)
        return candidates

    def _score_all(
        self, candidates: List[CandidateProfile], eval_date: date
    ) -> List[ScoredCandidate]:
        scored: List[ScoredCandidate] = []
        for c in candidates:
            skill = self._skill.score(c)
            career = self._career.score(c)
            experience = self._experience.score(c)
            behavioral = self._behavioral.score(c, eval_date)
            education = self._education.score(c)
            location = self._location.score(c)

            components = {
                "skill": skill,
                "career": career,
                "experience": experience,
                "behavioral": behavioral,
                "education": education,
                "location_work_mode": location,
            }
            composite = self._composite.compute(components, penalty_multiplier=1.0)
            penalized, reason = self._disqualifier.apply(c, composite, eval_date)
            penalty = (penalized / composite) if (reason and composite > 0) else 1.0

            scored.append(
                ScoredCandidate(
                    candidate_id=c.candidate_id,
                    skill_score=skill,
                    career_score=career,
                    experience_score=experience,
                    behavioral_score=behavioral,
                    education_score=education,
                    location_work_mode_score=location,
                    composite_score=composite,
                    penalty_multiplier=penalty,
                    disqualification_reason=reason,
                    final_score=round(penalized, 4),
                )
            )
        return scored

    # ------------------------------------------------------------------
    # Internal: enrichment into the UI "AI panel" shape
    # ------------------------------------------------------------------
    def _enrich(
        self,
        ordered: List[ScoredCandidate],
        profiles_by_id: Dict[str, CandidateProfile],
        eval_date: date,
    ) -> List[EnrichedCandidate]:
        enriched: List[EnrichedCandidate] = []
        for i, sc in enumerate(ordered):
            profile = profiles_by_id[sc.candidate_id]
            enriched.append(self._enrich_one(i + 1, sc, profile, eval_date))
        return enriched

    def _enrich_one(
        self,
        rank: int,
        sc: ScoredCandidate,
        profile: CandidateProfile,
        eval_date: date,
    ) -> EnrichedCandidate:
        # Map the three "reviewer perspectives" to real, distinct signals:
        # - Hiring Manager weighs skill + career fit (can they do the job).
        # - Peer Interviewer weighs career + experience (collaboration/depth).
        # - Devil's Advocate is the skeptical view: penalizes any disqualifier
        #   and weak behavioral engagement.
        composite_10 = _clamp10(sc.final_score * 10)

        hiring_manager = _clamp10((0.6 * sc.skill_score + 0.4 * sc.career_score) * 10)
        peer = _clamp10(
            (0.5 * sc.career_score + 0.5 * sc.experience_score) * 10
        )
        devils = _clamp10(
            (0.5 * sc.skill_score + 0.3 * sc.behavioral_score + 0.2 * sc.education_score)
            * 10
            * (sc.penalty_multiplier if sc.disqualification_reason else 1.0)
        )
        trajectory = _clamp10(
            (0.5 * sc.career_score + 0.3 * sc.experience_score + 0.2 * sc.skill_score)
            * 10
        )

        panel = [hiring_manager, peer, devils]
        mean = sum(panel) / len(panel)
        variance = sum((p - mean) ** 2 for p in panel) / len(panel)

        strengths = self._strengths(sc, profile)
        concerns = self._concerns(sc, profile)

        narrative = self._reasoning.generate(profile, sc)

        # Counterfactual fairness proxy: re-score with the candidate's name and
        # institution influence neutralized. Our scorer ignores name/gender
        # entirely (it never reads anonymized_name), so the delta is driven only
        # by education tier — a transparent, real sensitivity measure rather than
        # a fabricated number.
        cf_delta = self._counterfactual_delta(sc, profile, eval_date)
        bias_flag = cf_delta is not None and cf_delta > BIAS_FLAG_THRESHOLD

        requires_review = bias_flag or variance > 2.0 or composite_10 < 4.0

        return EnrichedCandidate(
            rank=rank,
            candidate_id=sc.candidate_id,
            name=profile.profile.anonymized_name or sc.candidate_id,
            composite_score=composite_10,
            trajectory_score=trajectory,
            hiring_manager_score=hiring_manager,
            peer_interviewer_score=peer,
            devils_advocate_score=devils,
            panel_variance=variance,
            requires_human_review=requires_review,
            verdict_consensus=_verdict(composite_10),
            strengths=strengths,
            concerns=concerns,
            narrative=narrative,
            bias_flag=bias_flag,
            counterfactual_delta=cf_delta,
        )

    def _strengths(self, sc: ScoredCandidate, profile: CandidateProfile) -> List[str]:
        out: List[str] = []
        if sc.skill_score >= 0.5:
            groups = sorted(
                {
                    skill_to_group(s.name)
                    for s in profile.skills
                    if skill_to_group(s.name) is not None
                }
            )
            label = ", ".join(g.replace("_", " ") for g in groups[:3])
            if label:
                out.append(f"Strong skill match ({label})")
        if sc.career_score >= 0.5:
            out.append(f"Relevant career fit — {profile.profile.current_title}")
        if sc.experience_score >= 0.8:
            out.append(
                f"Experience in target band ({profile.profile.years_of_experience:.1f} yrs)"
            )
        if sc.behavioral_score >= 0.7:
            out.append("Active, responsive on platform")
        if sc.education_score >= 0.7:
            out.append("Strong educational background")
        if not out:
            out.append(f"Top contributing factor: {self._top_factor(sc)}")
        return out

    def _concerns(self, sc: ScoredCandidate, profile: CandidateProfile) -> List[str]:
        out: List[str] = []
        if sc.disqualification_reason:
            out.append(
                f"Disqualifier triggered: {sc.disqualification_reason.replace('_', ' ')}"
            )
        if sc.skill_score < 0.4:
            out.append("Limited match against required AI/ML skills")
        if sc.experience_score < 0.5:
            out.append(
                f"Experience outside 5-9 yr target ({profile.profile.years_of_experience:.1f} yrs)"
            )
        if sc.behavioral_score < 0.5:
            out.append("Low recent platform engagement")
        if sc.career_score < 0.4:
            out.append("Career history weakly aligned with the role")
        return out

    def _top_factor(self, sc: ScoredCandidate) -> str:
        from ranking.reasoning import top_contributing_component, COMPONENT_LABELS

        component, _ = top_contributing_component(sc)
        return COMPONENT_LABELS.get(component, component)

    def _counterfactual_delta(
        self, sc: ScoredCandidate, profile: CandidateProfile, eval_date: date
    ) -> Optional[float]:
        """Measure score sensitivity to a fairness-relevant perturbation.

        We neutralize the education tier to "unknown" and re-score, returning the
        absolute change in the final score. Because the ranker never reads the
        candidate's name or gender, name/pronoun swaps produce a zero delta by
        construction; the institution-tier swap is the meaningful test.
        """
        if not profile.education:
            return 0.0

        import copy

        perturbed = copy.deepcopy(profile)
        for entry in perturbed.education:
            entry.tier = "unknown"

        components = {
            "skill": self._skill.score(perturbed),
            "career": self._career.score(perturbed),
            "experience": self._experience.score(perturbed),
            "behavioral": self._behavioral.score(perturbed, eval_date),
            "education": self._education.score(perturbed),
            "location_work_mode": self._location.score(perturbed),
        }
        composite = self._composite.compute(components, penalty_multiplier=1.0)
        penalized, _reason = self._disqualifier.apply(perturbed, composite, eval_date)
        return abs(sc.final_score - round(penalized, 4))


# Module-level singleton used by the FastAPI app.
service = PipelineService()
