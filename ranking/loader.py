"""Streaming JSONL data loader for the Candidate Ranking System.

The :class:`DataLoader` reads a ``candidates.jsonl`` file one line at a time
(never loading the whole file into memory), parses each line as JSON, validates
the presence of the required top-level fields, and maps each valid record into
the dataclasses defined in :mod:`ranking.models`.

Design references (see design.md "Component: DataLoader" and "Error Handling"):

- Streaming reads keep RAM usage linear with record size, not dataset size, so
  100K records can be processed in <60s using <8 GB RAM (Req 1.1).
- Malformed JSON lines are skipped with a 1-based line-number warning to stderr
  and never terminate processing (Req 1.2).
- Records missing any required field are skipped, counted, and reported by
  candidate_id (or line number when candidate_id is absent) to stderr (Req 1.3).
- Parsed/skipped totals are written to stdout when loading completes (Req 1.4).
- A missing or unreadable file path causes a non-zero process exit with a
  stderr error message (Req 1.5).

Defensive parsing rules (design "Error Handling"):

- Dates are parsed from ISO "YYYY-MM-DD" strings; unparseable/absent dates
  become ``None``.
- ``duration_months`` that is negative or absent is treated as ``0``.
- ``certifications`` and ``languages`` are optional arrays and default to an
  empty list when absent.
"""

import json
import sys
from datetime import date, datetime
from typing import Any, Dict, List, Optional, TextIO, Tuple

from ranking.models import (
    CandidateProfile,
    CareerEntry,
    CertificationEntry,
    EducationEntry,
    LanguageEntry,
    LoadStats,
    ProfileData,
    RedrobSignals,
    SkillEntry,
)

# Top-level fields a record must contain to be considered valid (Req 1.3).
REQUIRED_FIELDS: Tuple[str, ...] = (
    "candidate_id",
    "profile",
    "career_history",
    "education",
    "skills",
    "redrob_signals",
)


class DataLoader:
    """Loads and validates candidate profiles from a JSONL file."""

    def load(self, filepath: str) -> Tuple[List[CandidateProfile], LoadStats]:
        """Parse ``filepath`` and return valid candidates plus load statistics.

        The file is read line-by-line. Each non-empty line is parsed as a
        single JSON object. Lines that fail JSON parsing or that are missing a
        required field are skipped (with a stderr warning) and counted; valid
        records are mapped to :class:`CandidateProfile` objects.

        On completion the parsed/skipped totals are written to stdout.

        Args:
            filepath: Path to the JSONL candidates file.

        Returns:
            A tuple ``(candidates, stats)`` where ``candidates`` is the list of
            successfully parsed :class:`CandidateProfile` objects and ``stats``
            is a :class:`LoadStats` with ``total_parsed`` = valid count and
            ``total_skipped`` = ``json_errors + validation_errors``.

        Raises:
            SystemExit: with a non-zero code if the file does not exist or is
                not readable (Req 1.5).
        """
        candidates: List[CandidateProfile] = []
        stats = LoadStats()

        handle = self._open_file(filepath)
        try:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    # Blank lines are not records; skip silently.
                    continue

                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    stats.json_errors += 1
                    print(
                        f"[loader] Malformed JSON on line {line_number}; skipping record.",
                        file=sys.stderr,
                    )
                    continue

                if not isinstance(record, dict) or not self._has_required_fields(record):
                    stats.validation_errors += 1
                    identifier = self._record_identifier(record, line_number)
                    print(
                        f"[loader] Record {identifier} missing required field(s); skipping record.",
                        file=sys.stderr,
                    )
                    continue

                try:
                    candidate = self._build_candidate(record)
                except (KeyError, TypeError, ValueError) as exc:
                    # A structurally-present-but-malformed record (e.g. profile
                    # is not an object) is treated as a validation failure
                    # rather than crashing the whole load (Req 1.3).
                    stats.validation_errors += 1
                    identifier = self._record_identifier(record, line_number)
                    print(
                        f"[loader] Record {identifier} could not be parsed ({exc}); skipping record.",
                        file=sys.stderr,
                    )
                    continue

                candidates.append(candidate)
        finally:
            handle.close()

        stats.total_parsed = len(candidates)
        stats.total_skipped = stats.json_errors + stats.validation_errors

        # Write parsed/skipped totals to stdout (Req 1.4).
        print(
            f"[loader] Parsed {stats.total_parsed} candidate(s); "
            f"skipped {stats.total_skipped} record(s) "
            f"({stats.json_errors} JSON error(s), "
            f"{stats.validation_errors} validation error(s)).",
            file=sys.stdout,
        )

        return candidates, stats

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    @staticmethod
    def _open_file(filepath: str) -> TextIO:
        """Open ``filepath`` for streaming reads or exit non-zero (Req 1.5)."""
        try:
            return open(filepath, "r", encoding="utf-8")
        except OSError as exc:
            print(
                f"[loader] Unable to open candidates file '{filepath}': {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _has_required_fields(record: Dict[str, Any]) -> bool:
        """Return ``True`` when every required top-level field is present."""
        return all(field in record and record[field] is not None for field in REQUIRED_FIELDS)

    @staticmethod
    def _record_identifier(record: Any, line_number: int) -> str:
        """Best-effort identifier for an invalid record (id or line number)."""
        if isinstance(record, dict):
            candidate_id = record.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id:
                return f"'{candidate_id}'"
        return f"on line {line_number}"

    # ------------------------------------------------------------------
    # Record mapping
    # ------------------------------------------------------------------
    def _build_candidate(self, record: Dict[str, Any]) -> CandidateProfile:
        """Map a validated JSON record into a :class:`CandidateProfile`."""
        return CandidateProfile(
            candidate_id=str(record["candidate_id"]),
            profile=self._build_profile(record["profile"]),
            career_history=[self._build_career_entry(e) for e in record["career_history"]],
            education=[self._build_education_entry(e) for e in record["education"]],
            skills=[self._build_skill_entry(e) for e in record["skills"]],
            certifications=[
                self._build_certification_entry(e) for e in record.get("certifications") or []
            ],
            languages=[self._build_language_entry(e) for e in record.get("languages") or []],
            redrob_signals=self._build_redrob_signals(record["redrob_signals"]),
        )

    @staticmethod
    def _build_profile(data: Dict[str, Any]) -> ProfileData:
        return ProfileData(
            anonymized_name=_as_str(data.get("anonymized_name")),
            headline=_as_str(data.get("headline")),
            summary=_as_str(data.get("summary")),
            location=_as_str(data.get("location")),
            country=_as_str(data.get("country")),
            years_of_experience=_as_float(data.get("years_of_experience"), 0.0),
            current_title=_as_str(data.get("current_title")),
            current_company=_as_str(data.get("current_company")),
            current_company_size=_as_str(data.get("current_company_size")),
            current_industry=_as_str(data.get("current_industry")),
        )

    def _build_career_entry(self, data: Dict[str, Any]) -> CareerEntry:
        return CareerEntry(
            company=_as_str(data.get("company")),
            title=_as_str(data.get("title")),
            start_date=_parse_date(data.get("start_date")),
            end_date=_parse_date(data.get("end_date")),
            duration_months=_as_non_negative_int(data.get("duration_months")),
            is_current=bool(data.get("is_current", False)),
            industry=_as_str(data.get("industry")),
            company_size=_as_str(data.get("company_size")),
            description=_as_str(data.get("description")),
        )

    @staticmethod
    def _build_education_entry(data: Dict[str, Any]) -> EducationEntry:
        grade = data.get("grade")
        return EducationEntry(
            institution=_as_str(data.get("institution")),
            degree=_as_str(data.get("degree")),
            field_of_study=_as_str(data.get("field_of_study")),
            start_year=_as_non_negative_int(data.get("start_year")),
            end_year=_as_non_negative_int(data.get("end_year")),
            grade=str(grade) if grade is not None else None,
            tier=_as_str(data.get("tier")) or "unknown",
        )

    @staticmethod
    def _build_skill_entry(data: Dict[str, Any]) -> SkillEntry:
        return SkillEntry(
            name=_as_str(data.get("name")),
            proficiency=_as_str(data.get("proficiency")),
            endorsements=_as_non_negative_int(data.get("endorsements")),
            duration_months=_as_non_negative_int(data.get("duration_months")),
        )

    @staticmethod
    def _build_certification_entry(data: Dict[str, Any]) -> CertificationEntry:
        return CertificationEntry(
            name=_as_str(data.get("name")),
            issuer=_as_str(data.get("issuer")),
            year=_as_non_negative_int(data.get("year")),
        )

    @staticmethod
    def _build_language_entry(data: Dict[str, Any]) -> LanguageEntry:
        return LanguageEntry(
            language=_as_str(data.get("language")),
            proficiency=_as_str(data.get("proficiency")),
        )

    @staticmethod
    def _build_redrob_signals(data: Dict[str, Any]) -> RedrobSignals:
        return RedrobSignals(
            profile_completeness_score=_as_float(data.get("profile_completeness_score"), 0.0),
            signup_date=_parse_date(data.get("signup_date")),
            last_active_date=_parse_date(data.get("last_active_date")),
            open_to_work_flag=bool(data.get("open_to_work_flag", False)),
            profile_views_received_30d=_as_non_negative_int(data.get("profile_views_received_30d")),
            applications_submitted_30d=_as_non_negative_int(data.get("applications_submitted_30d")),
            recruiter_response_rate=_as_float(data.get("recruiter_response_rate"), 0.0),
            avg_response_time_hours=_as_float(data.get("avg_response_time_hours"), 0.0),
            skill_assessment_scores=_as_float_dict(data.get("skill_assessment_scores")),
            connection_count=_as_non_negative_int(data.get("connection_count")),
            endorsements_received=_as_non_negative_int(data.get("endorsements_received")),
            notice_period_days=_as_non_negative_int(data.get("notice_period_days")),
            expected_salary_range_inr_lpa=_as_float_dict(data.get("expected_salary_range_inr_lpa")),
            preferred_work_mode=_as_str(data.get("preferred_work_mode")),
            willing_to_relocate=bool(data.get("willing_to_relocate", False)),
            github_activity_score=_as_float(data.get("github_activity_score"), 0.0),
            search_appearance_30d=_as_non_negative_int(data.get("search_appearance_30d")),
            saved_by_recruiters_30d=_as_non_negative_int(data.get("saved_by_recruiters_30d")),
            interview_completion_rate=_as_float(data.get("interview_completion_rate"), 0.0),
            offer_acceptance_rate=_as_float(data.get("offer_acceptance_rate"), 0.0),
            verified_email=bool(data.get("verified_email", False)),
            verified_phone=bool(data.get("verified_phone", False)),
            linkedin_connected=bool(data.get("linkedin_connected", False)),
        )


# ---------------------------------------------------------------------------
# Defensive coercion helpers
# ---------------------------------------------------------------------------
def _parse_date(value: Any) -> Optional[date]:
    """Parse an ISO ``YYYY-MM-DD`` date; return ``None`` when unparseable.

    Treats ``None``, empty strings, and any malformed value as ``None`` so the
    pipeline can continue scoring with a graceful fallback (design Error
    Handling: "Treat that date as None, continue scoring").
    """
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        # Fall back to fromisoformat for tolerant ISO variants; None on failure.
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _as_non_negative_int(value: Any) -> int:
    """Coerce ``value`` to a non-negative int; negative/absent/invalid -> 0.

    Used for ``duration_months`` and other count fields (design Error
    Handling: "Negative or absent duration_months -> Treat as 0").
    """
    if value is None or isinstance(value, bool):
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return result if result >= 0 else 0


def _as_float(value: Any, default: float) -> float:
    """Coerce ``value`` to a float, returning ``default`` on failure.

    Note: ``-1`` sentinels (e.g. github_activity_score, offer_acceptance_rate)
    are preserved as-is because they are meaningful to downstream scorers.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any) -> str:
    """Coerce ``value`` to a string; ``None`` becomes an empty string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _as_float_dict(value: Any) -> Dict[str, float]:
    """Coerce a mapping of name -> number into ``Dict[str, float]``.

    Non-dict inputs and non-numeric values are dropped rather than raising, so
    a malformed nested object does not invalidate an otherwise-valid record.
    """
    if not isinstance(value, dict):
        return {}
    result: Dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, bool):
            continue
        try:
            result[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return result
