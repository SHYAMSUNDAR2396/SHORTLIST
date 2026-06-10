"""Shared helpers for the integration / end-to-end tests.

These tests exercise the full pipeline (``rank.run_pipeline``) on a *small*
subset of the real ``candidates.jsonl`` so they stay fast (the full file is
100K lines). :func:`make_sample_jsonl` copies the first ``n`` non-empty lines
from the real dataset into a destination path (typically under pytest's
``tmp_path``). When the real dataset cannot be located the caller should skip
the test gracefully via :func:`require_sample_jsonl`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

# Repo root = two levels up from this file: tests/integration/_helpers.py ->
# tests/ -> <repo root>.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The real (large) dataset lives at the repo root.
REAL_CANDIDATES = REPO_ROOT / "candidates.jsonl"


def find_real_candidates() -> Optional[Path]:
    """Return the path to the real ``candidates.jsonl`` or ``None`` if absent."""
    return REAL_CANDIDATES if REAL_CANDIDATES.is_file() else None


def make_sample_jsonl(dest_path, n: int = 300) -> Path:
    """Write the first ``n`` non-empty lines of the real dataset to ``dest_path``.

    Args:
        dest_path: Destination path for the sample JSONL (str or ``Path``).
        n: Number of non-empty candidate lines to copy.

    Returns:
        The destination :class:`~pathlib.Path` that was written.

    Raises:
        pytest.skip.Exception: via :func:`pytest.skip` when the real dataset is
            missing, so dependent tests skip gracefully rather than error.
    """
    source = find_real_candidates()
    if source is None:
        pytest.skip(
            f"Real dataset not found at {REAL_CANDIDATES}; skipping integration test."
        )

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(source, "r", encoding="utf-8") as src, open(
        dest, "w", encoding="utf-8"
    ) as out:
        for raw_line in src:
            if written >= n:
                break
            if not raw_line.strip():
                # Skip blank lines so we copy exactly ``n`` real records.
                continue
            # Normalise to a single trailing newline per record.
            out.write(raw_line.rstrip("\n") + "\n")
            written += 1

    if written == 0:
        pytest.skip("Real dataset contained no usable candidate lines.")

    return dest
