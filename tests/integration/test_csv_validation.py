"""CSV validation integration test (task 21.3).

Runs the full pipeline on a small sample and asserts the generated CSV passes
the official ``validate_submission.py`` validator (returns an empty error
list). This covers the end-to-end submission contract (Req 9.1-9.7).

_Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

import rank
from tests.integration._helpers import REPO_ROOT, make_sample_jsonl

SAMPLE_SIZE = 300


def _load_validate_submission():
    """Import ``validate_submission`` from the repo root by file path."""
    module_path = REPO_ROOT / "validate_submission.py"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("validate_submission", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_submission


@pytest.mark.integration
def test_generated_csv_passes_official_validator(tmp_path):
    """The pipeline output validates with zero errors against the official rules."""
    sample = make_sample_jsonl(tmp_path / "sample.jsonl", n=SAMPLE_SIZE)
    # Use a participant-id-style filename so the validator's filename checks pass.
    out_path = tmp_path / "team_001.csv"

    assert rank.run_pipeline(str(sample), str(out_path)) == 0
    assert out_path.exists()

    validate_submission = _load_validate_submission()
    errors = validate_submission(str(out_path))

    assert errors == [], f"Validator reported errors: {errors}"
