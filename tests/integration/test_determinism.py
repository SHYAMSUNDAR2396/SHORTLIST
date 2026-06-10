"""Determinism integration test (task 21.1).

The challenge requires bit-identical output across runs (Req 10.5). Running the
pipeline twice on the same input must produce byte-for-byte identical CSV files.

_Requirements: 10.5_
"""

from __future__ import annotations

import pytest

import rank
from tests.integration._helpers import make_sample_jsonl

SAMPLE_SIZE = 300


@pytest.mark.integration
def test_pipeline_output_is_byte_identical_across_runs(tmp_path):
    """Two runs on the same input yield byte-identical output files."""
    sample = make_sample_jsonl(tmp_path / "sample.jsonl", n=SAMPLE_SIZE)
    out_a = tmp_path / "submission_a.csv"
    out_b = tmp_path / "submission_b.csv"

    assert rank.run_pipeline(str(sample), str(out_a)) == 0
    assert rank.run_pipeline(str(sample), str(out_b)) == 0

    bytes_a = out_a.read_bytes()
    bytes_b = out_b.read_bytes()

    assert bytes_a == bytes_b
