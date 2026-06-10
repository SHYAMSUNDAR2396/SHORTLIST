"""No-network / no-LLM integration test (task 21.2).

The pipeline must be fully offline and CPU-only (Req 10.2, 10.3). We install a
socket guard that raises if any socket is created or any outbound connection is
attempted, then run the full pipeline on a small sample and assert it completes
without touching the network. ``socket`` underlies ``urllib``/``http`` so
guarding it covers higher-level HTTP clients too. Local file I/O does not use
sockets, so a correct offline pipeline passes cleanly.

_Requirements: 10.2, 10.3_
"""

from __future__ import annotations

import socket

import pytest

import rank
from tests.integration._helpers import make_sample_jsonl

SAMPLE_SIZE = 300


@pytest.mark.integration
def test_pipeline_makes_no_network_calls(tmp_path, monkeypatch):
    """run_pipeline completes under a socket guard with no connection attempts."""
    sample = make_sample_jsonl(tmp_path / "sample.jsonl", n=SAMPLE_SIZE)
    out_path = tmp_path / "submission.csv"

    def _blocked_socket(*args, **kwargs):
        raise AssertionError("Network access attempted: socket.socket() called")

    def _blocked_create_connection(*args, **kwargs):
        raise AssertionError(
            "Network access attempted: socket.create_connection() called"
        )

    # monkeypatch automatically restores the originals after the test.
    monkeypatch.setattr(socket, "socket", _blocked_socket)
    monkeypatch.setattr(socket, "create_connection", _blocked_create_connection)

    # Should complete without raising the guard's AssertionError.
    exit_code = rank.run_pipeline(str(sample), str(out_path))

    assert exit_code == 0
    assert out_path.exists()
