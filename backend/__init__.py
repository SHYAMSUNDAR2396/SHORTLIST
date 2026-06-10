"""FastAPI backend for the Candidate Ranking System frontend.

This package adapts the deterministic, rule-based ranking pipeline (in the
``ranking`` package and ``rank.py``) into the HTTP API the React/Vite frontend
expects. It runs the pipeline in-process, enriches each top-ranked candidate
with the "AI panel" view the UI renders, and serves it over ``/api/*``.
"""
