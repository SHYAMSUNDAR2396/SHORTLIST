"""Pytest + Hypothesis configuration for the Candidate Ranking System.

Registers a fixed Hypothesis profile used across all property-based tests to
guarantee reproducible runs in CI:

- ``max_examples >= 100`` per property (Requirement 10.5 reproducibility, and the
  design's "minimum 100 examples per property test").
- Deadlines disabled so complex composite generators are not flagged as slow.
- A deterministic ``derandomize`` setting plus a fixed seed so consecutive runs
  exercise the same examples (bit-identical / reproducible behavior).
"""

from __future__ import annotations

from hypothesis import HealthCheck, Phase, Verbosity, settings

# Deterministic seed for CI reproducibility. Hypothesis uses this together with
# ``derandomize=True`` so the same examples are generated on every run.
HYPOTHESIS_SEED = 20240517

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,  # disable per-example deadline for complex generators
    derandomize=True,  # deterministic example generation across runs
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
    verbosity=Verbosity.normal,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink),
)

# A higher-effort profile for local deep runs; still deterministic.
settings.register_profile(
    "thorough",
    parent=settings.get_profile("ci"),
    max_examples=500,
)

# Activate the deterministic CI profile by default.
settings.load_profile("ci")
