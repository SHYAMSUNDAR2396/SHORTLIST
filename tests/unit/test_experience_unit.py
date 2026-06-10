"""Unit tests for the experience-fit boundary values (Task 9.4).

Exercises the piecewise :func:`experience_fit` function at and around its
breakpoints. **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
"""

import pytest

from ranking.scorers.experience import experience_fit


@pytest.mark.parametrize(
    "v, expected",
    [
        (4.0, 0.5),    # lower ramp start -> 0.5
        (5.0, 1.0),    # plateau start -> 1.0
        (9.0, 1.0),    # plateau end -> 1.0
        (11.0, 0.5),   # upper ramp end -> 0.5
        (3.99, 0.2),   # just below 4 -> out-of-range floor
        (11.01, 0.2),  # just above 11 -> out-of-range floor
        (4.5, 0.75),   # midpoint of lower ramp -> 0.75
        (10.0, 0.75),  # midpoint of upper ramp -> 0.75
    ],
)
def test_experience_fit_boundary_values(v, expected):
    assert experience_fit(v) == pytest.approx(expected)
