"""Regressions for shape/iteration/length validation in numerical primitives."""

from __future__ import annotations

import pytest

from cds.diffeq.solvers import solve_system
from cds.math_utils.linalg import determinant
from cds.optimization.minimize import line_search, newton_method


def test_determinant_rejects_rectangular_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        determinant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_determinant_rejects_ragged_matrix() -> None:
    with pytest.raises(ValueError, match="ragged"):
        determinant([[1.0, 2.0], [3.0]])


def test_newton_reports_actual_iterations_when_derivative_vanishes() -> None:
    result = newton_method(lambda x: 1.0, 0.0, max_iter=500)
    assert result.converged is False
    assert result.iterations == 0


def test_line_search_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="a <= b"):
        line_search(lambda x: x * x, 1.0, 0.0)


def test_solve_system_rejects_short_derivative() -> None:
    with pytest.raises(ValueError, match="state of length 2"):
        solve_system(lambda t, y: [1.0], 0.0, [0.0, 0.0], 1.0, dt=0.5)


def test_solve_system_rejects_long_derivative() -> None:
    with pytest.raises(ValueError, match="state of length 1"):
        solve_system(lambda t, y: [1.0, 2.0], 0.0, [0.0], 1.0, dt=0.5)
