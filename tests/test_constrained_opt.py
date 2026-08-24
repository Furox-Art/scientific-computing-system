"""Tests for :mod:`cds.optimization.constrained` (projected GD + penalty)."""

from __future__ import annotations

import pytest

from cds.optimization.constrained import (
    ProjectedGradientResult,
    _penalized_objective,
    projected_gradient_descent,
    quadratic_penalty,
)

# --------------------------------------------------------------------- #
# Projected gradient descent                                             #
# --------------------------------------------------------------------- #


def test_projected_gradient_descent_clamps_at_active_upper_bound() -> None:
    def f(v: list[float]) -> float:
        return (v[0] - 3.0) ** 2

    res = projected_gradient_descent(f, [0.0], [-10.0], [1.0])
    assert isinstance(res, ProjectedGradientResult)
    assert res.converged
    assert res.x[0] == pytest.approx(1.0, abs=1e-6)
    assert res.fun == pytest.approx(4.0, abs=1e-6)
    assert res.iterations < 500


def test_projected_gradient_descent_mixed_active_and_interior_coordinates() -> None:
    # Coordinate 0 is clamped at its lower bound (the gradient pulls below
    # it), while coordinate 1 converges to an interior optimum.
    def f(v: list[float]) -> float:
        return (v[0] + 3.0) ** 2 + (v[1] + 2.0) ** 2

    res = projected_gradient_descent(f, [0.0, 0.0], [0.0, -10.0], [1.0, 10.0])
    assert res.converged
    assert res.x[0] == pytest.approx(0.0, abs=1e-8)
    assert res.x[1] == pytest.approx(-2.0, abs=1e-6)


def test_projected_gradient_descent_interior_matches_unconstrained_optimum() -> None:
    def rosenbrock_lite(v: list[float]) -> float:
        x, y = v
        return (1.0 - x) ** 2 + 5.0 * (y - x * x) ** 2

    res = projected_gradient_descent(
        rosenbrock_lite, [-1.2, 1.0], [-5.0, -5.0], [5.0, 5.0], lr=0.02, max_iter=20000
    )
    assert res.converged
    assert res.fun == pytest.approx(0.0, abs=1e-4)
    assert res.x[0] == pytest.approx(1.0, abs=1e-3)
    assert res.x[1] == pytest.approx(1.0, abs=1e-3)


def test_projected_gradient_descent_reports_max_iter_when_not_converged() -> None:
    res = projected_gradient_descent(
        lambda v: v[0] ** 2, [5.0], [-100.0], [100.0], lr=0.001, max_iter=3
    )
    assert not res.converged
    assert res.iterations == 3


def test_projected_gradient_descent_projects_starting_point_into_box() -> None:
    res = projected_gradient_descent(lambda v: v[0] ** 2, [7.0], [0.0], [1.0])
    assert res.converged
    assert res.x[0] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= res.x[0] <= 1.0


def test_projected_gradient_descent_validation() -> None:
    def obj(v: list[float]) -> float:
        return 0.0

    with pytest.raises(ValueError, match="x0 must be non-empty"):
        projected_gradient_descent(obj, [], [], [])
    with pytest.raises(ValueError, match="lower must have the same length as x0"):
        projected_gradient_descent(obj, [1.0], [], [1.0])
    with pytest.raises(ValueError, match="upper must have the same length as x0"):
        projected_gradient_descent(obj, [1.0], [0.0], [])
    with pytest.raises(ValueError, match="lr must be positive"):
        projected_gradient_descent(obj, [1.0], [0.0], [2.0], lr=0.0)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        projected_gradient_descent(obj, [1.0], [0.0], [2.0], max_iter=0)
    with pytest.raises(ValueError, match="tol must be positive"):
        projected_gradient_descent(obj, [1.0], [0.0], [2.0], tol=-1e-9)
    with pytest.raises(ValueError, match="lower <= upper"):
        projected_gradient_descent(obj, [1.0], [3.0], [2.0])


# --------------------------------------------------------------------- #
# Quadratic penalty method                                               #
# --------------------------------------------------------------------- #


def test_quadratic_penalty_drives_violation_below_1e_3() -> None:
    def f(v: list[float]) -> float:
        return (v[0] - 5.0) ** 2

    def c(v: list[float]) -> float:
        return v[0] - 2.0

    res = quadratic_penalty(f, [c], [0.0])
    assert res.converged
    assert res.iterations > 0
    assert res.x[0] == pytest.approx(2.0, abs=1e-3)
    assert abs(c(res.x)) <= 1e-3


def test_quadratic_penalty_multiple_constraints_with_inactive_one() -> None:
    def f(v: list[float]) -> float:
        return (v[0] - 4.0) ** 2 + (v[1] - 4.0) ** 2

    def c_active(v: list[float]) -> float:
        return v[0] + v[1] - 3.0

    def c_inactive(v: list[float]) -> float:
        return v[1] - 10.0

    res = quadratic_penalty(f, [c_active, c_inactive], [0.0, 0.0])
    assert res.converged
    assert res.x[0] == pytest.approx(1.5, abs=1e-2)
    assert res.x[1] == pytest.approx(1.5, abs=1e-2)
    assert c_active(res.x) <= 1e-3


def test_quadratic_penalty_more_rounds_improve_feasibility() -> None:
    def f(v: list[float]) -> float:
        return (v[0] - 5.0) ** 2

    def c(v: list[float]) -> float:
        return v[0] - 2.0

    single = quadratic_penalty(f, [c], [0.0], rounds=1)
    many = quadratic_penalty(f, [c], [0.0], rounds=6)
    assert abs(c(many.x)) < abs(c(single.x))


def test_quadratic_penalty_without_constraints_is_plain_descent() -> None:
    res = quadratic_penalty(lambda v: (v[0] - 1.0) ** 2, [], [4.0])
    assert res.converged
    assert res.x[0] == pytest.approx(1.0, abs=1e-4)
    assert res.fun == pytest.approx(0.0, abs=1e-6)


def test_penalized_objective_penalizes_only_violated_constraints() -> None:
    def f(v: list[float]) -> float:
        return v[0] ** 2

    def satisfied(v: list[float]) -> float:
        return v[0] - 5.0

    def violated(v: list[float]) -> float:
        return v[0] - 0.5

    penalized = _penalized_objective(f, [satisfied, violated], penalty=2.0)
    assert penalized([1.0]) == pytest.approx(1.0 + 0.5 * 2.0 * 0.5**2)


def test_quadratic_penalty_validation() -> None:
    def obj(v: list[float]) -> float:
        return 0.0

    with pytest.raises(ValueError, match="x0 must be non-empty"):
        quadratic_penalty(obj, [], [])
    with pytest.raises(ValueError, match="penalty0 must be positive"):
        quadratic_penalty(obj, [], [1.0], penalty0=0.0)
    with pytest.raises(ValueError, match="growth must be greater than 1"):
        quadratic_penalty(obj, [], [1.0], growth=1.0)
    with pytest.raises(ValueError, match="rounds must be >= 1"):
        quadratic_penalty(obj, [], [1.0], rounds=0)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        quadratic_penalty(obj, [], [1.0], max_iter=0)
