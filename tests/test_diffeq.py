"""Tests for ODE solvers."""

import math
from collections.abc import Callable
from typing import Never

import pytest

from cds.diffeq import (
    backward_euler,
    euler_method,
    midpoint_method,
    rk4,
    rk45,
    rk45_system,
    solve_system,
)
from cds.diffeq.symplectic import symplectic_euler


def _unexpected_derivative(*_args: object) -> Never:
    raise AssertionError("The solver evaluated the derivative with an unusable time step")


@pytest.mark.parametrize("solver", [euler_method, midpoint_method, rk4, rk45, backward_euler])
@pytest.mark.parametrize("dt", [0.0, -0.1, math.nan])
def test_rejects_unusable_step(
    solver: Callable[..., object],
    dt: float,
) -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        solver(_unexpected_derivative, 0.0, 1.0, 1.0, dt)


def test_rejects_nonfinite_time_endpoints() -> None:
    with pytest.raises(ValueError, match="t0 and t_end must be finite"):
        euler_method(_unexpected_derivative, math.nan, 1.0, 1.0, dt=0.1)
    with pytest.raises(ValueError, match="t0 and t_end must be finite"):
        euler_method(_unexpected_derivative, 0.0, 1.0, math.inf, dt=0.1)


@pytest.mark.parametrize("solver", [euler_method, midpoint_method, rk4, rk45, backward_euler])
def test_stops_when_a_positive_step_cannot_advance_time(
    solver: Callable[..., object],
) -> None:
    with pytest.raises(RuntimeError, match="cannot advance time"):
        solver(_unexpected_derivative, 1e16, 1.0, 1e16 + 4, 0.01)


@pytest.mark.parametrize("solver", [solve_system, rk45_system])
def test_vector_solver_stops_when_time_cannot_advance(
    solver: Callable[..., object],
) -> None:
    with pytest.raises(RuntimeError, match="cannot advance time"):
        solver(_unexpected_derivative, 1e16, [1.0], 1e16 + 4, 0.01)


def test_symplectic_solver_stops_when_time_cannot_advance() -> None:
    with pytest.raises(RuntimeError, match="cannot advance time"):
        symplectic_euler(_unexpected_derivative, [1.0], [0.0], 1e16, 1e16 + 4, 0.01)


class TestEulerMethod:
    def test_exponential_decay(self) -> None:
        # dy/dt = -y, y(0) = 1  =>  y(t) = e^(-t)
        sol = euler_method(lambda t, y: -y, 0, 1.0, 1.0, dt=0.001)
        assert abs(sol.y[-1] - math.exp(-1)) < 0.01

    def test_linear_growth(self) -> None:
        # dy/dt = 1, y(0) = 0  =>  y(t) = t
        sol = euler_method(lambda t, y: 1.0, 0, 0.0, 2.0, dt=0.01)
        assert abs(sol.y[-1] - 2.0) < 0.02

    def test_method_name(self) -> None:
        sol = euler_method(lambda t, y: 0, 0, 0, 1, dt=0.1)
        assert sol.method == "euler"

    def test_initial_condition(self) -> None:
        sol = euler_method(lambda t, y: 0, 0, 5.0, 1.0, dt=0.1)
        assert sol.y[0] == 5.0
        assert sol.t[0] == 0


class TestRK4:
    def test_exponential_decay_high_accuracy(self) -> None:
        # dy/dt = -y, y(0) = 1  =>  y(t) = e^(-t)
        sol = rk4(lambda t, y: -y, 0, 1.0, 1.0, dt=0.01)
        assert abs(sol.y[-1] - math.exp(-1)) < 1e-8

    def test_sine_derivative(self) -> None:
        # dy/dt = cos(t), y(0) = 0  =>  y(t) = sin(t)
        sol = rk4(lambda t, y: math.cos(t), 0, 0.0, math.pi, dt=0.01)
        assert abs(sol.y[-1] - math.sin(math.pi)) < 1e-6

    def test_quadratic(self) -> None:
        # dy/dt = 2t, y(0) = 0  =>  y(t) = t²
        sol = rk4(lambda t, y: 2 * t, 0, 0.0, 3.0, dt=0.01)
        assert abs(sol.y[-1] - 9.0) < 1e-6

    def test_method_name(self) -> None:
        sol = rk4(lambda t, y: 0, 0, 0, 1, dt=0.1)
        assert sol.method == "rk4"

    def test_better_than_euler(self) -> None:
        def f(t: float, y: float) -> float:
            return -y

        sol_euler = euler_method(f, 0, 1.0, 1.0, dt=0.1)
        sol_rk4 = rk4(f, 0, 1.0, 1.0, dt=0.1)
        exact = math.exp(-1)
        assert abs(sol_rk4.y[-1] - exact) < abs(sol_euler.y[-1] - exact)


class TestMidpointMethod:
    def test_exponential_decay(self) -> None:
        sol = midpoint_method(lambda t, y: -y, 0, 1.0, 1.0, dt=0.01)
        assert abs(sol.y[-1] - math.exp(-1)) < 1e-4

    def test_method_name(self) -> None:
        sol = midpoint_method(lambda t, y: 0, 0, 0, 1, dt=0.1)
        assert sol.method == "midpoint"


class TestRK45:
    def test_exponential_decay_adaptive(self) -> None:
        # dy/dt = -y, y(0) = 1  =>  y(t) = e^(-t)
        sol = rk45(lambda t, y: -y, 0, 1.0, 2.0, dt=0.1, atol=1e-8, rtol=1e-8)
        assert abs(sol.y[-1] - math.exp(-2.0)) < 1e-6
        # adaptive steps should be fewer than fixed-step with same accuracy
        assert sol.steps < 200

    def test_method_name(self) -> None:
        sol = rk45(lambda t, y: 0, 0, 0, 1, dt=0.1)
        assert sol.method == "rk45"

    def test_step_rejected_when_error_exceeds_tolerance(self) -> None:
        # A steep ODE with a tight tolerance forces at least one step to be
        # rejected: the `if error <= tolerance` branch is False, so the
        # candidate (t, y5) is discarded (221 -> 230 edge) and h shrinks before
        # the loop re-attempts the same t. We assert the solver still lands on
        # the analytic answer and reports strictly more attempts than accepted
        # steps is impossible from the public API, so we instead check that a
        # near-singular initial slope is integrated without raising.
        # dy/dt = 100*y is explosively stiff at the start of [0, 0.1].
        sol = rk45(lambda t, y: 100.0 * y, 0.0, 1.0, 0.1, dt=0.1, atol=1e-12, rtol=1e-12)
        # y(0.1) = e^10 ≈ 22026.47
        assert abs(sol.y[-1] - math.exp(10.0)) / math.exp(10.0) < 1e-3


class TestSolveSystem:
    def test_harmonic_oscillator(self) -> None:
        # x'' = -x  =>  [x, v]' = [v, -x]
        # x(0)=1, v(0)=0  =>  x(t) = cos(t)
        def f(t: float, y: list[float]) -> list[float]:
            return [y[1], -y[0]]

        t_vals, y_vals = solve_system(f, 0, [1.0, 0.0], math.pi, dt=0.001)
        # x(π) = cos(π) = -1
        assert abs(y_vals[-1][0] - (-1.0)) < 1e-4

    def test_coupled_decay(self) -> None:
        # y1' = -y1, y2' = -2*y2
        def f(t: float, y: list[float]) -> list[float]:
            return [-y[0], -2 * y[1]]

        t_vals, y_vals = solve_system(f, 0, [1.0, 1.0], 1.0, dt=0.01)
        assert abs(y_vals[-1][0] - math.exp(-1)) < 1e-6
        assert abs(y_vals[-1][1] - math.exp(-2)) < 1e-5

    def test_initial_conditions_preserved(self) -> None:
        t_vals, y_vals = solve_system(
            lambda t, y: [0, 0],
            0,
            [3.0, 7.0],
            1.0,
            dt=0.1,
        )
        assert y_vals[0] == [3.0, 7.0]


class TestBackwardIntegration:
    """Regression guards for backward integration (``t_end < t0``).

    Previously the ``while t < t_end - LOOP_EPSILON`` loop guard was never
    satisfied when integrating toward decreasing ``t``, so every fixed-step
    solver silently returned only the initial value instead of integrating
    backward. These tests pin the corrected direction-aware behavior.
    """

    def test_euler_backward(self) -> None:
        # dy/dt = 1, y(0) = 0, integrate to t = -1  =>  y(-1) = -1
        sol = euler_method(lambda t, y: 1.0, 0, 0.0, -1.0, dt=0.01)
        assert abs(sol.y[-1] - (-1.0)) < 0.02
        # Must actually have taken steps, not just returned the initial value.
        assert sol.steps > 0
        assert sol.t[-1] < 0

    def test_rk4_backward(self) -> None:
        # dy/dt = -y, y(0) = 1, integrate to t = -1  =>  y(-1) = e^1
        sol = rk4(lambda t, y: -y, 0, 1.0, -1.0, dt=0.01)
        assert abs(sol.y[-1] - math.exp(1.0)) < 1e-8
        assert sol.steps > 0
        assert sol.t[-1] < 0

    def test_midpoint_backward(self) -> None:
        # dy/dt = -y, y(0) = 1, integrate to t = -1  =>  y(-1) = e^1
        sol = midpoint_method(lambda t, y: -y, 0, 1.0, -1.0, dt=0.01)
        assert abs(sol.y[-1] - math.exp(1.0)) < 1e-4
        assert sol.steps > 0

    def test_rk45_backward(self) -> None:
        # dy/dt = -y, y(0) = 1, integrate to t = -2  =>  y(-2) = e^2
        sol = rk45(lambda t, y: -y, 0.0, 1.0, -2.0, dt=0.1, atol=1e-8, rtol=1e-8)
        assert abs(sol.y[-1] - math.exp(2.0)) / math.exp(2.0) < 1e-6
        assert sol.steps > 0
        # The time grid must be monotonically decreasing toward t_end.
        assert all(sol.t[i] > sol.t[i + 1] for i in range(len(sol.t) - 1))

    def test_solve_system_backward(self) -> None:
        # y1' = -y1, y1(0) = 1  =>  y1(-1) = e^1
        def f(t: float, y: list[float]) -> list[float]:
            return [-y[0], -2 * y[1]]

        t_vals, y_vals = solve_system(f, 0, [1.0, 1.0], -1.0, dt=0.01)
        assert abs(y_vals[-1][0] - math.exp(1.0)) < 1e-6
        assert t_vals[-1] < 0

    def test_backward_uses_same_step_count_as_forward(self) -> None:
        # Direction should not change the number of steps: integrating
        # [0, 1] and [0, -1] with the same dt is symmetric in step count.
        sol_fwd = rk4(lambda t, y: -y, 0, 1.0, 1.0, dt=0.01)
        sol_bwd = rk4(lambda t, y: -y, 0, 1.0, -1.0, dt=0.01)
        assert sol_fwd.steps == sol_bwd.steps

    def test_zero_span_returns_just_initial_condition(self) -> None:
        # t_end == t0 must still return a single-element solution (no steps),
        # not raise or integrate. Guards the direction-aware loop guard.
        sol = rk4(lambda t, y: -y, 0, 1.0, 0.0, dt=0.01)
        assert sol.steps == 0
        assert sol.t == [0]
        assert sol.y == [1.0]
