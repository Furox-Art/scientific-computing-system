"""Tests for :mod:`cds.diffeq.symplectic` (symplectic Euler + velocity Verlet)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest

from cds.diffeq.symplectic import (
    SymplecticSolution,
    symplectic_euler,
    velocity_verlet,
)

PERIOD = 2.0 * math.pi


def _harmonic_force(t: float, q: list[float], p: list[float]) -> list[float]:
    """Generalized force of H = p^2/2 + q^2/2 (unit harmonic oscillator)."""
    return [-qi for qi in q]


def _harmonic_energy(q: list[float], p: list[float]) -> float:
    """H = p^2/2 + q^2/2 for a single degree of freedom."""
    return 0.5 * (sum(v * v for v in q) + sum(v * v for v in p))


# --------------------------------------------------------------------- #
# Structure of the result object                                         #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("integrator", [symplectic_euler, velocity_verlet])
def test_solution_shape_includes_initial_row(
    integrator: Callable[..., SymplecticSolution],
) -> None:
    q0, p0 = [1.0, 0.5], [0.0, -0.25]
    sol = integrator(_harmonic_force, q0, p0, 0.0, 0.3, 0.1)
    assert isinstance(sol, SymplecticSolution)
    assert len(sol.ts) == len(sol.qs) == len(sol.ps)
    assert sol.ts[0] == 0.0
    assert sol.qs[0] == q0
    assert sol.ps[0] == p0


def test_solution_is_frozen() -> None:
    sol = symplectic_euler(_harmonic_force, [0.0], [1.0], 0.0, 0.1, 0.05)
    with pytest.raises(FrozenInstanceError):
        sol.ts = []  # type: ignore[misc]


# --------------------------------------------------------------------- #
# One-step semantics                                                     #
# --------------------------------------------------------------------- #


def test_symplectic_euler_step_semantics() -> None:
    # p <- p + h*f(t,q,p) first; then q <- q + h*p_new.
    sol = symplectic_euler(_harmonic_force, [0.0], [1.0], 0.0, 0.1, 0.1)
    assert sol.qs[1] == [0.1]
    assert sol.ps[1] == [1.0]


def test_velocity_verlet_step_semantics() -> None:
    # Kick-drift-kick: p_half = 1, q1 = 0.1, p1 = 1 - 0.05*q1 = 0.995.
    sol = velocity_verlet(_harmonic_force, [0.0], [1.0], 0.0, 0.1, 0.1)
    assert sol.qs[1][0] == pytest.approx(0.1)
    assert sol.ps[1][0] == pytest.approx(0.995)


def test_velocity_verlet_more_accurate_than_symplectic_euler() -> None:
    exact_q = math.cos(1.0)
    vv = velocity_verlet(_harmonic_force, [1.0], [0.0], 0.0, 1.0, 0.0625)
    se = symplectic_euler(_harmonic_force, [1.0], [0.0], 0.0, 1.0, 0.0625)
    assert abs(vv.qs[-1][0] - exact_q) < abs(se.qs[-1][0] - exact_q) / 10.0


# --------------------------------------------------------------------- #
# Long-horizon energy behaviour                                          #
# --------------------------------------------------------------------- #


def test_velocity_verlet_energy_drift_below_1e3_over_100_periods() -> None:
    sol = velocity_verlet(_harmonic_force, [1.0], [0.0], 0.0, 100 * PERIOD, 0.01)
    energies = [_harmonic_energy(q, p) for q, p in zip(sol.qs, sol.ps)]
    e0 = energies[0]
    max_drift = max(abs(e - e0) for e in energies) / e0
    assert max_drift < 1e-3


def test_symplectic_euler_energy_stays_bounded_over_100_periods() -> None:
    # First-order but symplectic: the energy error oscillates inside an
    # O(dt)-wide band instead of drifting like explicit Euler would.
    sol = symplectic_euler(_harmonic_force, [1.0], [0.0], 0.0, 100 * PERIOD, 0.01)
    energies = [_harmonic_energy(q, p) for q, p in zip(sol.qs, sol.ps)]
    e0 = energies[0]
    max_drift = max(abs(e - e0) for e in energies) / e0
    assert max_drift < 0.05
    amplitudes = max(abs(q[0]) for q in sol.qs)
    assert amplitudes < 1.05


def test_two_dof_coupled_chain_conserves_energy_with_verlet() -> None:
    k2 = 0.5

    def force(t: float, q: list[float], p: list[float]) -> list[float]:
        return [
            -(q[0] + k2 * (q[0] - q[1])),
            -(q[1] + k2 * (q[1] - q[0])),
        ]

    def total_energy(q: list[float], p: list[float]) -> float:
        kinetic = 0.5 * sum(v * v for v in p)
        potential = 0.5 * (q[0] ** 2 + q[1] ** 2) + 0.5 * k2 * (q[0] - q[1]) ** 2
        return kinetic + potential

    sol = velocity_verlet(force, [1.0, -0.5], [0.0, 0.0], 0.0, 10 * PERIOD, 0.01)
    energies = [total_energy(q, p) for q, p in zip(sol.qs, sol.ps)]
    e0 = energies[0]
    max_drift = max(abs(e - e0) for e in energies) / abs(e0)
    assert max_drift < 1e-4


# --------------------------------------------------------------------- #
# Direction handling and endpoint truncation                             #
# --------------------------------------------------------------------- #


def test_backward_round_trip_recovers_initial_conditions() -> None:
    forward = velocity_verlet(_harmonic_force, [1.0], [0.0], 0.0, PERIOD, 0.01)
    back = velocity_verlet(_harmonic_force, forward.qs[-1], forward.ps[-1], PERIOD, 0.0, 0.01)
    assert back.qs[-1][0] == pytest.approx(1.0, abs=1e-2)
    assert back.ps[-1][0] == pytest.approx(0.0, abs=1e-2)


@pytest.mark.parametrize(
    ("integrator", "dt"),
    [(symplectic_euler, 0.001), (velocity_verlet, 0.01)],
)
def test_backward_integration_matches_analytic_solution(
    integrator: Callable[..., SymplecticSolution], dt: float
) -> None:
    # q(t) = cos t, p(t) = -sin t; at t = -2*pi both return to (1, 0).
    sol = integrator(_harmonic_force, [1.0], [0.0], 0.0, -PERIOD, dt)
    assert sol.ts[-1] == pytest.approx(-PERIOD)
    # Time grid must decrease monotonically (backward integration).
    assert all(t_next < t_prev for t_prev, t_next in zip(sol.ts, sol.ts[1:]))
    assert sol.qs[-1][0] == pytest.approx(1.0, abs=0.01)
    assert sol.ps[-1][0] == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("integrator", [symplectic_euler, velocity_verlet])
def test_final_step_truncated_to_land_exactly_on_t_end(
    integrator: Callable[..., SymplecticSolution],
) -> None:
    # Span 1.0 with dt = 0.3 needs three full steps plus a truncated 0.1 step.
    sol = integrator(_harmonic_force, [1.0], [0.0], 0.0, 1.0, 0.3)
    assert len(sol.ts) == 5
    assert sol.ts[-2] == pytest.approx(0.9)
    assert sol.ts[-1] == 1.0


def test_exact_step_multiple_lands_on_t_end_without_extra_step() -> None:
    # Remaining distance equals |dt| at the last grid point: the truncation
    # edge must produce exactly one final step and no epsilon-sized extra.
    sol = symplectic_euler(_harmonic_force, [1.0], [0.0], 0.0, 1.0, 0.25)
    assert sol.ts == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert sol.ts[-1] == 1.0


# --------------------------------------------------------------------- #
# Validation                                                             #
# --------------------------------------------------------------------- #


def test_validation_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        symplectic_euler(_harmonic_force, [], [], 0.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="non-empty"):
        velocity_verlet(_harmonic_force, [1.0], [], 0.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="same length"):
        symplectic_euler(_harmonic_force, [1.0, 2.0], [0.5], 0.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="dt must be non-zero"):
        velocity_verlet(_harmonic_force, [1.0], [0.0], 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="different from t0"):
        symplectic_euler(_harmonic_force, [1.0], [0.0], 2.5, 2.5, 0.1)


def test_ragged_force_output_rejected_on_first_call() -> None:
    calls: list[float] = []

    def bad_force(t: float, q: list[float], p: list[float]) -> list[float]:
        calls.append(t)
        return [-q[0], 0.0]

    with pytest.raises(ValueError, match="same length as q"):
        velocity_verlet(bad_force, [1.0], [0.0], 0.0, 1.0, 0.25)
    assert len(calls) == 1
