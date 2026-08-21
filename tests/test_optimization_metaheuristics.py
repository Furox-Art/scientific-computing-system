"""Tests for :mod:`cds.optimization._metaheuristics` (NM + annealing)."""

from __future__ import annotations

import math

import pytest

from cds.optimization import nelder_mead, simulated_annealing
from cds.optimization._metaheuristics import (
    _clamp_point,
    _metropolis_accept,
    _nm_action,
)
from cds.optimization.minimize import OptResult

# --------------------------------------------------------------------- #
# Pure helpers (every branch exercised deterministically)                #
# --------------------------------------------------------------------- #


def test_nm_action_covers_all_four_moves() -> None:
    assert _nm_action(f_r=0.0, f_best=1.0, f_second_worst=2.0, f_worst=3.0) == "expand"
    assert _nm_action(f_r=1.5, f_best=1.0, f_second_worst=2.0, f_worst=3.0) == "reflect"
    assert _nm_action(f_r=2.5, f_best=1.0, f_second_worst=2.0, f_worst=3.0) == "contract-outside"
    assert _nm_action(f_r=3.5, f_best=1.0, f_second_worst=2.0, f_worst=3.0) == "contract-inside"


def test_metropolis_accept_branches() -> None:
    assert _metropolis_accept(delta=-1.0, temperature=1.0, u=0.999999)
    assert not _metropolis_accept(delta=10.0, temperature=1.0, u=0.99)
    assert _metropolis_accept(delta=0.001, temperature=1.0, u=0.99)
    assert not _metropolis_accept(delta=0.001, temperature=1e-9, u=0.99)


def test_clamp_point_branches() -> None:
    assert _clamp_point([5.0], None) == [5.0]
    clamped = _clamp_point([-3.0, 0.5, 7.0], [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)])
    assert clamped == [0.0, 0.5, 2.0]


# --------------------------------------------------------------------- #
# Nelder–Mead integration                                                #
# --------------------------------------------------------------------- #


def test_nelder_meed_solves_rosenbrock() -> None:
    def rosenbrock(v: list[float]) -> float:
        x, y = v
        return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2

    res = nelder_mead(rosenbrock, [-1.2, 1.0], step=0.5, max_iter=5000)
    assert res.converged
    assert res.x[0] == pytest.approx(1.0, abs=1e-3)
    assert res.x[1] == pytest.approx(1.0, abs=1e-3)
    assert isinstance(res, OptResult)


def test_nelder_mead_sphere_1d() -> None:
    res = nelder_mead(lambda v: v[0] ** 2 + 3.0, [4.0])
    assert res.value == pytest.approx(3.0, abs=1e-8)
    assert res.x[0] == pytest.approx(0.0, abs=1e-4)


def test_nelder_mead_escapes_symmetric_false_convergence() -> None:
    # A pure objective-spread criterion stops instantly once the simplex
    # becomes symmetric (±ε gives equal f) even far from the optimum; the
    # diameter criterion must keep it running until it truly reaches x=0.
    res = nelder_mead(lambda v: v[0] ** 2, [4.0])
    assert res.converged
    assert abs(res.x[0]) < 1e-5


def test_nelder_mead_shrinks_toward_valley_minimum() -> None:
    # |x| has a non-smooth kink at 0: reflections overshoot, contractions
    # oscillate, and the simplex must shrink its way down into the minimum.
    res = nelder_mead(lambda v: abs(v[0]), [1.0], step=0.25, max_iter=200)
    assert res.x[0] == pytest.approx(0.0, abs=1e-6)
    assert res.value == pytest.approx(0.0, abs=1e-6)


def test_nelder_mead_max_iter_reported_when_not_converged() -> None:
    res = nelder_mead(
        lambda v: math.sin(v[0]) * 1000.0 + v[0] ** 2, [3.0], step=0.01, tol_fx=1e-15, max_iter=5
    )
    assert not res.converged
    assert res.iterations == 5


def test_nelder_mead_validation() -> None:
    with pytest.raises(ValueError, match="x0 must be non-empty"):
        nelder_mead(lambda v: 0.0, [])
    with pytest.raises(ValueError, match="step must be positive"):
        nelder_mead(lambda v: 0.0, [1.0], step=0.0)
    with pytest.raises(ValueError, match="tol_fx must be positive"):
        nelder_mead(lambda v: 0.0, [1.0], tol_fx=-1.0)
    with pytest.raises(ValueError, match="tol_x must be positive"):
        nelder_mead(lambda v: 0.0, [1.0], tol_x=0.0)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        nelder_mead(lambda v: 0.0, [1.0], max_iter=0)


def test_nelder_mead_shrinks_when_contractions_fail_on_rosenbrock() -> None:
    # A big initial simplex over Rosenbrock's curved valley forces rejected
    # contractions and full shrinks before converging into the minimum.
    def rosenbrock(v: list[float]) -> float:
        x, y = v
        return (1.0 - x) ** 2 + 100.0 * (y - x * x) ** 2

    res = nelder_mead(rosenbrock, [1.0, 2.0], step=2.0, max_iter=4000)
    assert res.converged
    assert res.x[0] == pytest.approx(1.0, abs=1e-3)
    assert res.x[1] == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------- #
# Simulated annealing integration                                        #
# --------------------------------------------------------------------- #


def test_annealing_finds_global_basin_of_bumpy_function() -> None:
    # Two basins: local min at x≈-1.5 (value ≈ -0.65), global at x≈+2.
    def bumpy(v: list[float]) -> float:
        return math.sin(3.0 * v[0]) + 0.2 * v[0] ** 2

    res = simulated_annealing(
        bumpy,
        [-1.5],
        t_init=2.0,
        cooling=0.99,
        sigma=0.5,
        max_iter=20_000,
        seed=42,
    )
    assert res.value == pytest.approx(bumpy([res.x[0]]))
    assert res.value < bumpy([-1.5]) - 0.3  # escaped the local basin


def test_annealing_deterministic_given_seed() -> None:
    def obj(v: list[float]) -> float:
        return sum(xi * xi for xi in v)

    a = simulated_annealing(obj, [2.0, -1.0], seed=7, max_iter=500)
    b = simulated_annealing(obj, [2.0, -1.0], seed=7, max_iter=500)
    assert a == b


def test_annealing_respects_bounds() -> None:
    def obj(v: list[float]) -> float:
        return v[0] ** 2

    res = simulated_annealing(
        obj,
        [9.0],
        bounds=[(-1.0, 1.0)],
        sigma=5.0,
        max_iter=2000,
        seed=1,
    )
    assert -1.0 <= res.x[0] <= 1.0


def test_annealing_reaches_cooling_floor() -> None:
    def obj(v: list[float]) -> float:
        return v[0] ** 2

    res = simulated_annealing(
        obj, [1.0], t_init=1.0, t_min=0.1, cooling=0.9, sigma=0.05, max_iter=10_000, seed=0
    )
    assert res.converged
    assert res.iterations < 10_000  # floor hit before budget exhaustion


def test_annealing_budget_exhaustion_without_floor() -> None:
    # Cooling so slow that the floor is never reached inside the budget:
    # exercises the loop-exhaustion exit with converged=False.
    def obj(v: list[float]) -> float:
        return v[0] ** 2

    res = simulated_annealing(
        obj,
        [5.0],
        t_init=1.0,
        t_min=1e-8,
        cooling=0.9999,
        sigma=0.01,
        max_iter=50,
        seed=4,
    )
    assert not res.converged
    assert res.iterations == 50


def test_annealing_improves_on_start() -> None:
    def obj(v: list[float]) -> float:
        return (v[0] - 3.0) ** 2

    res = simulated_annealing(obj, [10.0], sigma=0.4, cooling=0.98, max_iter=5000, seed=3)
    assert res.value <= obj([10.0])


def test_annealing_validation() -> None:
    def obj(v: list[float]) -> float:
        return 0.0

    with pytest.raises(ValueError, match="x0 must be non-empty"):
        simulated_annealing(obj, [])
    with pytest.raises(ValueError, match="t_init must be positive"):
        simulated_annealing(obj, [0.0], t_init=0.0)
    with pytest.raises(ValueError, match=r"cooling must be in \(0, 1\]"):
        simulated_annealing(obj, [0.0], cooling=1.5)
    with pytest.raises(ValueError, match="t_min must be positive"):
        simulated_annealing(obj, [0.0], t_min=-1.0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        simulated_annealing(obj, [0.0], sigma=0.0)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        simulated_annealing(obj, [0.0], max_iter=0)
    with pytest.raises(ValueError, match="low < high"):
        simulated_annealing(obj, [0.0], bounds=[(2.0, 1.0)])
    with pytest.raises(ValueError, match="per axis"):
        simulated_annealing(obj, [0.0, 1.0], bounds=[(-1.0, 1.0)])
