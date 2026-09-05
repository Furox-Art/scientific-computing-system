"""Regression coverage for defensive scientific-audit guard paths."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

import pytest

import cds.montecarlo.methods as montecarlo_methods
import cds.quantum.multi_qubit as multi_qubit
from cds.quantum.multi_qubit import QuantumRegister


def _fail_isfinite_on_call(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    target_call: int,
) -> None:
    """Make one selected ``math.isfinite`` check fail while preserving all others."""
    original: Callable[[float], bool] = module.math.isfinite
    calls = 0

    def controlled(value: float) -> bool:
        nonlocal calls
        calls += 1
        if calls == target_call:
            return False
        return original(value)

    monkeypatch.setattr(module.math, "isfinite", controlled)


def test_buffon_needle_rejects_nonfinite_final_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two input finiteness checks run first; the third check validates the
    # derived estimator. Seed 1 guarantees a crossing for this one-throw case.
    _fail_isfinite_on_call(monkeypatch, montecarlo_methods, 3)
    with pytest.raises(ArithmeticError, match="Buffon estimator became non-finite"):
        montecarlo_methods.buffon_needle(
            needle_length=1.0,
            line_spacing=1.0,
            n_throws=1,
            seed=1,
        )


def test_mc_expectation_rejects_nonfinite_derived_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With one sample, the eighth finiteness check is the derived variance
    # guard after finite inputs, width, integrand, and accumulators.
    _fail_isfinite_on_call(monkeypatch, montecarlo_methods, 8)
    with pytest.raises(ArithmeticError, match="Monte Carlo expectation became non-finite"):
        montecarlo_methods.mc_expectation(lambda _x: 1.0, n_samples=1, seed=0)


def test_mc_expectation_rejects_nonfinite_standard_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The ninth check validates the final standard error.
    _fail_isfinite_on_call(monkeypatch, montecarlo_methods, 9)
    with pytest.raises(ArithmeticError, match="Monte Carlo standard error became non-finite"):
        montecarlo_methods.mc_expectation(lambda _x: 1.0, n_samples=1, seed=0)


def test_hit_or_miss_rejects_nonfinite_final_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Four bound checks, two width checks, and the finite box-area check run
    # before the eighth check validates the derived estimate.
    _fail_isfinite_on_call(monkeypatch, montecarlo_methods, 8)
    with pytest.raises(ArithmeticError, match="hit-or-miss estimate became non-finite"):
        montecarlo_methods.hit_or_miss(
            lambda _x, _y: True,
            (0.0, 1.0),
            (0.0, 1.0),
            n_samples=1,
            seed=0,
        )


def test_quantum_normalize_rejects_nonfinite_accumulated_norm() -> None:
    register = QuantumRegister.zeros(1)
    # Each squared magnitude is finite (1e308), while their sum overflows to
    # infinity. This exercises the accumulated-norm guard rather than the
    # per-amplitude input validation in __post_init__.
    register.amplitudes = [complex(1e154, 0.0), complex(1e154, 0.0)]

    with pytest.raises(ValueError, match="quantum register norm must be finite"):
        register.normalize()


def test_measure_shots_falls_back_to_final_basis_state_on_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register = QuantumRegister(
        2,
        [complex(0.1, 0.0), complex(0.1, 0.0), complex(0.1, 0.0), complex(2.3, 0.0)],
    )
    probabilities = register.probabilities()
    cumulative = sum(probabilities)
    assert cumulative < 1.0

    class BoundaryRandom:
        def random(self) -> float:
            return cumulative

    monkeypatch.setattr(multi_qubit.random, "Random", lambda _seed=None: BoundaryRandom())

    counts = register.measure_shots(shots=1, seed=0)
    assert counts == {"11": 1}
