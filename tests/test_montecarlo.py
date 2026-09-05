"""Tests for Monte Carlo methods."""

import math
from typing import cast
from unittest import mock

import pytest

from cds.montecarlo import (
    buffon_needle,
    estimate_pi,
    hit_or_miss,
    mc_expectation,
    mc_integrate,
    random_walk_1d,
    random_walk_2d,
)


class TestEstimatePi:
    def test_pi_within_tolerance(self) -> None:
        result = estimate_pi(n_samples=50_000, seed=42)
        assert abs(result.estimate - math.pi) < 0.1

    def test_standard_error_positive(self) -> None:
        result = estimate_pi(n_samples=1000, seed=42)
        assert result.std_error > 0

    def test_samples_recorded(self) -> None:
        result = estimate_pi(n_samples=500, seed=42)
        assert result.samples == 500

    def test_reproducible_with_seed(self) -> None:
        r1 = estimate_pi(n_samples=1000, seed=123)
        r2 = estimate_pi(n_samples=1000, seed=123)
        assert r1.estimate == r2.estimate

    def test_single_sample_and_generated_seed(self) -> None:
        result = estimate_pi(n_samples=1)
        assert result.samples == 1
        assert result.std_error == 0.0

    def test_sample_count_is_strict_positive_integer(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            estimate_pi(n_samples=0)
        with pytest.raises(ValueError, match="n_samples"):
            estimate_pi(n_samples=True)
        with pytest.raises(ValueError, match="n_samples"):
            estimate_pi(n_samples=cast(int, 1.5))


class TestMCIntegrate:
    def test_integrate_x_squared(self) -> None:
        result = mc_integrate(lambda x: x**2, 0, 1, n_samples=50_000, seed=42)
        assert abs(result.estimate - 1 / 3) < 0.02

    def test_integrate_sine(self) -> None:
        result = mc_integrate(math.sin, 0, math.pi, n_samples=50_000, seed=42)
        assert abs(result.estimate - 2.0) < 0.1

    def test_standard_error(self) -> None:
        result = mc_integrate(lambda x: x, 0, 1, n_samples=10_000, seed=42)
        assert result.std_error > 0

    def test_reversed_and_zero_width_intervals(self) -> None:
        reversed_result = mc_integrate(lambda _x: 2.0, 1.0, 0.0, n_samples=2, seed=1)
        assert reversed_result.estimate == pytest.approx(-2.0)
        assert reversed_result.std_error == 0.0

        zero = mc_integrate(lambda _x: 3.0, 2.0, 2.0, n_samples=1, seed=1)
        assert zero.estimate == 0.0
        assert zero.std_error == 0.0

    def test_rejects_invalid_sample_count_and_bounds(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            mc_integrate(lambda x: x, 0.0, 1.0, n_samples=0)
        with pytest.raises(ValueError, match="a must be finite"):
            mc_integrate(lambda x: x, math.nan, 1.0, n_samples=1)
        with pytest.raises(ValueError, match="b must be finite"):
            mc_integrate(lambda x: x, 0.0, math.inf, n_samples=1)
        with pytest.raises(ArithmeticError, match="interval width"):
            mc_integrate(lambda x: x, -1e308, 1e308, n_samples=1)

    def test_rejects_nonfinite_integrand_and_accumulation(self) -> None:
        with pytest.raises(ValueError, match="integrand"):
            mc_integrate(lambda _x: math.nan, 0.0, 1.0, n_samples=1)
        with pytest.raises(ArithmeticError, match="accumulation"):
            mc_integrate(lambda _x: 1e308, 0.0, 1.0, n_samples=1)

    def test_rejects_overflowed_integral_and_standard_error(self) -> None:
        with pytest.raises(ArithmeticError, match="integral became non-finite"):
            mc_integrate(lambda _x: 1e154, 0.0, 1e155, n_samples=1, seed=1)

        with mock.patch(
            "cds.montecarlo.methods.random.Random.random", side_effect=[0.25, 0.75]
        ):
            with pytest.raises(ArithmeticError, match="standard error"):
                mc_integrate(
                    lambda x: -2.0 if x < 0.0 else 2.0,
                    -8e307,
                    8e307,
                    n_samples=2,
                )


class TestRandomWalk1D:
    def test_starts_at_zero(self) -> None:
        walk = random_walk_1d(10, seed=42)
        assert walk[0] == 0.0

    def test_correct_length(self) -> None:
        walk = random_walk_1d(100, seed=42)
        assert len(walk) == 101

    def test_step_size(self) -> None:
        walk = random_walk_1d(1, step_size=2.5, seed=42)
        assert abs(walk[1]) == 2.5

    def test_reproducible(self) -> None:
        w1 = random_walk_1d(50, seed=99)
        w2 = random_walk_1d(50, seed=99)
        assert w1 == w2

    def test_validation_and_overflow(self) -> None:
        with pytest.raises(ValueError, match="steps"):
            random_walk_1d(-1)
        with pytest.raises(ValueError, match="steps"):
            random_walk_1d(True)
        with pytest.raises(ValueError, match="step_size must be finite"):
            random_walk_1d(1, step_size=math.nan)
        with pytest.raises(ValueError, match="non-negative"):
            random_walk_1d(1, step_size=-1.0)
        with mock.patch("cds.montecarlo.methods.random.Random.random", return_value=0.0):
            with pytest.raises(ArithmeticError, match="position"):
                random_walk_1d(2, step_size=1e308)


class TestRandomWalk2D:
    def test_starts_at_origin(self) -> None:
        walk = random_walk_2d(10, seed=42)
        assert walk[0] == (0.0, 0.0)

    def test_correct_length(self) -> None:
        walk = random_walk_2d(50, seed=42)
        assert len(walk) == 51

    def test_step_distance(self) -> None:
        walk = random_walk_2d(1, step_size=1.0, seed=42)
        x, y = walk[1]
        dist = math.hypot(x, y)
        assert abs(dist - 1.0) < 1e-10

    def test_validation_and_overflow(self) -> None:
        with pytest.raises(ValueError, match="steps"):
            random_walk_2d(cast(int, 1.5))
        with pytest.raises(ValueError, match="step_size must be finite"):
            random_walk_2d(1, step_size=math.inf)
        with pytest.raises(ValueError, match="non-negative"):
            random_walk_2d(1, step_size=-1.0)
        with mock.patch("cds.montecarlo.methods.random.Random.uniform", return_value=0.0):
            with pytest.raises(ArithmeticError, match="position"):
                random_walk_2d(2, step_size=1e308)


class TestBuffonNeedle:
    def test_pi_estimate(self) -> None:
        result = buffon_needle(
            needle_length=1.0,
            line_spacing=2.0,
            n_throws=50_000,
            seed=42,
        )
        assert abs(result.estimate - math.pi) < 0.2

    def test_needle_too_long_raises(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            buffon_needle(needle_length=3.0, line_spacing=2.0)

    def test_reproducible(self) -> None:
        r1 = buffon_needle(n_throws=1000, seed=42)
        r2 = buffon_needle(n_throws=1000, seed=42)
        assert r1.estimate == r2.estimate

    def test_validation_and_zero_crossing_estimator(self) -> None:
        with pytest.raises(ValueError, match="n_throws"):
            buffon_needle(n_throws=0)
        with pytest.raises(ValueError, match="needle_length must be finite"):
            buffon_needle(needle_length=math.nan)
        with pytest.raises(ValueError, match="line_spacing must be finite"):
            buffon_needle(line_spacing=math.inf)
        with pytest.raises(ValueError, match="must be positive"):
            buffon_needle(needle_length=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            buffon_needle(needle_length=1.0, line_spacing=0.0)
        with pytest.raises(ArithmeticError, match="zero crossings"):
            buffon_needle(needle_length=1e-300, line_spacing=1.0, n_throws=1, seed=1)


def test_mc_expectation() -> None:
    r = mc_expectation(lambda x: x, n_samples=20_000, a=0.0, b=1.0, seed=1)
    assert abs(r.estimate - 0.5) < 0.05


def test_mc_expectation_validation_and_numerical_failures() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        mc_expectation(lambda x: x, n_samples=0)
    with pytest.raises(ValueError, match="a must be finite"):
        mc_expectation(lambda x: x, n_samples=1, a=math.nan, b=1.0)
    with pytest.raises(ValueError, match="b must be finite"):
        mc_expectation(lambda x: x, n_samples=1, a=0.0, b=math.inf)
    with pytest.raises(ValueError, match="a must be less than b"):
        mc_expectation(lambda x: x, n_samples=10, a=1.0, b=0.0)
    with pytest.raises(ArithmeticError, match="interval width"):
        mc_expectation(lambda x: x, n_samples=1, a=-1e308, b=1e308)
    with pytest.raises(ValueError, match="integrand"):
        mc_expectation(lambda _x: math.inf, n_samples=1)
    with pytest.raises(ArithmeticError, match="accumulation"):
        mc_expectation(lambda _x: 1e308, n_samples=1)


def test_hit_or_miss_unit_disk() -> None:
    r = hit_or_miss(
        lambda x, y: x * x + y * y <= 1.0,
        (-1.0, 1.0),
        (-1.0, 1.0),
        n_samples=50_000,
        seed=2,
    )
    assert abs(r.estimate - math.pi) < 0.15


def test_hit_or_miss_validation_and_overflow() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        hit_or_miss(lambda x, y: True, (0.0, 1.0), (0.0, 1.0), n_samples=0)
    with pytest.raises(ValueError, match="x0 must be finite"):
        hit_or_miss(lambda x, y: True, (math.nan, 1.0), (0.0, 1.0), n_samples=1)
    with pytest.raises(ValueError, match="ranges"):
        hit_or_miss(lambda x, y: True, (1.0, 0.0), (0.0, 1.0), n_samples=10)
    with pytest.raises(ArithmeticError, match="x-range width"):
        hit_or_miss(lambda x, y: True, (-1e308, 1e308), (0.0, 1.0), n_samples=1)
    with pytest.raises(ArithmeticError, match="bounding-box area"):
        hit_or_miss(
            lambda x, y: True,
            (0.0, 1e200),
            (0.0, 1e200),
            n_samples=1,
        )
