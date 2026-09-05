"""Tests for Monte Carlo methods."""

import math

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

    def test_single_sample_has_zero_standard_error(self) -> None:
        result = estimate_pi(n_samples=1, seed=5)
        assert result.samples == 1
        assert result.std_error == 0.0

    @pytest.mark.parametrize("count", [0, -1])
    def test_nonpositive_sample_count_rejected(self, count: int) -> None:
        with pytest.raises(ValueError, match="n_samples must be positive"):
            estimate_pi(n_samples=count, seed=1)

    @pytest.mark.parametrize("count", [True, 1.5])
    def test_noninteger_sample_count_rejected(self, count: object) -> None:
        with pytest.raises(TypeError, match="n_samples must be an integer"):
            estimate_pi(n_samples=count, seed=1)  # type: ignore[arg-type]


class TestMCIntegrate:
    def test_integrate_x_squared(self) -> None:
        # ∫₀¹ x² dx = 1/3
        result = mc_integrate(lambda x: x**2, 0, 1, n_samples=50_000, seed=42)
        assert abs(result.estimate - 1 / 3) < 0.02

    def test_integrate_sine(self) -> None:
        # ∫₀^π sin(x) dx = 2
        result = mc_integrate(math.sin, 0, math.pi, n_samples=50_000, seed=42)
        assert abs(result.estimate - 2.0) < 0.1

    def test_standard_error(self) -> None:
        result = mc_integrate(lambda x: x, 0, 1, n_samples=10_000, seed=42)
        assert result.std_error > 0

    def test_reversed_bounds_keep_standard_error_nonnegative(self) -> None:
        result = mc_integrate(lambda x: x, 1.0, 0.0, n_samples=10_000, seed=42)
        assert abs(result.estimate + 0.5) < 0.02
        assert result.std_error > 0.0

    def test_single_sample_is_valid(self) -> None:
        result = mc_integrate(lambda _x: 3.0, 0.0, 2.0, n_samples=1, seed=1)
        assert result.estimate == 6.0
        assert result.std_error == 0.0

    @pytest.mark.parametrize("count", [0, -5])
    def test_nonpositive_sample_count_rejected(self, count: int) -> None:
        with pytest.raises(ValueError, match="n_samples must be positive"):
            mc_integrate(lambda x: x, 0.0, 1.0, n_samples=count)

    @pytest.mark.parametrize("a,b", [(math.inf, 1.0), (0.0, math.nan)])
    def test_nonfinite_bounds_rejected(self, a: float, b: float) -> None:
        with pytest.raises(ValueError, match="interval bounds must be finite"):
            mc_integrate(lambda x: x, a, b, n_samples=10)


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

    def test_zero_step_walk_is_origin(self) -> None:
        assert random_walk_1d(0, seed=1) == [0.0]


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

    def test_zero_step_walk_is_origin(self) -> None:
        assert random_walk_2d(0, seed=1) == [(0.0, 0.0)]


@pytest.mark.parametrize("walk", [random_walk_1d, random_walk_2d])
def test_random_walk_rejects_negative_steps(walk: object) -> None:
    callable_walk = walk
    assert callable(callable_walk)
    with pytest.raises(ValueError, match="steps must be non-negative"):
        callable_walk(-1)  # type: ignore[operator]


@pytest.mark.parametrize("walk", [random_walk_1d, random_walk_2d])
@pytest.mark.parametrize("steps", [True, 1.5])
def test_random_walk_rejects_noninteger_steps(walk: object, steps: object) -> None:
    callable_walk = walk
    assert callable(callable_walk)
    with pytest.raises(TypeError, match="steps must be an integer"):
        callable_walk(steps)  # type: ignore[operator]


@pytest.mark.parametrize("walk", [random_walk_1d, random_walk_2d])
@pytest.mark.parametrize("step_size", [-1.0, math.inf])
def test_random_walk_rejects_invalid_step_size(walk: object, step_size: float) -> None:
    callable_walk = walk
    assert callable(callable_walk)
    with pytest.raises(ValueError, match="step_size must be finite and non-negative"):
        callable_walk(1, step_size=step_size)  # type: ignore[operator]


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
        with pytest.raises(ValueError, match="shorter"):
            buffon_needle(needle_length=3.0, line_spacing=2.0)

    def test_reproducible(self) -> None:
        r1 = buffon_needle(n_throws=1000, seed=42)
        r2 = buffon_needle(n_throws=1000, seed=42)
        assert r1.estimate == r2.estimate

    @pytest.mark.parametrize("needle", [0.0, -1.0, math.inf])
    def test_invalid_needle_length_rejected(self, needle: float) -> None:
        with pytest.raises(ValueError, match="needle_length must be finite and positive"):
            buffon_needle(needle_length=needle, n_throws=10)

    @pytest.mark.parametrize("spacing", [0.0, -1.0, math.inf])
    def test_invalid_line_spacing_rejected(self, spacing: float) -> None:
        with pytest.raises(ValueError, match="line_spacing must be finite and positive"):
            buffon_needle(line_spacing=spacing, n_throws=10)

    @pytest.mark.parametrize("throws", [0, -1])
    def test_nonpositive_throw_count_rejected(self, throws: int) -> None:
        with pytest.raises(ValueError, match="n_throws must be positive"):
            buffon_needle(n_throws=throws)

    def test_noninteger_throw_count_rejected(self) -> None:
        with pytest.raises(TypeError, match="n_throws must be an integer"):
            buffon_needle(n_throws=1.5)  # type: ignore[arg-type]


def test_mc_expectation() -> None:
    # E[x] on [0,1] ≈ 0.5
    r = mc_expectation(lambda x: x, n_samples=20_000, a=0.0, b=1.0, seed=1)
    assert abs(r.estimate - 0.5) < 0.05


def test_mc_expectation_single_sample() -> None:
    r = mc_expectation(lambda _x: 2.0, n_samples=1, a=0.0, b=1.0, seed=1)
    assert r.estimate == 2.0
    assert r.std_error == 0.0


def test_hit_or_miss_unit_disk() -> None:
    r = hit_or_miss(
        lambda x, y: x * x + y * y <= 1.0,
        (-1.0, 1.0),
        (-1.0, 1.0),
        n_samples=50_000,
        seed=2,
    )
    assert abs(r.estimate - math.pi) < 0.15


def test_hit_or_miss_single_sample() -> None:
    r = hit_or_miss(lambda _x, _y: True, (0.0, 2.0), (0.0, 3.0), n_samples=1, seed=1)
    assert r.estimate == 6.0
    assert r.std_error == 0.0


def test_mc_new_errors() -> None:
    with pytest.raises(ValueError, match="n_samples must be positive"):
        mc_expectation(lambda x: x, n_samples=0)
    with pytest.raises(ValueError, match="ranges must be non-empty"):
        hit_or_miss(lambda x, y: True, (1.0, 0.0), (0.0, 1.0), n_samples=10)


def test_mc_expectation_a_ge_b() -> None:
    with pytest.raises(ValueError, match="a must be less than b"):
        mc_expectation(lambda x: x, n_samples=10, a=1.0, b=0.0)


def test_mc_expectation_nonfinite_bounds() -> None:
    with pytest.raises(ValueError, match="interval bounds must be finite"):
        mc_expectation(lambda x: x, n_samples=10, a=0.0, b=math.inf)


def test_hit_or_miss_n_samples() -> None:
    with pytest.raises(ValueError, match="n_samples must be positive"):
        hit_or_miss(lambda x, y: True, (0.0, 1.0), (0.0, 1.0), n_samples=0)


def test_hit_or_miss_nonfinite_ranges() -> None:
    with pytest.raises(ValueError, match="ranges must contain finite bounds"):
        hit_or_miss(lambda x, y: True, (0.0, math.inf), (0.0, 1.0), n_samples=10)
