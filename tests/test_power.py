"""Tests for :mod:`cds.stats.power` (statistical power analysis)."""

from __future__ import annotations

import math

import pytest

from cds.stats.hypothesis_tests import t_sf
from cds.stats.power import (
    PowerResult,
    _invert_decreasing,
    _norm_sf,
    _tail_upper_t,
    power_proportion_test,
    power_t_test,
    required_n_per_group,
)


class TestPowerResult:
    def test_stores_all_fields(self) -> None:
        r = PowerResult(power=0.81, alpha=0.05, effect_size=0.5, n_per_group=64, n=128)
        assert r.power == 0.81
        assert r.alpha == 0.05
        assert r.effect_size == 0.5
        assert r.n_per_group == 64
        assert r.n == 128

    def test_sample_size_fields_optional(self) -> None:
        r = PowerResult(power=0.5, alpha=0.01, effect_size=-0.3)
        assert r.n_per_group is None
        assert r.n is None


class TestTailHelpers:
    def test_norm_sf_center(self) -> None:
        assert _norm_sf(0.0) == pytest.approx(0.5, abs=1e-12)

    def test_norm_sf_known_value(self) -> None:
        """P(Z >= 1.959964) = 0.025: the two-sided 5% critical point."""
        assert _norm_sf(1.959964) == pytest.approx(0.025, abs=1e-6)

    def test_tail_upper_t_half_at_zero(self) -> None:
        assert _tail_upper_t(0.0, 10.0) == pytest.approx(0.5, abs=1e-12)

    def test_tail_upper_t_positive_arm_is_half_two_tailed(self) -> None:
        assert _tail_upper_t(2.0, 10.0) == pytest.approx(0.5 * t_sf(2.0, 10.0))

    def test_tail_upper_t_negative_arm_complements(self) -> None:
        assert _tail_upper_t(-1.5, 8.0) == pytest.approx(1.0 - 0.5 * t_sf(1.5, 8.0))


class TestInvertDecreasing:
    def test_exp_inverse_with_inner_bracket(self) -> None:
        """exp(-x) = 0.5 at x = ln 2; the default bracket already contains it."""
        root = _invert_decreasing(lambda x: math.exp(-x), 0.5)
        assert root == pytest.approx(math.log(2.0), abs=1e-12)

    def test_expands_bracket_leftward(self) -> None:
        """target=0.9 exceeds func(-1)=0.8413 for the normal sf, forcing the
        left endpoint to expand; root is Phi^-1(0.1) = -1.2815515655."""
        root = _invert_decreasing(_norm_sf, 0.9)
        assert root == pytest.approx(-1.2815515655446004, abs=1e-9)


class TestPowerTTestKnownValues:
    def test_d08_n64_verified_reference(self) -> None:
        """Shift-method reference verified against this implementation and
        noncentral-t tables: delta = 0.8*sqrt(32) = 4.5255 on df = 126 with
        critical t = 1.9790 gives power ~= 0.9946 (the oft-quoted 0.92 for
        this pairing belongs to other n conventions)."""
        assert power_t_test(0.8, 64) == pytest.approx(0.9946, abs=0.02)

    def test_d05_n64_classic_value(self) -> None:
        assert power_t_test(0.5, 64) == pytest.approx(0.8023, abs=0.02)

    def test_one_sided_d05_n64(self) -> None:
        assert power_t_test(0.5, 64, two_sided=False) == pytest.approx(0.8792, abs=0.02)

    def test_null_like_effect_sits_at_alpha(self) -> None:
        p = power_t_test(0.01, 2)
        assert 0.04 < p < 0.06


class TestPowerTTestMonotonicity:
    def test_power_strictly_increases_with_n(self) -> None:
        powers = [power_t_test(0.5, n) for n in (8, 16, 32, 64)]
        assert powers[0] < powers[1] < powers[2] < powers[3]

    def test_power_strictly_increases_with_effect_size(self) -> None:
        powers = [power_t_test(d, 32) for d in (0.2, 0.5, 0.8)]
        assert powers[0] < powers[1] < powers[2]

    def test_small_n_stays_in_sane_band(self) -> None:
        assert 0.05 < power_t_test(0.5, 2) < 0.5


class TestPowerTTestHighAlpha:
    def test_two_sided_high_alpha(self) -> None:
        """alpha = 0.9 keeps power >= level and exercises near-zero criticals."""
        assert 0.9 <= power_t_test(0.5, 30, alpha=0.9) <= 1.0

    def test_one_sided_high_alpha_expands_left_bracket(self) -> None:
        """The one-sided critical value goes negative here (level > tail(-1)),
        driving the geometric left-bracket expansion of the inverter."""
        assert 0.9 <= power_t_test(0.5, 30, alpha=0.9, two_sided=False) <= 1.0


class TestPowerTTestValidation:
    def test_zero_effect_size_raises(self) -> None:
        with pytest.raises(ValueError, match="effect_size"):
            power_t_test(0.0, 10)

    @pytest.mark.parametrize("bad_n", [0, 1, -3])
    def test_tiny_n_raises(self, bad_n: int) -> None:
        with pytest.raises(ValueError, match="n_per_group"):
            power_t_test(0.5, bad_n)

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
    def test_bad_alpha_raises(self, bad_alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            power_t_test(0.5, 10, alpha=bad_alpha)


class TestRequiredNPerGroup:
    def test_medium_effect_matches_classic_table(self) -> None:
        """Cohen's table / statsmodels give ~63.8 per group for d = 0.5,
        power 0.8, two-sided alpha 0.05."""
        assert 62 <= required_n_per_group(0.5) <= 66

    def test_result_is_minimal_and_sufficient(self) -> None:
        n = required_n_per_group(0.5)
        assert power_t_test(0.5, n) >= 0.8
        assert power_t_test(0.5, n - 1) < 0.8

    def test_huge_effect_hits_floor(self) -> None:
        assert required_n_per_group(8.0) == 2

    def test_unattainable_effect_raises_at_cap(self) -> None:
        with pytest.raises(ValueError, match="unattainable"):
            required_n_per_group(0.01)

    def test_one_sided_needs_no_fewer_than_floor_or_more_than_two_sided(self) -> None:
        one = required_n_per_group(0.5, two_sided=False)
        assert 2 <= one <= required_n_per_group(0.5)

    def test_smaller_alpha_needs_more_n(self) -> None:
        assert required_n_per_group(0.5, alpha=0.01) > required_n_per_group(0.5)

    def test_higher_target_needs_more_n(self) -> None:
        assert required_n_per_group(0.5, target_power=0.9) > required_n_per_group(0.5)

    def test_zero_effect_size_raises(self) -> None:
        with pytest.raises(ValueError, match="effect_size"):
            required_n_per_group(0.0)

    @pytest.mark.parametrize("bad_target", [0.0, 1.0, -0.25, 1.75])
    def test_bad_target_power_raises(self, bad_target: float) -> None:
        with pytest.raises(ValueError, match="target_power"):
            required_n_per_group(0.5, target_power=bad_target)

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0])
    def test_bad_alpha_raises(self, bad_alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            required_n_per_group(0.5, alpha=bad_alpha)


class TestPowerProportionTest:
    def test_known_value_p05_p06_n100(self) -> None:
        """Pooled-null SE = sqrt(2*0.55*0.45/100), unpooled alternative SE =
        sqrt((0.25 + 0.24)/100): analytic power ~= 0.2944."""
        assert power_proportion_test(0.5, 0.6, 100) == pytest.approx(0.2944, abs=0.002)

    def test_argument_symmetry(self) -> None:
        low = power_proportion_test(0.5, 0.6, 100)
        high = power_proportion_test(0.6, 0.5, 100)
        assert high == pytest.approx(low, abs=1e-12)

    def test_equal_proportions_collapse_to_exact_size(self) -> None:
        assert power_proportion_test(0.3, 0.3, 80) == pytest.approx(0.05, abs=1e-9)

    def test_equal_proportions_custom_alpha(self) -> None:
        assert power_proportion_test(0.2, 0.2, 50, alpha=0.01) == pytest.approx(0.01, abs=1e-9)

    def test_boundary_probabilities_are_valid(self) -> None:
        for p1, p2 in ((0.0, 0.2), (0.5, 1.0)):
            power = power_proportion_test(p1, p2, 30)
            assert 0.0 <= power <= 1.0

    def test_power_strictly_increases_with_n(self) -> None:
        powers = [power_proportion_test(0.1, 0.3, n) for n in (10, 25, 50, 200)]
        assert powers[0] < powers[1] < powers[2] < powers[3]

    def test_positive_direction_one_sided_beats_two_sided(self) -> None:
        """The one-tailed test targets p1 > p2; aligned direction concentrates
        all rejection mass in the single upper tail."""
        two = power_proportion_test(0.35, 0.2, 40)
        one = power_proportion_test(0.35, 0.2, 40, two_sided=False)
        assert one >= two

    def test_very_high_alpha_one_sided_expands_left_bracket(self) -> None:
        """Level 0.9 pushes the z critical value negative (left bracket
        expansion inside the shared inverter); power stays >= level."""
        power = power_proportion_test(0.4, 0.3, 25, alpha=0.9, two_sided=False)
        assert 0.9 <= power <= 1.0


class TestPowerProportionValidation:
    def test_p1_below_range_raises(self) -> None:
        with pytest.raises(ValueError, match="p1"):
            power_proportion_test(-0.1, 0.5, 20)

    def test_p2_above_range_raises(self) -> None:
        with pytest.raises(ValueError, match="p2"):
            power_proportion_test(0.5, 1.5, 20)

    @pytest.mark.parametrize("bad_n", [0, 1])
    def test_tiny_n_raises(self, bad_n: int) -> None:
        with pytest.raises(ValueError, match="n_per_group"):
            power_proportion_test(0.2, 0.4, bad_n)

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.5])
    def test_bad_alpha_raises(self, bad_alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            power_proportion_test(0.2, 0.4, 20, alpha=bad_alpha)
