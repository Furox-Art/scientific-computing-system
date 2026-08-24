"""Tests for Markov chain Monte Carlo (Metropolis-Hastings)."""

import math

import pytest

from cds.montecarlo.markov import MHResult, metropolis_hastings


def _normal_log_pdf(x: float) -> float:
    """Log-density of the standard normal distribution."""
    return -0.5 * x * x


def _half_normal_log_pdf(x: float) -> float:
    """Log-density of N(0,1) truncated to x >= 0 (-inf elsewhere)."""
    return -0.5 * x * x if x >= 0.0 else -math.inf


class TestValidation:
    def test_n_samples_below_one(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            metropolis_hastings(_normal_log_pdf, 0.0, n_samples=0)

    def test_negative_burn_in(self) -> None:
        with pytest.raises(ValueError, match="burn_in"):
            metropolis_hastings(_normal_log_pdf, 0.0, burn_in=-1)

    def test_thin_below_one(self) -> None:
        with pytest.raises(ValueError, match="thin"):
            metropolis_hastings(_normal_log_pdf, 0.0, thin=0)

    def test_nonpositive_proposal_scale(self) -> None:
        with pytest.raises(ValueError, match="proposal_scale"):
            metropolis_hastings(_normal_log_pdf, 0.0, proposal_scale=0.0)


class TestSamplerBasics:
    def test_result_type_and_sample_count(self) -> None:
        res = metropolis_hastings(_normal_log_pdf, 0.0, n_samples=50, burn_in=10, thin=2, seed=1)
        assert isinstance(res, MHResult)
        assert len(res.samples) == 50

    def test_reproducible_with_seed(self) -> None:
        a = metropolis_hastings(_normal_log_pdf, 0.0, n_samples=100, seed=7)
        b = metropolis_hastings(_normal_log_pdf, 0.0, n_samples=100, seed=7)
        assert a.samples == b.samples
        assert a.acceptance_rate == b.acceptance_rate

    def test_acceptance_rate_within_bounds(self) -> None:
        res = metropolis_hastings(_normal_log_pdf, 0.0, n_samples=300, proposal_scale=4.0, seed=3)
        assert 0.0 < res.acceptance_rate < 1.0

    def test_high_acceptance_for_flat_target(self) -> None:
        res = metropolis_hastings(lambda x: 0.0, 0.0, n_samples=200, seed=5)
        assert res.acceptance_rate > 0.95

    def test_burn_in_zero_default_thin(self) -> None:
        res = metropolis_hastings(_normal_log_pdf, 0.0, n_samples=25, burn_in=0, seed=11)
        assert len(res.samples) == 25

    def test_thin_three_burn_in_zero(self) -> None:
        res = metropolis_hastings(_normal_log_pdf, 2.0, n_samples=10, burn_in=0, thin=3, seed=11)
        assert len(res.samples) == 10

    def test_starting_point_moves_after_burn_in(self) -> None:
        res = metropolis_hastings(_normal_log_pdf, 100.0, n_samples=500, seed=13)
        assert max(res.samples) < 100.0


class TestInfiniteDensityHandling:
    def test_zero_density_proposals_always_rejected(self) -> None:
        res = metropolis_hastings(_half_normal_log_pdf, 1.0, n_samples=400, burn_in=0, seed=21)
        assert all(s >= 0.0 for s in res.samples)
        assert res.acceptance_rate < 1.0

    def test_inf_current_state_accepts_finite_proposal(self) -> None:
        res = metropolis_hastings(_half_normal_log_pdf, -1.0, n_samples=300, burn_in=150, seed=42)
        assert all(s >= 0.0 for s in res.samples)
        assert res.acceptance_rate > 0.0

    def test_inf_current_escapes_quickly(self) -> None:
        res = metropolis_hastings(
            _half_normal_log_pdf,
            -2.0,
            n_samples=200,
            burn_in=100,
            proposal_scale=2.0,
            seed=9,
        )
        assert len(res.samples) == 200
        assert res.samples[-1] >= 0.0
        assert sum(res.samples) > 0.0


class TestStatistics:
    def test_standard_normal_mean_and_std(self) -> None:
        res = metropolis_hastings(
            _normal_log_pdf,
            0.0,
            n_samples=20_000,
            burn_in=1_000,
            thin=1,
            proposal_scale=1.0,
            seed=2024,
        )
        mean = sum(res.samples) / len(res.samples)
        var = sum((s - mean) ** 2 for s in res.samples) / len(res.samples)
        std = math.sqrt(var)
        assert abs(mean) < 0.15
        assert 0.8 < std < 1.2

    def test_half_normal_mean_near_expected(self) -> None:
        res = metropolis_hastings(_half_normal_log_pdf, 1.0, n_samples=8_000, burn_in=500, seed=77)
        mean = sum(res.samples) / len(res.samples)
        assert 0.65 < mean < 0.95
