"""Tests for :mod:`cds.probability._advanced` (chi2 / t / gamma / beta)."""

from __future__ import annotations

import math

import pytest

from cds.probability import (
    beta_pdf,
    chi2_cdf,
    chi2_pdf,
    chi2_ppf,
    gamma_pdf,
    sample_beta,
    sample_gamma,
    t_cdf,
    t_pdf,
    t_ppf,
)

# --------------------------------------------------------------------- #
# Chi-square                                                             #
# --------------------------------------------------------------------- #


def test_chi2_ppf_matches_known_critical_value() -> None:
    assert chi2_ppf(0.95, 1) == pytest.approx(3.841458820694124, abs=1e-6)
    assert chi2_ppf(0.95, 10) == pytest.approx(18.307038053275146, abs=1e-5)


def test_chi2_cdf_round_trip() -> None:
    assert chi2_cdf(3.841458820694124, 1) == pytest.approx(0.95, abs=1e-6)
    assert chi2_cdf(0.0, 5) == 0.0
    assert chi2_cdf(-1.0, 5) == 0.0
    assert chi2_cdf(1e6, 5) == pytest.approx(1.0)


def test_chi2_pdf_values_and_edges() -> None:
    # df=2 gives the memoryless exponential-like density x/2 * exp(-x/2)... in
    # fact pdf(0)=0.5 exactly for df=2.
    assert chi2_pdf(0.0, 2) == 0.5
    assert chi2_pdf(0.0, 1) == math.inf
    assert chi2_pdf(0.0, 5) == 0.0
    assert chi2_pdf(-1.0, 3) == 0.0
    assert chi2_pdf(1.0, 1) == pytest.approx(0.24197072451914337)


def test_chi2_pdf_integrates_to_one() -> None:
    df = 4
    n = 20000
    grid = [i * 60.0 / n for i in range(1, n + 1)]
    mass = sum(chi2_pdf(x, df) for x in grid) * (grid[1] - grid[0])
    assert mass == pytest.approx(1.0, abs=1e-3)


def test_chi2_validation() -> None:
    with pytest.raises(ValueError, match="df must be positive"):
        chi2_pdf(1.0, 0)
    with pytest.raises(ValueError, match="df must be positive"):
        chi2_cdf(1.0, -1)
    with pytest.raises(ValueError, match="df must be positive"):
        chi2_ppf(0.5, -1)
    with pytest.raises(ValueError, match=r"p must be in \(0, 1\)"):
        chi2_ppf(0.0, 1)
    with pytest.raises(ValueError, match=r"p must be in \(0, 1\)"):
        chi2_ppf(1.0, 1)


# --------------------------------------------------------------------- #
# Student's t                                                            #
# --------------------------------------------------------------------- #


def test_t_cdf_matches_known_values() -> None:
    assert t_cdf(1.8124611228107335, 10) == pytest.approx(0.95, abs=1e-6)
    assert t_cdf(0.0, 7) == 0.5
    assert t_cdf(-1.8124611228107335, 10) == pytest.approx(0.05, abs=1e-6)


def test_t_ppf_matches_known_critical_values() -> None:
    assert t_ppf(0.975, 10) == pytest.approx(2.228138851986273, abs=1e-6)
    assert t_ppf(0.025, 10) == pytest.approx(-2.228138851986273, abs=1e-6)


def test_t_pdf_symmetry_and_normalization() -> None:
    assert t_pdf(1.0, 5) == t_pdf(-1.0, 5)
    n = 40000
    grid = [(i - n / 2) * 80.0 / n for i in range(n)]
    mass = sum(t_pdf(x, 3) for x in grid) * (80.0 / n)
    assert mass == pytest.approx(1.0, abs=1e-3)


def test_t_converges_to_normal_for_large_df() -> None:
    assert t_pdf(0.0, 5000) == pytest.approx(1.0 / math.sqrt(2 * math.pi), rel=1e-3)


def test_t_validation() -> None:
    with pytest.raises(ValueError, match="df must be positive"):
        t_pdf(1.0, 0)
    with pytest.raises(ValueError, match="df must be positive"):
        t_cdf(1.0, -3)
    with pytest.raises(ValueError, match="df must be positive"):
        t_ppf(0.5, 0)
    with pytest.raises(ValueError, match=r"p must be in \(0, 1\)"):
        t_ppf(1.5, 5)


# --------------------------------------------------------------------- #
# Gamma                                                                  #
# --------------------------------------------------------------------- #


def test_gamma_pdf_values_and_edges() -> None:
    assert gamma_pdf(0.0, 1, 2.0) == 0.5
    assert gamma_pdf(0.0, 0.5) == math.inf
    assert gamma_pdf(0.0, 3) == 0.0
    assert gamma_pdf(-1.0, 2) == 0.0
    # Exponential is Gamma(shape=1): pdf(x) = exp(-x)/scale.
    assert gamma_pdf(2.0, 1, 1.0) == pytest.approx(math.exp(-2.0))
    assert gamma_pdf(1.0, 2, 2.0) == pytest.approx(1.0 * math.exp(-0.5) / (math.gamma(2) * 4.0))


def test_gamma_pdf_integrates_to_one() -> None:
    n = 40000
    grid = [(i + 1) * 80.0 / n for i in range(n)]
    mass = sum(gamma_pdf(x, 2.5, 1.5) for x in grid) * (80.0 / n)
    assert mass == pytest.approx(1.0, abs=1e-3)


def test_sample_gamma_moments() -> None:
    draws = sample_gamma(200_000, shape=3.0, scale=2.0, seed=42)
    mean = sum(draws) / len(draws)
    var = sum((g - mean) ** 2 for g in draws) / len(draws)
    assert mean == pytest.approx(6.0, abs=0.05)
    assert var == pytest.approx(12.0, abs=0.5)


def test_sample_gamma_shape_below_one_boost() -> None:
    draws = sample_gamma(50_000, shape=0.5, scale=1.0, seed=7)
    mean = sum(draws) / len(draws)
    assert mean == pytest.approx(0.5, abs=0.02)


def test_sample_gamma_deterministic_given_seed() -> None:
    assert sample_gamma(16, 2.0, seed=11) == sample_gamma(16, 2.0, seed=11)


def test_sample_gamma_validation() -> None:
    with pytest.raises(ValueError, match="n must be non-negative"):
        sample_gamma(-1, 1.0)
    with pytest.raises(ValueError, match="shape must be positive"):
        sample_gamma(10, 0.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        sample_gamma(10, 1.0, scale=0.0)


# --------------------------------------------------------------------- #
# Beta                                                                   #
# --------------------------------------------------------------------- #


def test_beta_pdf_values_and_clamps() -> None:
    assert beta_pdf(0.5, 2, 2) == pytest.approx(1.5)
    assert beta_pdf(-0.1, 2, 5) == 0.0
    assert beta_pdf(0.0, 2, 5) == 0.0
    assert beta_pdf(1.0, 2, 5) == 0.0
    assert beta_pdf(1.1, 2, 5) == 0.0
    assert beta_pdf(0.25, 2, 5) == pytest.approx(30 * 0.25 * (0.75) ** 4)


def test_beta_pdf_integrates_to_one() -> None:
    n = 20000
    grid = [i / n for i in range(1, n)]
    mass = sum(beta_pdf(x, 2, 5) for x in grid) / n
    assert mass == pytest.approx(1.0, abs=1e-3)


def test_sample_beta_moments() -> None:
    draws = sample_beta(100_000, 2, 5, seed=99)
    mean = sum(draws) / len(draws)
    assert mean == pytest.approx(2.0 / 7.0, abs=0.01)
    assert all(0.0 <= d <= 1.0 for d in draws[:1000])


def test_sample_beta_symmetric_case_has_spread() -> None:
    draws = sample_beta(20_000, 3, 3, seed=5)
    mean = sum(draws) / len(draws)
    var = sum((d - mean) ** 2 for d in draws) / len(draws)
    assert mean == pytest.approx(0.5, abs=0.005)
    assert 0.001 < var < 0.05


def test_sample_beta_validation() -> None:
    with pytest.raises(ValueError, match="n must be non-negative"):
        sample_beta(-1, 1, 1)
    with pytest.raises(ValueError, match="a must be positive"):
        sample_beta(10, 0, 1)
    with pytest.raises(ValueError, match="b must be positive"):
        sample_beta(10, 1, -1)


def test_pdf_validations_direct() -> None:
    with pytest.raises(ValueError, match="shape must be positive"):
        gamma_pdf(1.0, 0.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        gamma_pdf(1.0, 2.0, scale=-1.0)
    with pytest.raises(ValueError, match="a must be positive"):
        beta_pdf(0.5, 0.0, 1.0)
    with pytest.raises(ValueError, match="b must be positive"):
        beta_pdf(0.5, 1.0, -2.0)


def test_sample_beta_uses_boost_for_small_shape() -> None:
    draws = sample_beta(2000, 0.5, 2.0, seed=13)
    mean = sum(draws) / len(draws)
    assert 0.0 < mean < 1.0
    # Beta(0.5, 2) concentrates below 0.5: mean = a/(a+b) = 0.2.
    assert mean == pytest.approx(0.2, abs=0.05)


def test_betacf_reexport_still_importable() -> None:
    # Guard the cross-module contract _advanced relies on.
    from cds.stats._distributions import _betai as betai

    assert betai(0.5, 0.5, 0.5) == pytest.approx(0.5)
