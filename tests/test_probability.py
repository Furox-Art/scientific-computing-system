"""Tests for probability distributions module."""

import math

import pytest

from cds.probability.distributions import (
    binomial_pmf,
    exponential_pdf,
    gaussian_pdf,
    poisson_pmf,
    uniform_pdf,
    uniform_sample,
)

# --- Gaussian ---


def test_gaussian_peak() -> None:
    # PDF peaks at mu
    assert gaussian_pdf(0.0) > gaussian_pdf(1.0)


def test_gaussian_symmetric() -> None:
    assert abs(gaussian_pdf(-1.0) - gaussian_pdf(1.0)) < 1e-12


def test_gaussian_custom_params() -> None:
    val = gaussian_pdf(5.0, mu=5.0, sigma=2.0)
    peak = 1 / (2 * math.sqrt(2 * math.pi))
    assert abs(val - peak) < 1e-9


def test_gaussian_invalid_sigma() -> None:
    with pytest.raises(ValueError):
        gaussian_pdf(0.0, sigma=0.0)


def test_gaussian_narrow() -> None:
    # smaller sigma -> taller peak
    narrow = gaussian_pdf(0.0, sigma=0.1)
    wide = gaussian_pdf(0.0, sigma=10.0)
    assert narrow > wide


# --- Uniform ---


def test_uniform_inside() -> None:
    assert abs(uniform_pdf(0.5) - 1.0) < 1e-9


def test_uniform_outside() -> None:
    assert uniform_pdf(-1.0) == 0.0
    assert uniform_pdf(2.0) == 0.0


def test_uniform_custom_range() -> None:
    val = uniform_pdf(5.0, a=0, b=10)
    assert abs(val - 0.1) < 1e-9


def test_uniform_invalid() -> None:
    with pytest.raises(ValueError):
        uniform_pdf(0.5, a=1, b=0)


# --- Exponential ---


def test_exponential_at_zero() -> None:
    assert abs(exponential_pdf(0.0, lam=2.0) - 2.0) < 1e-9


def test_exponential_negative() -> None:
    assert exponential_pdf(-1.0) == 0.0


def test_exponential_decreasing() -> None:
    assert exponential_pdf(1.0) > exponential_pdf(2.0)


def test_exponential_invalid_lambda() -> None:
    with pytest.raises(ValueError):
        exponential_pdf(1.0, lam=-1.0)


# --- Binomial ---


def test_binomial_fair_coin() -> None:
    # P(3 heads in 5 flips) = C(5,3) * 0.5^5
    val = binomial_pmf(3, 5, 0.5)
    expected = math.comb(5, 3) * 0.5**5
    assert abs(val - expected) < 1e-12


def test_binomial_certain() -> None:
    # p=1, k=n => probability 1
    assert abs(binomial_pmf(5, 5, 1.0) - 1.0) < 1e-12


def test_binomial_impossible() -> None:
    # p=0, k>0 => probability 0
    assert binomial_pmf(1, 5, 0.0) == 0.0


def test_binomial_out_of_range() -> None:
    assert binomial_pmf(6, 5, 0.5) == 0.0
    assert binomial_pmf(-1, 5, 0.5) == 0.0


def test_binomial_invalid_p() -> None:
    with pytest.raises(ValueError):
        binomial_pmf(1, 5, 1.5)


def test_binomial_sums_to_one() -> None:
    total = sum(binomial_pmf(k, 10, 0.3) for k in range(11))
    assert abs(total - 1.0) < 1e-9


# --- Poisson ---


def test_poisson_basic() -> None:
    # P(k=0, lam=1) = e^-1
    val = poisson_pmf(0, 1.0)
    assert abs(val - math.exp(-1)) < 1e-12


def test_poisson_mode() -> None:
    # mode of Poisson(5) is near 5
    probs = [poisson_pmf(k, 5.0) for k in range(15)]
    mode = probs.index(max(probs))
    assert mode in (4, 5)


def test_poisson_negative_k() -> None:
    assert poisson_pmf(-1, 3.0) == 0.0


def test_poisson_invalid_lambda() -> None:
    with pytest.raises(ValueError):
        poisson_pmf(0, -1.0)


def test_poisson_sums_approx_one() -> None:
    total = sum(poisson_pmf(k, 3.0) for k in range(50))
    assert abs(total - 1.0) < 1e-9


# --- Sampling ---


def test_uniform_sample_count() -> None:
    samples = uniform_sample(0, 1, 100, seed=42)
    assert len(samples) == 100


def test_uniform_sample_range() -> None:
    samples = uniform_sample(2, 5, 1000, seed=7)
    assert all(2 <= s <= 5 for s in samples)


def test_uniform_sample_reproducible() -> None:
    s1 = uniform_sample(0, 1, 10, seed=99)
    s2 = uniform_sample(0, 1, 10, seed=99)
    assert s1 == s2


# --- CDF / geometric / samples ---


def test_gaussian_cdf_median() -> None:
    from cds.probability.distributions import gaussian_cdf

    assert abs(gaussian_cdf(0.0) - 0.5) < 1e-9


def test_uniform_cdf_bounds() -> None:
    from cds.probability.distributions import uniform_cdf

    assert uniform_cdf(-1.0) == 0.0
    assert uniform_cdf(2.0) == 1.0
    assert abs(uniform_cdf(0.5) - 0.5) < 1e-12


def test_exponential_cdf() -> None:
    from cds.probability.distributions import exponential_cdf

    assert exponential_cdf(-1.0) == 0.0
    assert abs(exponential_cdf(0.0) - 0.0) < 1e-12
    assert 0.0 < exponential_cdf(1.0, lam=1.0) < 1.0


def test_geometric_pmf() -> None:
    from cds.probability.distributions import geometric_pmf

    assert abs(geometric_pmf(1, 0.3) - 0.3) < 1e-12
    assert geometric_pmf(0, 0.3) == 0.0
    with pytest.raises(ValueError):
        geometric_pmf(1, 0.0)


def test_gaussian_sample_seed() -> None:
    from cds.probability.distributions import gaussian_sample
    from cds.stats import mean

    s = gaussian_sample(200, mu=5.0, sigma=1.0, seed=7961)
    assert len(s) == 200
    assert abs(mean(s) - 5.0) < 0.3


def test_exponential_sample_positive() -> None:
    from cds.probability.distributions import exponential_sample

    s = exponential_sample(50, lam=2.0, seed=1)
    assert all(v >= 0 for v in s)


def test_poisson_sample_nonneg() -> None:
    from cds.probability.distributions import poisson_sample

    s = poisson_sample(30, lam=3.0, seed=7926)
    assert all(v >= 0 for v in s)
    assert poisson_sample(3, lam=0.0, seed=0) == [0, 0, 0]


def test_sample_invalid() -> None:
    from cds.probability.distributions import (
        exponential_sample,
        gaussian_sample,
        poisson_sample,
        uniform_sample,
    )

    with pytest.raises(ValueError):
        uniform_sample(1, 0, 3)
    with pytest.raises(ValueError):
        gaussian_sample(-1)
    with pytest.raises(ValueError):
        exponential_sample(1, lam=0)
    with pytest.raises(ValueError):
        poisson_sample(1, lam=-1)


def test_cdf_invalid_params() -> None:
    from cds.probability.distributions import exponential_cdf, gaussian_cdf, uniform_cdf

    with pytest.raises(ValueError):
        gaussian_cdf(0.0, sigma=0.0)
    with pytest.raises(ValueError):
        uniform_cdf(0.5, a=1.0, b=0.0)
    with pytest.raises(ValueError):
        exponential_cdf(1.0, lam=0.0)


def test_sample_n_negative() -> None:
    from cds.probability.distributions import (
        exponential_sample,
        gaussian_sample,
        poisson_sample,
        uniform_sample,
    )

    with pytest.raises(ValueError):
        uniform_sample(0, 1, -1)
    with pytest.raises(ValueError):
        gaussian_sample(1, sigma=0)
    with pytest.raises(ValueError):
        exponential_sample(-1, lam=1)
    with pytest.raises(ValueError):
        poisson_sample(-1, lam=1)
