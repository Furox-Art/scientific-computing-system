"""Tests for :mod:`cds.bayes.conjugate` (v1 pure-Python port)."""

from __future__ import annotations

import pytest

from cds.bayes.conjugate import (
    bayes_factor,
    beta_binomial_update,
    beta_credible_interval,
    credible_interval,
    gamma_credible_interval,
    gamma_poisson_update,
    normal_credible_interval,
    normal_normal_update,
)


def test_beta_binomial_posterior_params_and_mean() -> None:
    res = beta_binomial_update(1.0, 1.0, 5, 10)
    assert res["alpha_post"] == pytest.approx(6.0)
    assert res["beta_post"] == pytest.approx(6.0)
    assert res["alpha"] == pytest.approx(6.0)
    assert res["beta"] == pytest.approx(6.0)
    assert res["mean"] == pytest.approx(0.5)
    assert abs(res["variance"] - (6 * 6) / (12 * 12 * 13)) < 1e-9


def test_beta_binomial_credible_interval_contains_mean() -> None:
    res = beta_binomial_update(2.0, 2.0, 7, 10, credible_level=0.95)
    assert res["ci_lower"] < res["mean"] < res["ci_upper"]
    assert 0.0 < res["ci_lower"] < res["ci_upper"] < 1.0
    assert res["credible_level"] == pytest.approx(0.95)
    # interval width should be < 1
    assert res["ci_upper"] - res["ci_lower"] > 0.0
    # direct helper agrees
    lo, hi = beta_credible_interval(9.0, 5.0, level=0.95)
    assert lo == pytest.approx(res["ci_lower"])
    assert hi == pytest.approx(res["ci_upper"])


def test_gamma_poisson_posterior_and_interval() -> None:
    res = gamma_poisson_update(2.0, 1.0, 5, exposure=1.0)
    assert res["alpha_post"] == pytest.approx(7.0)
    assert res["beta_post"] == pytest.approx(2.0)
    assert res["mean"] == pytest.approx(3.5)
    assert res["variance"] == pytest.approx(7.0 / 4.0)
    assert res["ci_lower"] < res["mean"] < res["ci_upper"]
    assert res["ci_lower"] > 0.0


def test_gamma_poisson_exposure_scales_rate() -> None:
    r1 = gamma_poisson_update(2.0, 1.0, 10, exposure=2.0)
    r2 = gamma_poisson_update(2.0, 1.0, 10, exposure=5.0)
    # larger exposure -> larger posterior rate -> smaller mean
    assert r2["mean"] < r1["mean"]
    # variance also shrinks with larger exposure
    assert r2["variance"] < r1["variance"]


def test_normal_normal_update_shrinkage_toward_prior() -> None:
    data = [1.0, 2.0, 3.0]
    # prior mu0=0, sigma0=10 very diffuse -> posterior near sample mean 2.0
    res_diffuse = normal_normal_update(0.0, 10.0, data, sigma=1.0)
    assert abs(res_diffuse["mu_post"] - 2.0) < 0.2
    assert res_diffuse["mean"] == pytest.approx(res_diffuse["mu_post"])
    # prior mu0=0, sigma0=0.1 very tight -> posterior stays near 0
    res_tight = normal_normal_update(0.0, 0.1, data, sigma=1.0)
    assert abs(res_tight["mu_post"] - 0.0) < 0.2
    # credible interval check
    assert res_diffuse["ci_lower"] < res_diffuse["mu_post"] < res_diffuse["ci_upper"]
    assert res_diffuse["variance"] == pytest.approx(res_diffuse["sigma_post"] ** 2)


def test_normal_normal_empty_data_returns_prior() -> None:
    res = normal_normal_update(5.0, 2.0, [], sigma=1.0)
    assert res["mu_post"] == pytest.approx(5.0)
    assert res["sigma_post"] == pytest.approx(2.0)
    assert res["mean"] == pytest.approx(5.0)
    assert res["n"] == pytest.approx(0.0)


def test_bayes_factor_uniform_prior_near_one_at_half() -> None:
    # With k=5,n=10,p0=0.5 under uniform Beta(1,1), BF10 = Beta(6,6)/0.5^10 ≈ 0.37
    bf = bayes_factor(5, 10, 0.5)
    assert bf == pytest.approx(0.367, rel=0.02)
    # BF01 = 1/BF10 > 1 (data slightly favour H0)
    assert 1.0 / bf > 1.0
    # With strong mismatch p0=0.1, BF10 should be << 1? Actually if p0 far, BF favours H1
    bf_far = bayes_factor(5, 10, 0.1)
    assert bf_far > 10.0


def test_bayes_factor_extreme_data() -> None:
    # All successes, p0=0.5 -> H1 with uniform should get BF >1
    bf = bayes_factor(10, 10, 0.5, alpha=1.0, beta=1.0)
    assert bf > 1.0
    # Same logic for all failures
    bf2 = bayes_factor(0, 10, 0.5)
    assert bf2 > 1.0


def test_credible_interval_helpers() -> None:
    lo, hi = credible_interval(2.0, 2.0, level=0.95)
    lo2, hi2 = beta_credible_interval(2.0, 2.0, level=0.95)
    assert lo == pytest.approx(lo2)
    assert hi == pytest.approx(hi2)
    # Gamma interval
    glo, ghi = gamma_credible_interval(7.0, 2.0, level=0.95)
    assert glo < 3.5 < ghi
    assert glo > 0
    # Normal interval symmetric
    nlo, nhi = normal_credible_interval(0.0, 1.0, level=0.95)
    assert abs(nlo + nhi) < 1e-9
    assert abs(nhi - 1.96) < 0.02  # 95% normal approx 1.96


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        beta_binomial_update(0.0, 1.0, 1, 5)
    with pytest.raises(ValueError):
        beta_binomial_update(1.0, 1.0, 6, 5)
    with pytest.raises(ValueError):
        beta_binomial_update(1.0, 1.0, 1, -1)
    with pytest.raises(ValueError):
        beta_binomial_update(1.0, 1.0, 1, 5, credible_level=1.5)
    with pytest.raises(ValueError):
        gamma_poisson_update(1.0, 1.0, -1)
    with pytest.raises(ValueError):
        gamma_poisson_update(1.0, 0.0, 1)
    with pytest.raises(ValueError):
        gamma_poisson_update(0.0, 1.0, 1)
    with pytest.raises(ValueError):
        gamma_poisson_update(1.0, 1.0, 1, exposure=0.0)
    with pytest.raises(ValueError):
        gamma_poisson_update(1.0, 1.0, 1, exposure=1.0, credible_level=0.0)
    with pytest.raises(ValueError):
        normal_normal_update(0.0, 0.0, [1.0], sigma=1.0)
    with pytest.raises(ValueError):
        normal_normal_update(0.0, 1.0, [1.0], sigma=-1.0)
    with pytest.raises(ValueError):
        normal_normal_update(0.0, 1.0, [1.0], sigma=1.0, credible_level=2.0)
    with pytest.raises(ValueError):
        beta_credible_interval(0.0, 1.0, level=0.95)
    with pytest.raises(ValueError):
        beta_credible_interval(1.0, 0.0, level=0.95)
    with pytest.raises(ValueError):
        beta_credible_interval(1.0, 1.0, level=1.5)
    with pytest.raises(ValueError):
        beta_credible_interval(1.0, 1.0, level=0.0)
    with pytest.raises(ValueError):
        gamma_credible_interval(0.0, 1.0, level=0.95)
    with pytest.raises(ValueError):
        gamma_credible_interval(1.0, 0.0, level=0.95)
    with pytest.raises(ValueError):
        gamma_credible_interval(1.0, 1.0, level=0.0)
    with pytest.raises(ValueError):
        gamma_credible_interval(1.0, 1.0, level=1.0)
    with pytest.raises(ValueError):
        normal_credible_interval(0.0, -1.0)
    with pytest.raises(ValueError):
        normal_credible_interval(0.0, 0.0, level=0.95)
    with pytest.raises(ValueError):
        normal_credible_interval(0.0, 1.0, level=1.5)
    with pytest.raises(ValueError):
        normal_credible_interval(0.0, 1.0, level=0.0)
    with pytest.raises(ValueError):
        bayes_factor(5, 10, 1.5)
    with pytest.raises(ValueError):
        bayes_factor(11, 10, 0.5)
    with pytest.raises(ValueError):
        bayes_factor(1, -1, 0.5)
    with pytest.raises(ValueError):
        bayes_factor(5, 10, -0.1)
    with pytest.raises(ValueError):
        bayes_factor(5, 10, 0.5, alpha=0.0)
    with pytest.raises(ValueError):
        bayes_factor(5, 10, 0.5, alpha=1.0, beta=0.0)


def test_bayes_factor_boundary_p0() -> None:
    # p0 == 0 with k == 0 -> finite (H0 likelihood 1, BF10 = Beta marginal)
    bf = bayes_factor(0, 5, 0.0)
    assert bf > 0 and bf != float("inf")
    # p0 == 0 with k > 0 -> H0 impossible -> BF10 inf
    assert bayes_factor(1, 5, 0.0) == float("inf")
    # p0 == 1 with k == n -> finite
    bf2 = bayes_factor(5, 5, 1.0)
    assert bf2 > 0 and bf2 != float("inf")
    # p0 == 1 with k != n -> inf
    assert bayes_factor(4, 5, 1.0) == float("inf")


def test_credible_interval_levels_and_edge_n_zero() -> None:
    # level variation changes width
    lo95, hi95 = beta_credible_interval(5.0, 5.0, level=0.95)
    lo50, hi50 = beta_credible_interval(5.0, 5.0, level=0.5)
    assert (hi95 - lo95) > (hi50 - lo50)
    # beta_binomial with n==0 returns prior
    res = beta_binomial_update(2.0, 3.0, 0, 0)
    assert res["alpha_post"] == pytest.approx(2.0)
    assert res["beta_post"] == pytest.approx(3.0)
    assert res["mean"] == pytest.approx(2.0 / 5.0)
    # gamma with k==0
    res2 = gamma_poisson_update(2.0, 1.0, 0, exposure=1.0)
    assert res2["alpha_post"] == pytest.approx(2.0)
    assert res2["beta_post"] == pytest.approx(2.0)
    assert res2["mean"] == pytest.approx(1.0)
