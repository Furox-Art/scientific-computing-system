"""C-2: log-space Poisson and binomial PMFs must not overflow or underflow.

The audited crash inputs and their immediate neighbours are all pinned to
values computed independently in log space, so a regression that reintroduces
either failure mode fails on a number rather than on an exception.
"""

from __future__ import annotations

import math

import pytest

from cds.probability.distributions import binomial_pmf, poisson_pmf


def _log_poisson(k: int, lam: float) -> float:
    """Independent oracle for ``ln P(X = k)``."""
    return k * math.log(lam) - lam - math.lgamma(k + 1.0)


def _log_binomial(k: int, n: int, p: float) -> float:
    """Independent oracle for ``ln P(X = k)``."""
    return (
        math.lgamma(n + 1.0)
        - math.lgamma(k + 1.0)
        - math.lgamma(n - k + 1.0)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


class TestPoissonPmfExtremes:
    """``poisson_pmf(1000, 1000)`` used to raise OverflowError."""

    @pytest.mark.parametrize("k", [998, 999, 1000, 1001, 1002])
    def test_audited_crash_inputs(self, k: int) -> None:
        """Every one of these raised before: ``lam**k`` exceeds the float range.

        The true probability sits near 0.0126 — comfortably representable —
        so returning a number here is a value assertion, not a smoke test.
        """
        value = poisson_pmf(k, 1000.0)
        assert 0.0 < value < 0.05
        assert value == pytest.approx(math.exp(_log_poisson(k, 1000.0)), rel=1e-12)

    def test_integer_lambda_matches_float_lambda(self) -> None:
        """``lam=1000`` (int) and ``lam=1000.0`` must agree exactly."""
        assert poisson_pmf(1000, 1000) == poisson_pmf(1000, 1000.0)

    @pytest.mark.parametrize("lam", [1.0, 3.0, 5.0, 20.0, 100.0])
    def test_small_lambda_values_unchanged(self, lam: float) -> None:
        """Legacy values keep the documented relative accuracy."""
        for k in range(0, 40):
            assert poisson_pmf(k, lam) == pytest.approx(math.exp(_log_poisson(k, lam)), rel=1e-12)

    def test_support_edges_are_exact(self) -> None:
        """Documented integer semantics: negative k is 0, Poisson(0) is a point mass."""
        assert poisson_pmf(-1, 3.0) == 0.0
        assert poisson_pmf(0, 0.0) == 1.0
        assert poisson_pmf(1, 0.0) == 0.0

    def test_very_large_lambda_stays_finite(self) -> None:
        """``lam = 10**6``: the mode stays representable and positive."""
        value = poisson_pmf(1_000_000, 1_000_000.0)
        assert value == pytest.approx(math.exp(_log_poisson(1_000_000, 1_000_000.0)), rel=1e-9)
        assert value > 0.0
        # Two standard deviations out is still positive, not underflowed.
        assert poisson_pmf(1_000_500, 1_000_000.0) > 0.0

    def test_invalid_lambda_still_rejected(self) -> None:
        with pytest.raises(ValueError, match="lambda must be non-negative"):
            poisson_pmf(0, -1.0)

    def test_non_finite_lambda_is_propagated_not_swallowed(self) -> None:
        """A NaN log-probability must not be rounded into a probability of zero.

        The underflow shortcut compares against ``-745.0``; since
        ``nan < -745.0`` is ``False`` it would fall through to ``exp(nan)``,
        but only because the NaN branch runs first. Asserting the value pins
        that ordering -- without it, a future refactor could silently turn
        NaN input into ``0.0``, which is a valid-looking probability.
        """
        assert math.isnan(poisson_pmf(3, float("nan")))
        assert math.isnan(poisson_pmf(3, float("inf")))


class TestBinomialPmfExtremes:
    """``binomial_pmf(1000, 2000, 0.5)`` used to raise OverflowError."""

    def test_audited_crash_input(self) -> None:
        """``C(2000, 1000)`` is far beyond the float range; the pmf is not."""
        value = binomial_pmf(1000, 2000, 0.5)
        assert value == pytest.approx(math.exp(_log_binomial(1000, 2000, 0.5)), rel=1e-12)
        assert 0.01 < value < 0.05

    @pytest.mark.parametrize(
        ("k", "n", "p"),
        [(1500, 2000, 0.5), (1999, 2000, 0.5), (0, 2000, 0.5), (1, 2000, 0.5)],
    )
    def test_neighbours_of_the_crash_point(self, k: int, n: int, p: float) -> None:
        """Support extremes for the same ``n`` must all be finite and correct."""
        value = binomial_pmf(k, n, p)
        assert math.isfinite(value)
        assert value == pytest.approx(math.exp(_log_binomial(k, n, p)), rel=1e-12)

    def test_subnormal_result_is_not_silently_zero(self) -> None:
        """The tail case where the direct product underflows to 0.0.

        For ``n = 1000, k = 500, p = 0.065`` the factor ``p**k`` is a 2.5e-594
        double, so the old ``comb * p**k * (1-p)**(n-k)`` collapsed to exactly
        ``0.0`` while the true probability is the representable subnormal
        1.97e-309. The bug was silence, not an exception, so the assertion is
        on the recovered magnitude.
        """
        n, k, p = 1000, 500, 0.065
        legacy_direct = math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))
        assert legacy_direct == 0.0

        value = binomial_pmf(k, n, p)
        assert value > 0.0
        assert value == pytest.approx(math.exp(_log_binomial(k, n, p)), rel=1e-12)
        assert value == pytest.approx(1.97e-309, rel=1e-3)

    @pytest.mark.parametrize("n", [10, 50, 200])
    def test_mass_sums_to_one(self, n: int) -> None:
        """The log-space form must stay normalised at every scale."""
        for p in (0.1, 0.25, 0.5, 0.75, 0.9):
            total = sum(binomial_pmf(k, n, p) for k in range(n + 1))
            assert total == pytest.approx(1.0, abs=1e-9)

    def test_degenerate_endpoints_stay_exact(self) -> None:
        """``p = 0`` / ``p = 1`` keep their documented integer semantics."""
        assert binomial_pmf(0, 5, 0.0) == 1.0
        assert binomial_pmf(1, 5, 0.0) == 0.0
        assert binomial_pmf(5, 5, 1.0) == 1.0
        assert binomial_pmf(3, 5, 1.0) == 0.0

    def test_out_of_support_and_invalid_p(self) -> None:
        assert binomial_pmf(6, 5, 0.5) == 0.0
        assert binomial_pmf(-1, 5, 0.5) == 0.0
        with pytest.raises(ValueError, match="p must be in"):
            binomial_pmf(1, 5, 1.5)

    def test_small_case_matches_closed_form(self) -> None:
        """Legacy small-n accuracy is preserved (was 0.3125 exactly)."""
        assert binomial_pmf(3, 5, 0.5) == pytest.approx(math.comb(5, 3) * 0.5**5, abs=1e-15)
        assert binomial_pmf(3, 10, 0.5) == pytest.approx(0.1171875, rel=1e-12)
