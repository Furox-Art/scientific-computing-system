import pytest

from cds.stats import (
    bonferroni_corrected_alpha,
    chi2_sf,
    chi_square_gof,
    chi_square_independence,
    cohens_d,
    cramers_v,
    eta_squared_from_f,
    f_sf,
    one_sample_ttest,
    one_way_anova,
    t_sf,
    two_sample_ttest,
)


class TestDistributionTails:
    def test_t_sf_known_value(self) -> None:
        # P(|T| >= 2) for t(10) ~ 0.07339
        assert abs(t_sf(2.0, 10) - 0.07339) < 1e-4

    def test_t_sf_zero_is_one(self) -> None:
        assert abs(t_sf(0.0, 5) - 1.0) < 1e-9

    def test_t_sf_symmetric(self) -> None:
        assert abs(t_sf(1.5, 8) - t_sf(-1.5, 8)) < 1e-12

    def test_chi2_sf_critical_value(self) -> None:
        # chi2 0.95 quantile for df=1 is ~3.841
        assert abs(chi2_sf(3.841, 1) - 0.05) < 1e-3

    def test_chi2_sf_df5(self) -> None:
        assert abs(chi2_sf(11.0705, 5) - 0.05) < 1e-3

    def test_chi2_sf_zero_is_one(self) -> None:
        assert chi2_sf(0.0, 3) == 1.0

    def test_f_sf_known_value(self) -> None:
        # F=4.0, df1=2, df2=12 -> ~0.0466
        assert abs(f_sf(4.0, 2, 12) - 0.04663) < 1e-3

    def test_f_sf_zero_is_one(self) -> None:
        assert f_sf(0.0, 3, 10) == 1.0


class TestOneSampleTTest:
    def test_zero_difference(self) -> None:
        # sample mean equals popmean -> t == 0, p == 1
        r = one_sample_ttest([4.9, 5.1, 5.0, 5.2, 4.8], 5.0)
        assert abs(r.statistic) < 1e-9
        assert abs(r.p_value - 1.0) < 1e-9

    def test_zero_variance_raises(self) -> None:
        with pytest.raises(ValueError):
            one_sample_ttest([5.0, 5.0, 5.0, 5.0], 5.0)

    def test_significant_shift(self) -> None:
        r = one_sample_ttest([10.0, 11.0, 9.0, 10.5, 9.5], 5.0)
        assert r.statistic > 0
        assert r.p_value < 0.01
        assert r.df == 4

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError):
            one_sample_ttest([1.0], 0.0)


class TestTwoSampleTTest:
    def test_pooled_known(self) -> None:
        # a=[1..5], b=[2,4,6,8,10] -> t=-1.897, df=8
        r = two_sample_ttest([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert abs(r.statistic - (-1.8974)) < 1e-3
        assert r.df == 8
        assert abs(r.p_value - 0.0943) < 1e-3

    def test_welch_df_noninteger(self) -> None:
        r = two_sample_ttest(
            [1, 2, 3, 4, 5],
            [10, 20, 30],
            equal_var=False,
        )
        assert r.df != int(r.df) or r.df < 6

    def test_identical_groups_p_one(self) -> None:
        r = two_sample_ttest([1, 2, 3, 4], [1, 2, 3, 4])
        assert abs(r.statistic) < 1e-9
        assert abs(r.p_value - 1.0) < 1e-9

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError):
            two_sample_ttest([1.0], [1.0, 2.0])


class TestChiSquare:
    def test_gof_known(self) -> None:
        r = chi_square_gof(
            [16, 18, 16, 14, 12, 12],
            [16, 16, 16, 16, 16, 8],
        )
        assert abs(r.statistic - 3.5) < 1e-9
        assert r.df == 5
        assert abs(r.p_value - 0.6234) < 1e-3

    def test_gof_perfect_fit(self) -> None:
        r = chi_square_gof([10, 20, 30], [10, 20, 30])
        assert r.statistic == 0.0
        assert abs(r.p_value - 1.0) < 1e-9

    def test_gof_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_gof([1, 2], [1, 2, 3])

    def test_independence_table(self) -> None:
        # classic 2x2 table
        table: list[list[float]] = [[10.0, 20.0], [30.0, 40.0]]
        r = chi_square_independence(table)
        assert r.df == 1
        assert r.statistic >= 0
        assert 0.0 <= r.p_value <= 1.0

    def test_independence_independent_data(self) -> None:
        # rows proportional -> chi2 ~ 0
        table: list[list[float]] = [[10.0, 20.0], [20.0, 40.0]]
        r = chi_square_independence(table)
        assert r.statistic < 1e-9

    def test_independence_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            chi_square_independence([[1.0, 2.0]])

    def test_independence_zero_row_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive marginal"):
            chi_square_independence([[10.0, 20.0], [0.0, 0.0]])

    def test_independence_zero_col_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive marginal"):
            chi_square_independence([[10.0, 0.0], [20.0, 0.0]])


class TestANOVA:
    def test_known_f(self) -> None:
        r = one_way_anova([1, 2, 3], [2, 3, 4], [4, 5, 6])
        assert abs(r.statistic - 7.0) < 1e-9
        assert abs(r.p_value - 0.02702) < 1e-3

    def test_identical_groups(self) -> None:
        r = one_way_anova([1, 2, 3], [1, 2, 3])
        assert abs(r.statistic) < 1e-9

    def test_too_few_groups_raises(self) -> None:
        with pytest.raises(ValueError):
            one_way_anova([1, 2, 3])


class TestCohensD:
    def test_known_value(self) -> None:
        # a=[1..5] (mean 3), b=[6..10] (mean 8), both var 2.5 -> sp=sqrt(2.5),
        # d = (3-8)/1.5811 ~= -3.1623
        d = cohens_d([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert abs(d - (-3.1623)) < 1e-3

    def test_identical_groups_zero(self) -> None:
        # Same values -> mean difference 0 -> d == 0
        d = cohens_d([1, 2, 3, 4], [1, 2, 3, 4])
        assert abs(d) < 1e-9

    def test_sign_reflects_direction(self) -> None:
        # Swapping the groups flips the sign of d
        d1 = cohens_d([1, 2, 3], [4, 5, 6])
        d2 = cohens_d([4, 5, 6], [1, 2, 3])
        assert d1 == -d2
        assert d1 < 0  # group_a smaller than group_b
        assert d2 > 0

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError):
            cohens_d([1.0], [1, 2, 3])

    def test_zero_variance_raises(self) -> None:
        # Both groups constant -> pooled variance 0 -> undefined
        with pytest.raises(ValueError):
            cohens_d([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])


class TestEtaSquared:
    def test_known_value(self) -> None:
        # F=7.0, df1=2, df2=6 -> eta2 = (7*2)/(7*2+6) = 14/20 = 0.7
        eta2 = eta_squared_from_f(7.0, 2, 6)
        assert abs(eta2 - 0.7) < 1e-9

    def test_zero_f_zero_eta(self) -> None:
        # F=0 -> no variance explained
        assert eta_squared_from_f(0.0, 2, 6) == 0.0

    def test_range_in_unit_interval(self) -> None:
        # Large F -> eta2 approaches 1 but never reaches it
        eta2 = eta_squared_from_f(1000.0, 2, 6)
        assert 0.0 < eta2 < 1.0

    def test_negative_f_raises(self) -> None:
        with pytest.raises(ValueError):
            eta_squared_from_f(-1.0, 2, 6)

    def test_invalid_df_raises(self) -> None:
        with pytest.raises(ValueError):
            eta_squared_from_f(7.0, 0, 6)
        with pytest.raises(ValueError):
            eta_squared_from_f(7.0, 2, 0)


class TestCramersV:
    def test_perfect_association(self) -> None:
        # Off-diagonal dominance: strong association, V near 1
        table: list[list[float]] = [[100.0, 0.0], [0.0, 100.0]]
        v = cramers_v(table)
        assert abs(v - 1.0) < 1e-9

    def test_independent_table_zero(self) -> None:
        # Proportional rows -> chi2 ~ 0 -> V ~ 0
        table = [[10.0, 20.0], [20.0, 40.0]]
        v = cramers_v(table)
        assert v < 1e-9

    def test_range_in_unit_interval(self) -> None:
        table = [[10.0, 20.0], [30.0, 40.0]]
        v = cramers_v(table)
        assert 0.0 <= v <= 1.0

    def test_too_small_table_raises(self) -> None:
        with pytest.raises(ValueError):
            cramers_v([[1.0, 2.0]])

    def test_too_few_columns_raises(self) -> None:
        # 2 rows but only 1 column -> cols < 2 branch (line 507).
        with pytest.raises(ValueError):
            cramers_v([[1.0], [2.0]])

    def test_non_rectangular_table_raises(self) -> None:
        # Ragged rows -> any(len(r) != cols) branch (line 507).
        with pytest.raises(ValueError):
            cramers_v([[1.0, 2.0], [3.0]])

    def test_zero_total_raises(self) -> None:
        with pytest.raises(ValueError):
            cramers_v([[0.0, 0.0], [0.0, 0.0]])


class TestBonferroni:
    def test_known_correction(self) -> None:
        # alpha=0.05, k=5 -> 0.01
        assert abs(bonferroni_corrected_alpha(0.05, 5) - 0.01) < 1e-12

    def test_k_one_is_noop(self) -> None:
        # Single comparison -> corrected alpha equals original
        assert bonferroni_corrected_alpha(0.05, 1) == 0.05

    def test_k_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            bonferroni_corrected_alpha(0.05, 0)

    def test_k_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            bonferroni_corrected_alpha(0.05, -3)

    def test_alpha_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            bonferroni_corrected_alpha(0.0, 5)
        with pytest.raises(ValueError):
            bonferroni_corrected_alpha(1.0, 5)

    def test_strict_monotone_in_k(self) -> None:
        # More comparisons -> stricter (smaller) corrected alpha
        a1 = bonferroni_corrected_alpha(0.05, 2)
        a2 = bonferroni_corrected_alpha(0.05, 10)
        assert a2 < a1
