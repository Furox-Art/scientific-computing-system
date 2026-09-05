import math

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
    paired_cohens_d,
    paired_ttest,
    t_sf,
    two_sample_ttest,
)


class TestDistributionTails:
    def test_t_sf_known_value(self) -> None:
        assert abs(t_sf(2.0, 10) - 0.07339) < 1e-4

    def test_t_sf_zero_is_one(self) -> None:
        assert abs(t_sf(0.0, 5) - 1.0) < 1e-9

    def test_t_sf_symmetric(self) -> None:
        assert abs(t_sf(1.5, 8) - t_sf(-1.5, 8)) < 1e-12

    def test_chi2_sf_critical_value(self) -> None:
        assert abs(chi2_sf(3.841, 1) - 0.05) < 1e-3

    def test_chi2_sf_df5(self) -> None:
        assert abs(chi2_sf(11.0705, 5) - 0.05) < 1e-3

    def test_chi2_sf_zero_is_one(self) -> None:
        assert chi2_sf(0.0, 3) == 1.0

    def test_f_sf_known_value(self) -> None:
        assert abs(f_sf(4.0, 2, 12) - 0.04663) < 1e-3

    def test_f_sf_zero_is_one(self) -> None:
        assert f_sf(0.0, 3, 10) == 1.0


class TestOneSampleTTest:
    def test_zero_difference(self) -> None:
        result = one_sample_ttest([4.9, 5.1, 5.0, 5.2, 4.8], 5.0)
        assert abs(result.statistic) < 1e-9
        assert abs(result.p_value - 1.0) < 1e-9

    def test_zero_variance_raises(self) -> None:
        with pytest.raises(ValueError):
            one_sample_ttest([5.0, 5.0, 5.0, 5.0], 5.0)

    def test_significant_shift(self) -> None:
        result = one_sample_ttest([10.0, 11.0, 9.0, 10.5, 9.5], 5.0)
        assert result.statistic > 0
        assert result.p_value < 0.01
        assert result.df == 4

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError):
            one_sample_ttest([1.0], 0.0)

    def test_nonfinite_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            one_sample_ttest([1.0, math.nan], 0.0)
        with pytest.raises(ValueError, match="population mean"):
            one_sample_ttest([1.0, 2.0], math.inf)


class TestTwoSampleTTest:
    def test_pooled_known(self) -> None:
        result = two_sample_ttest([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert abs(result.statistic - (-1.8974)) < 1e-3
        assert result.df == 8
        assert abs(result.p_value - 0.0943) < 1e-3

    def test_welch_df_noninteger(self) -> None:
        result = two_sample_ttest([1, 2, 3, 4, 5], [10, 20, 30], equal_var=False)
        assert result.df != int(result.df) or result.df < 6

    def test_identical_groups_p_one(self) -> None:
        result = two_sample_ttest([1, 2, 3, 4], [1, 2, 3, 4])
        assert abs(result.statistic) < 1e-9
        assert abs(result.p_value - 1.0) < 1e-9

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError):
            two_sample_ttest([1.0], [1.0, 2.0])

    def test_nonfinite_and_welch_zero_variance_raise(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            two_sample_ttest([1.0, 2.0], [1.0, math.inf])
        with pytest.raises(ValueError, match="zero variance"):
            two_sample_ttest([1.0, 1.0], [2.0, 2.0], equal_var=False)


class TestPairedTTest:
    def test_paired_uses_differences(self) -> None:
        first = [10.0, 11.0, 9.0, 12.0, 8.0]
        second = [9.0, 10.5, 8.5, 11.0, 7.5]
        paired = paired_ttest(first, second)
        independent = two_sample_ttest(first, second)
        assert paired.df == 4
        assert paired.statistic != independent.statistic
        assert paired.p_value != independent.p_value

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            paired_ttest([1.0, 2.0], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="at least 2"):
            paired_ttest([1.0], [2.0])
        with pytest.raises(ValueError, match="finite"):
            paired_ttest([1.0, math.nan], [0.0, 0.0])

    def test_constant_differences_have_explicit_limits(self) -> None:
        identical = paired_ttest([1.0, 2.0], [1.0, 2.0])
        assert identical.statistic == 0.0
        assert identical.p_value == 1.0
        shifted = paired_ttest([1.0, 2.0], [3.0, 4.0])
        assert math.isinf(shifted.statistic)
        assert shifted.p_value == 0.0

    def test_paired_cohens_d(self) -> None:
        assert paired_cohens_d([1.0, 2.0], [1.0, 2.0]) == 0.0
        assert math.isinf(paired_cohens_d([1.0, 2.0], [3.0, 4.0]))
        with pytest.raises(ValueError, match="same length"):
            paired_cohens_d([1.0, 2.0], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="finite"):
            paired_cohens_d([1.0, math.inf], [1.0, 2.0])


class TestChiSquare:
    def test_gof_known(self) -> None:
        result = chi_square_gof(
            [16, 18, 16, 14, 12, 12],
            [16, 16, 16, 16, 16, 8],
        )
        assert abs(result.statistic - 3.5) < 1e-9
        assert result.df == 5
        assert abs(result.p_value - 0.6234) < 1e-3

    def test_gof_perfect_fit(self) -> None:
        result = chi_square_gof([10, 20, 30], [10, 20, 30])
        assert result.statistic == 0.0
        assert abs(result.p_value - 1.0) < 1e-9

    def test_gof_validation(self) -> None:
        with pytest.raises(ValueError):
            chi_square_gof([1, 2], [1, 2, 3])
        with pytest.raises(ValueError, match="2 categories"):
            chi_square_gof([1.0], [1.0])
        with pytest.raises(ValueError, match="non-negative"):
            chi_square_gof([10.0, -1.0], [5.0, 4.0])
        with pytest.raises(ValueError, match="finite"):
            chi_square_gof([10.0, math.nan], [5.0, 5.0])
        with pytest.raises(ValueError, match="positive"):
            chi_square_gof([10.0, 0.0], [10.0, 0.0])
        with pytest.raises(ValueError, match="finite"):
            chi_square_gof([10.0, 10.0], [10.0, math.inf])
        with pytest.raises(ValueError, match="equal totals"):
            chi_square_gof([10.0, 10.0], [9.0, 9.0])

    def test_independence_table(self) -> None:
        table: list[list[float]] = [[10.0, 20.0], [30.0, 40.0]]
        result = chi_square_independence(table)
        assert result.df == 1
        assert result.statistic >= 0
        assert 0.0 <= result.p_value <= 1.0

    def test_independence_independent_data(self) -> None:
        result = chi_square_independence([[10.0, 20.0], [20.0, 40.0]])
        assert result.statistic < 1e-9

    def test_independence_shape_validation(self) -> None:
        with pytest.raises(ValueError):
            chi_square_independence([[1.0, 2.0]])
        with pytest.raises(ValueError, match="rectangular"):
            chi_square_independence([[1.0], [2.0]])
        with pytest.raises(ValueError, match="rectangular"):
            chi_square_independence([[1.0, 2.0], [3.0]])

    def test_independence_count_validation(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            chi_square_independence([[1.0, -1.0], [2.0, 3.0]])
        with pytest.raises(ValueError, match="finite"):
            chi_square_independence([[1.0, math.inf], [2.0, 3.0]])
        with pytest.raises(ValueError, match="table total"):
            chi_square_independence([[0.0, 0.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="positive marginal"):
            chi_square_independence([[10.0, 20.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="positive marginal"):
            chi_square_independence([[10.0, 0.0], [20.0, 0.0]])


class TestANOVA:
    def test_known_f(self) -> None:
        result = one_way_anova([1, 2, 3], [2, 3, 4], [4, 5, 6])
        assert abs(result.statistic - 7.0) < 1e-9
        assert abs(result.p_value - 0.02702) < 1e-3

    def test_identical_groups(self) -> None:
        result = one_way_anova([1, 2, 3], [1, 2, 3])
        assert abs(result.statistic) < 1e-9

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            one_way_anova([1, 2, 3])
        with pytest.raises(ValueError, match="at least 1"):
            one_way_anova([], [1.0, 2.0])
        with pytest.raises(ValueError, match="more observations"):
            one_way_anova([1.0], [2.0])
        with pytest.raises(ValueError, match="finite"):
            one_way_anova([1.0, math.nan], [2.0, 3.0])
        with pytest.raises(ValueError, match="zero within-group variance"):
            one_way_anova([1.0, 1.0], [2.0, 2.0])


class TestCohensD:
    def test_known_value(self) -> None:
        value = cohens_d([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert abs(value - (-3.1623)) < 1e-3

    def test_identical_groups_zero(self) -> None:
        value = cohens_d([1, 2, 3, 4], [1, 2, 3, 4])
        assert abs(value) < 1e-9

    def test_sign_reflects_direction(self) -> None:
        first = cohens_d([1, 2, 3], [4, 5, 6])
        second = cohens_d([4, 5, 6], [1, 2, 3])
        assert first == -second
        assert first < 0
        assert second > 0

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            cohens_d([1.0], [1, 2, 3])
        with pytest.raises(ValueError, match="finite"):
            cohens_d([1.0, math.nan], [1.0, 2.0])
        with pytest.raises(ValueError):
            cohens_d([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])


class TestEtaSquared:
    def test_known_value(self) -> None:
        assert abs(eta_squared_from_f(7.0, 2, 6) - 0.7) < 1e-9

    def test_zero_f_zero_eta(self) -> None:
        assert eta_squared_from_f(0.0, 2, 6) == 0.0

    def test_range_in_unit_interval(self) -> None:
        value = eta_squared_from_f(1000.0, 2, 6)
        assert 0.0 < value < 1.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            eta_squared_from_f(-1.0, 2, 6)
        with pytest.raises(ValueError):
            eta_squared_from_f(math.nan, 2, 6)
        with pytest.raises(ValueError):
            eta_squared_from_f(7.0, 0, 6)
        with pytest.raises(ValueError):
            eta_squared_from_f(7.0, 2, 0)


class TestCramersV:
    def test_perfect_association(self) -> None:
        table: list[list[float]] = [[100.0, 0.0], [0.0, 100.0]]
        assert abs(cramers_v(table) - 1.0) < 1e-9

    def test_independent_table_zero(self) -> None:
        assert cramers_v([[10.0, 20.0], [20.0, 40.0]]) < 1e-9

    def test_range_in_unit_interval(self) -> None:
        value = cramers_v([[10.0, 20.0], [30.0, 40.0]])
        assert 0.0 <= value <= 1.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            cramers_v([[1.0, 2.0]])
        with pytest.raises(ValueError):
            cramers_v([[1.0], [2.0]])
        with pytest.raises(ValueError):
            cramers_v([[1.0, 2.0], [3.0]])
        with pytest.raises(ValueError):
            cramers_v([[0.0, 0.0], [0.0, 0.0]])
        with pytest.raises(ValueError):
            cramers_v([[math.nan, 1.0], [2.0, 3.0]])


class TestBonferroni:
    def test_known_correction(self) -> None:
        assert abs(bonferroni_corrected_alpha(0.05, 5) - 0.01) < 1e-12

    def test_k_one_is_noop(self) -> None:
        assert bonferroni_corrected_alpha(0.05, 1) == 0.05

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            bonferroni_corrected_alpha(0.05, 0)
        with pytest.raises(ValueError):
            bonferroni_corrected_alpha(0.05, -3)
        with pytest.raises(ValueError):
            bonferroni_corrected_alpha(0.0, 5)
        with pytest.raises(ValueError):
            bonferroni_corrected_alpha(1.0, 5)
        with pytest.raises(ValueError):
            bonferroni_corrected_alpha(math.nan, 5)

    def test_strict_monotone_in_k(self) -> None:
        assert bonferroni_corrected_alpha(0.05, 10) < bonferroni_corrected_alpha(0.05, 2)
