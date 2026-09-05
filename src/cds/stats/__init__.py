"""Statistical analysis tools."""

from cds.stats.bootstrap import BootstrapResult, bootstrap_ci, bootstrap_diff_ci
from cds.stats.descriptive import (
    correlation,
    mean,
    median,
    percentile,
    spearman_correlation,
    stdev,
    variance,
    z_scores,
)
from cds.stats.hypothesis_tests import (
    TestResult,
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
from cds.stats.multiple_testing import (
    MultipleTestingMethod,
    MultipleTestingResult,
    adjust_p_values,
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    holm,
)
from cds.stats.nonparametric import mann_whitney_u, wilcoxon_signed_rank
from cds.stats.power import (
    PowerResult,
    power_proportion_test,
    power_t_test,
    required_n_per_group,
)
from cds.stats.regression import RegressionResult, linear_regression
from cds.stats.time_series import (
    LjungBoxResult,
    StationarityResult,
    autocorrelation,
    autocorrelation_function,
    difference,
    exponential_smoothing,
    kpss_statistic,
    ljung_box,
    moving_average,
    partial_autocorrelation,
    seasonal_decompose,
)

__all__ = [
    "mean",
    "median",
    "stdev",
    "variance",
    "correlation",
    "spearman_correlation",
    "percentile",
    "z_scores",
    "linear_regression",
    "RegressionResult",
    "TestResult",
    "one_sample_ttest",
    "two_sample_ttest",
    "chi_square_gof",
    "chi_square_independence",
    "one_way_anova",
    "t_sf",
    "chi2_sf",
    "f_sf",
    "cohens_d",
    "eta_squared_from_f",
    "cramers_v",
    "bonferroni_corrected_alpha",
    # Multiple testing
    "MultipleTestingMethod",
    "MultipleTestingResult",
    "adjust_p_values",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "bonferroni",
    "holm",
    # Time-series analysis
    "autocorrelation",
    "autocorrelation_function",
    "partial_autocorrelation",
    "moving_average",
    "exponential_smoothing",
    "difference",
    "kpss_statistic",
    "seasonal_decompose",
    "ljung_box",
    "LjungBoxResult",
    "StationarityResult",
    # Nonparametric tests
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    # Bootstrap resampling
    "BootstrapResult",
    "bootstrap_ci",
    "bootstrap_diff_ci",
    # Power analysis
    "PowerResult",
    "power_t_test",
    "power_proportion_test",
    "required_n_per_group",
]
