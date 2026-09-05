"""Targeted regression coverage for statistical audit hardening."""

import math

from cds.stats import paired_cohens_d


def test_paired_cohens_d_nonconstant_differences() -> None:
    """Paired Cohen dz uses the sample SD of non-constant pair differences."""
    first = [2.0, 5.0, 9.0, 10.0]
    second = [1.0, 3.0, 6.0, 8.0]
    differences = [left - right for left, right in zip(first, second)]
    mean_difference = sum(differences) / len(differences)
    variance = sum((value - mean_difference) ** 2 for value in differences) / (
        len(differences) - 1
    )
    expected = mean_difference / math.sqrt(variance)

    assert math.isclose(paired_cohens_d(first, second), expected, rel_tol=1e-12, abs_tol=1e-12)
