"""``cds.stats.hypothesis_tests`` must not advertise private NR helpers.

The module's ``__all__`` used to list ten underscore-prefixed Numerical
Recipes internals (``_betacf``, ``_gser``, ``_EPS``, ...) "for compatibility".
That turned implementation details into a star-import surface and inflated
the advertised API. The helpers remain importable directly (backward
compatibility), but ``__all__`` — the declared public contract — carries only
the statistical tests and their distribution tails.
"""

from __future__ import annotations

import cds.stats.hypothesis_tests as ht

EXPECTED_PUBLIC = {
    "TestResult",
    "one_sample_ttest",
    "two_sample_ttest",
    "paired_ttest",
    "chi_square_gof",
    "chi_square_independence",
    "one_way_anova",
    "cohens_d",
    "paired_cohens_d",
    "eta_squared_from_f",
    "cramers_v",
    "bonferroni_corrected_alpha",
    "t_sf",
    "chi2_sf",
    "f_sf",
}

PRIVATE_HELPERS = (
    "_EPS",
    "_FPMIN",
    "_MAXIT",
    "_betacf",
    "_betai",
    "_gammln",
    "_gammp",
    "_gammq",
    "_gcf",
    "_gser",
)


def test_all_declares_only_the_public_statistical_surface() -> None:
    assert set(ht.__all__) == EXPECTED_PUBLIC
    assert not any(name.startswith("_") for name in ht.__all__)


def test_private_helpers_remain_directly_importable() -> None:
    # Backward compatibility: removing names from __all__ must not remove the
    # module attributes existing code imports explicitly.
    for name in PRIVATE_HELPERS:
        assert hasattr(ht, name), f"{name} vanished from the module body"
    assert ht._gser(1.0, 0.5) >= 0.0
    assert 0.0 <= ht._betai(0.5, 0.5, 0.5) <= 1.0


def test_star_import_no_longer_leaks_private_helpers() -> None:
    namespace: dict[str, object] = {}
    exec("from cds.stats.hypothesis_tests import *", namespace)  # noqa: S102
    # exec injects __builtins__ into a fresh globals dict; it is not an export.
    assert set(namespace) - {"__builtins__"} == EXPECTED_PUBLIC
    for name in PRIVATE_HELPERS:
        assert name not in namespace
