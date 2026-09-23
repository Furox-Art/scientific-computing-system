from __future__ import annotations

import math

import pytest

from cds.validation import (
    CheckStatus,
    drift_validation_check,
    empirical_ks_distance,
    feature_drift,
    ood_validation_check,
    screen_ood,
)


def test_empirical_ks_distance_identical_disjoint_and_tied_samples() -> None:
    assert empirical_ks_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert empirical_ks_distance([0.0, 1.0], [2.0, 3.0]) == 1.0
    assert empirical_ks_distance([0.0, 0.0, 1.0], [0.0, 1.0, 1.0]) == pytest.approx(1.0 / 3.0)


def test_feature_drift_flags_shifted_feature_only() -> None:
    reference = [[0.0, 10.0], [1.0, 10.0], [2.0, 10.0], [3.0, 10.0]]
    current = [[10.0, 10.0], [11.0, 10.0], [12.0, 10.0], [13.0, 10.0]]
    report = feature_drift(reference, current, threshold=0.5)

    assert report.reference_rows == 4
    assert report.current_rows == 4
    assert report.features[0].ks_distance == 1.0
    assert report.features[0].drifted
    assert report.features[1].ks_distance == 0.0
    assert not report.features[1].drifted
    assert report.any_drift


def test_feature_at_exact_threshold_is_not_silently_promoted_to_drift() -> None:
    reference = [[0.0], [1.0]]
    current = [[0.0], [2.0]]
    report = feature_drift(reference, current, threshold=0.5)
    assert report.features[0].ks_distance == 0.5
    assert not report.features[0].drifted
    assert not report.any_drift


def test_ood_screening_handles_standard_and_zero_variance_features() -> None:
    reference = [[0.0, 5.0], [1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]
    report = screen_ood(reference, [[1.5, 5.0], [10.0, 5.0], [1.5, 6.0]], z_threshold=3.0)

    assert not report.observations[0].out_of_distribution
    assert report.observations[1].out_of_distribution
    assert report.observations[2].out_of_distribution
    assert math.isinf(report.observations[2].max_abs_z)
    assert report.any_ood


def test_ood_report_without_flags() -> None:
    report = screen_ood([[0.0], [1.0], [2.0]], [[1.0]], z_threshold=3.0)
    assert not report.any_ood
    assert report.observations[0].max_abs_z == 0.0


def test_drift_and_ood_translate_to_common_validation_checks() -> None:
    clean_drift = feature_drift([[0.0], [1.0]], [[0.0], [1.0]], threshold=0.2)
    shifted_drift = feature_drift([[0.0], [1.0]], [[10.0], [11.0]], threshold=0.2)
    assert drift_validation_check(clean_drift).status is CheckStatus.PASS
    warning = drift_validation_check(shifted_drift)
    failure = drift_validation_check(shifted_drift, fail_on_drift=True)
    assert warning.status is CheckStatus.WARNING
    assert failure.status is CheckStatus.FAIL
    assert "[0]" in warning.message

    clean_ood = screen_ood([[0.0], [1.0]], [[0.5]], z_threshold=4.0)
    shifted_ood = screen_ood([[0.0], [1.0]], [[10.0]], z_threshold=4.0)
    assert ood_validation_check(clean_ood).status is CheckStatus.PASS
    assert ood_validation_check(shifted_ood).status is CheckStatus.WARNING
    assert ood_validation_check(shifted_ood, fail_on_ood=True).status is CheckStatus.FAIL
    assert "[0]" in ood_validation_check(shifted_ood).message


def test_drift_threshold_and_matrix_contracts() -> None:
    for threshold in (-0.1, 1.1, math.inf, math.nan):
        with pytest.raises(ValueError, match="threshold"):
            feature_drift([[0.0]], [[0.0]], threshold=threshold)

    with pytest.raises(ValueError, match="reference must contain at least one row"):
        feature_drift([], [[0.0]])
    with pytest.raises(ValueError, match="current must contain at least one row"):
        feature_drift([[0.0]], [])
    with pytest.raises(ValueError, match="at least one feature"):
        feature_drift([[]], [[]])
    with pytest.raises(ValueError, match="equal width"):
        feature_drift([[0.0], [1.0, 2.0]], [[0.0]])
    with pytest.raises(ValueError, match="finite values"):
        feature_drift([[math.nan]], [[0.0]])
    with pytest.raises(ValueError, match="equal feature count"):
        feature_drift([[0.0]], [[0.0, 1.0]])


def test_ks_and_ood_input_contracts() -> None:
    with pytest.raises(ValueError, match="reference must not be empty"):
        empirical_ks_distance([], [1.0])
    with pytest.raises(ValueError, match="current must not be empty"):
        empirical_ks_distance([1.0], [])
    with pytest.raises(ValueError, match="finite values"):
        empirical_ks_distance([math.inf], [1.0])

    for threshold in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="z_threshold"):
            screen_ood([[0.0]], [[0.0]], z_threshold=threshold)
    with pytest.raises(ValueError, match="observations must contain at least one row"):
        screen_ood([[0.0]], [])
    with pytest.raises(ValueError, match="equal feature count"):
        screen_ood([[0.0]], [[0.0, 1.0]])
    with pytest.raises(ValueError, match="observations row 0"):
        screen_ood([[0.0]], [[math.nan]])
