"""Tests for explicit data-sufficiency policy checks."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from cds.validation import (
    CheckStatus,
    DataProfile,
    DataRequirement,
    assess_data_adequacy,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"observations": -1}, "observations"),
        ({"observations": 1, "total_values": -1}, "total_values"),
        ({"observations": 1, "missing_values": -1}, "missing_values"),
        (
            {"observations": 1, "total_values": 2, "missing_values": 3},
            "cannot exceed",
        ),
        ({"observations": 1, "parameters": -1}, "parameters"),
    ],
)
def test_data_profile_rejects_invalid_counts(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DataProfile(**kwargs)


def test_data_profile_missing_fraction_handles_empty_and_nonempty_data() -> None:
    assert DataProfile(observations=0).missing_fraction == 0.0
    profile = DataProfile(observations=3, total_values=10, missing_values=2)
    assert profile.missing_fraction == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: DataRequirement(min_observations=-1), "min_observations"),
        (lambda: DataRequirement(max_missing_fraction=-0.1), "max_missing_fraction"),
        (lambda: DataRequirement(max_missing_fraction=1.1), "max_missing_fraction"),
        (
            lambda: DataRequirement(min_observations_per_parameter=0.0),
            "min_observations_per_parameter",
        ),
    ],
)
def test_data_requirement_rejects_invalid_thresholds(
    factory: Callable[[], DataRequirement],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_assess_data_adequacy_passes_explicit_requirements() -> None:
    profile = DataProfile(
        observations=100,
        total_values=400,
        missing_values=20,
        parameters=4,
    )
    requirement = DataRequirement(
        min_observations=50,
        max_missing_fraction=0.1,
        min_observations_per_parameter=10.0,
    )

    report = assess_data_adequacy(profile, requirement)

    assert report.passed
    assert [check.status for check in report.checks] == [
        CheckStatus.PASS,
        CheckStatus.PASS,
        CheckStatus.PASS,
    ]
    assert report.checks[-1].details["ratio"] == 25.0


def test_assess_data_adequacy_reports_declared_failures() -> None:
    profile = DataProfile(
        observations=8,
        total_values=10,
        missing_values=4,
        parameters=4,
    )
    requirement = DataRequirement(
        min_observations=10,
        max_missing_fraction=0.2,
        min_observations_per_parameter=3.0,
    )

    report = assess_data_adequacy(profile, requirement, name_prefix="fit")

    assert not report.passed
    assert [check.name for check in report.failures] == [
        "fit:observations",
        "fit:missingness",
        "fit:parameter-support",
    ]


def test_parameter_support_fails_closed_when_parameter_count_is_missing() -> None:
    report = assess_data_adequacy(
        DataProfile(observations=30),
        DataRequirement(min_observations_per_parameter=5.0),
    )

    support = report.checks[-1]
    assert support.status is CheckStatus.FAIL
    assert "parameter count is required" in support.message


def test_parameter_support_check_is_optional() -> None:
    report = assess_data_adequacy(
        DataProfile(observations=2),
        DataRequirement(min_observations=1),
    )
    assert len(report.checks) == 2


def test_assess_data_adequacy_rejects_empty_name_prefix() -> None:
    with pytest.raises(ValueError, match="name_prefix"):
        assess_data_adequacy(DataProfile(observations=1), DataRequirement(), name_prefix="")
