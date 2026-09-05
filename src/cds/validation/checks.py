"""Reusable scientific validation checks."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from cds.validation.report import CheckStatus, ValidationCheck, ValidationReport


def check_finite(values: Iterable[float], *, name: str = "finite-values") -> ValidationCheck:
    """Reject NaN and infinite numerical outputs."""
    materialized = list(values)
    bad = [index for index, value in enumerate(materialized) if not math.isfinite(value)]
    if bad:
        return ValidationCheck(
            name=name,
            status=CheckStatus.FAIL,
            message=f"non-finite values found at indices {bad}",
            details={"bad_indices": bad, "count": len(materialized)},
        )
    return ValidationCheck(
        name=name,
        status=CheckStatus.PASS,
        message=f"all {len(materialized)} values are finite",
        details={"count": len(materialized)},
    )


def cross_method_agreement(
    primary: float,
    secondary: float,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    name: str = "cross-method-agreement",
) -> ValidationCheck:
    """Compare two independently obtained scalar results."""
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if not (math.isfinite(primary) and math.isfinite(secondary)):
        return ValidationCheck(
            name=name,
            status=CheckStatus.FAIL,
            message="cannot compare non-finite method outputs",
            details={"primary": primary, "secondary": secondary},
        )
    delta = abs(primary - secondary)
    tolerance = atol + rtol * max(abs(primary), abs(secondary))
    status = CheckStatus.PASS if delta <= tolerance else CheckStatus.FAIL
    message = (
        "independent methods agree"
        if status is CheckStatus.PASS
        else "independent methods disagree"
    )
    return ValidationCheck(
        name=name,
        status=status,
        message=message,
        details={
            "primary": primary,
            "secondary": secondary,
            "delta": delta,
            "tolerance": tolerance,
        },
    )


def check_bounds(
    values: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    *,
    name: str = "domain-bounds",
) -> ValidationCheck:
    """Verify values lie inside declared scientific/domain bounds."""
    if len(values) != len(bounds):
        raise ValueError("values and bounds must have the same length")
    violations = [
        index
        for index, (value, (lower, upper)) in enumerate(zip(values, bounds))
        if value < lower or value > upper
    ]
    if violations:
        return ValidationCheck(
            name=name,
            status=CheckStatus.FAIL,
            message=f"domain-bound violations at indices {violations}",
            details={"violations": violations},
        )
    return ValidationCheck(
        name=name,
        status=CheckStatus.PASS,
        message="all values satisfy declared bounds",
    )


def check_positive(
    values: Sequence[float],
    *,
    allow_zero: bool = False,
    name: str = "positive-domain",
) -> ValidationCheck:
    """Validate positivity constraints common to physical/biological quantities."""
    violations = [
        index
        for index, value in enumerate(values)
        if (value < 0 if allow_zero else value <= 0) or not math.isfinite(value)
    ]
    if violations:
        return ValidationCheck(
            name=name,
            status=CheckStatus.FAIL,
            message=f"positivity violations at indices {violations}",
            details={"violations": violations, "allow_zero": allow_zero},
        )
    return ValidationCheck(
        name=name,
        status=CheckStatus.PASS,
        message="all values satisfy the positivity constraint",
        details={"allow_zero": allow_zero},
    )


def check_conservation(
    before: float,
    after: float,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    name: str = "conservation",
) -> ValidationCheck:
    """Check a scalar conservation law such as mass, charge, or energy."""
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if not (math.isfinite(before) and math.isfinite(after)):
        return ValidationCheck(
            name=name,
            status=CheckStatus.FAIL,
            message="cannot evaluate conservation with non-finite values",
            details={"before": before, "after": after},
        )
    delta = abs(after - before)
    tolerance = atol + rtol * max(abs(before), abs(after))
    status = CheckStatus.PASS if delta <= tolerance else CheckStatus.FAIL
    message = (
        "conservation law satisfied" if status is CheckStatus.PASS else "conservation law violated"
    )
    return ValidationCheck(
        name=name,
        status=status,
        message=message,
        details={"before": before, "after": after, "delta": delta, "tolerance": tolerance},
    )


def check_monotonic(
    values: Sequence[float],
    *,
    increasing: bool = True,
    strict: bool = False,
    name: str = "monotonic-domain",
) -> ValidationCheck:
    """Check monotonic trends required by a declared domain model."""
    finite = check_finite(values, name=name)
    if finite.status is CheckStatus.FAIL:
        return finite
    valid_pair: Callable[[float, float], bool]
    if strict:
        valid_pair = (
            (lambda left, right: left < right) if increasing else (lambda left, right: left > right)
        )
    else:
        valid_pair = (
            (lambda left, right: left <= right)
            if increasing
            else (lambda left, right: left >= right)
        )
    violations = [
        index
        for index, (left, right) in enumerate(zip(values, values[1:]))
        if not valid_pair(left, right)
    ]
    if violations:
        return ValidationCheck(
            name=name,
            status=CheckStatus.FAIL,
            message=f"monotonicity violations after indices {violations}",
            details={"violations": violations, "increasing": increasing, "strict": strict},
        )
    return ValidationCheck(
        name=name,
        status=CheckStatus.PASS,
        message="values satisfy the monotonicity constraint",
        details={"increasing": increasing, "strict": strict},
    )


def check_duplicate_rows(
    train_rows: Sequence[Sequence[object]],
    test_rows: Sequence[Sequence[object]],
    *,
    name: str = "data-leakage",
) -> ValidationCheck:
    """Detect exact train/test duplicates, a common validation leakage source."""
    train = {tuple(row) for row in train_rows}
    leaked = [index for index, row in enumerate(test_rows) if tuple(row) in train]
    if leaked:
        return ValidationCheck(
            name=name,
            status=CheckStatus.FAIL,
            message=f"{len(leaked)} test rows also occur in training data",
            details={"test_indices": leaked},
        )
    return ValidationCheck(
        name=name,
        status=CheckStatus.PASS,
        message="no exact train/test duplicate rows detected",
    )


def check_group_leakage(
    train_groups: Sequence[object],
    test_groups: Sequence[object],
    *,
    name: str = "group-leakage",
) -> ValidationCheck:
    """Detect entity/group identifiers crossing train/test boundaries."""
    train = set(train_groups)
    overlap = tuple(sorted(repr(group) for group in train.intersection(test_groups)))
    if overlap:
        return ValidationCheck(
            name=name,
            status=CheckStatus.FAIL,
            message=f"{len(overlap)} group identifiers occur in both train and test",
            details={"overlap": overlap},
        )
    return ValidationCheck(
        name=name,
        status=CheckStatus.PASS,
        message="no group identifiers cross train/test boundaries",
    )


def check_ood_ranges(
    training_rows: Sequence[Sequence[float]],
    candidate_rows: Sequence[Sequence[float]],
    *,
    allowed_fraction: float = 0.0,
    margin_fraction: float = 0.0,
    name: str = "out-of-distribution-range",
) -> ValidationCheck:
    """Flag rows outside per-feature training ranges.

    This intentionally uses an interpretable range check rather than claiming a
    full density model. ``margin_fraction`` expands each observed training range
    before evaluation, and ``allowed_fraction`` controls how much OOD traffic is
    tolerated before the check fails.
    """
    if not 0.0 <= allowed_fraction <= 1.0:
        raise ValueError("allowed_fraction must lie in [0, 1]")
    if margin_fraction < 0.0:
        raise ValueError("margin_fraction must be non-negative")
    if not training_rows:
        raise ValueError("training_rows must not be empty")
    width = len(training_rows[0])
    if width == 0:
        raise ValueError("rows must contain at least one feature")
    all_rows = [*training_rows, *candidate_rows]
    if any(len(row) != width for row in all_rows):
        raise ValueError("all OOD rows must have the same feature count")
    if any(not math.isfinite(value) for row in all_rows for value in row):
        return ValidationCheck(
            name=name,
            status=CheckStatus.FAIL,
            message="OOD range check cannot use non-finite feature values",
        )

    lower = [min(row[index] for row in training_rows) for index in range(width)]
    upper = [max(row[index] for row in training_rows) for index in range(width)]
    expanded = []
    for low, high in zip(lower, upper):
        scale = max(abs(low), abs(high), high - low, 1.0)
        margin = margin_fraction * scale
        expanded.append((low - margin, high + margin))

    outside = [
        row_index
        for row_index, row in enumerate(candidate_rows)
        if any(
            value < expanded[index][0] or value > expanded[index][1]
            for index, value in enumerate(row)
        )
    ]
    fraction = len(outside) / len(candidate_rows) if candidate_rows else 0.0
    status = CheckStatus.PASS if fraction <= allowed_fraction else CheckStatus.FAIL
    return ValidationCheck(
        name=name,
        status=status,
        message=(
            "candidate rows remain within declared training-domain ranges"
            if status is CheckStatus.PASS
            else f"{len(outside)} candidate rows fall outside training-domain ranges"
        ),
        details={
            "outside_indices": outside,
            "outside_fraction": fraction,
            "allowed_fraction": allowed_fraction,
            "ranges": expanded,
        },
    )


def check_distribution_drift(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    max_standardized_mean_shift: float = 0.5,
    max_variance_ratio: float = 2.0,
    name: str = "distribution-drift",
) -> ValidationCheck:
    """Detect large first/second-moment shifts between scalar distributions."""
    if max_standardized_mean_shift < 0:
        raise ValueError("max_standardized_mean_shift must be non-negative")
    if max_variance_ratio < 1.0:
        raise ValueError("max_variance_ratio must be >= 1")
    if len(reference) < 2 or len(current) < 2:
        raise ValueError("drift check requires at least two values per sample")
    finite = check_finite([*reference, *current], name=name)
    if finite.status is CheckStatus.FAIL:
        return finite

    def moments(values: Sequence[float]) -> tuple[float, float]:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return mean, variance

    ref_mean, ref_var = moments(reference)
    cur_mean, cur_var = moments(current)
    scale = max(math.sqrt(ref_var), 1e-12)
    mean_shift = abs(cur_mean - ref_mean) / scale
    if ref_var <= 1e-24 and cur_var <= 1e-24:
        variance_ratio = 1.0
    elif min(ref_var, cur_var) <= 1e-24:
        variance_ratio = math.inf
    else:
        variance_ratio = max(ref_var, cur_var) / min(ref_var, cur_var)
    drifted = mean_shift > max_standardized_mean_shift or variance_ratio > max_variance_ratio
    return ValidationCheck(
        name=name,
        status=CheckStatus.WARNING if drifted else CheckStatus.PASS,
        message="distribution drift detected" if drifted else "no material moment drift detected",
        details={
            "standardized_mean_shift": mean_shift,
            "variance_ratio": variance_ratio,
            "mean_shift_threshold": max_standardized_mean_shift,
            "variance_ratio_threshold": max_variance_ratio,
        },
    )


def check_residual_diagnostics(
    residuals: Sequence[float],
    *,
    max_bias_z: float = 2.0,
    max_abs_lag1: float = 0.3,
    max_variance_ratio: float = 2.0,
    name: str = "residual-diagnostics",
) -> ValidationCheck:
    """Screen residual bias, lag-1 dependence, and scale instability."""
    if max_bias_z < 0 or max_abs_lag1 < 0 or max_variance_ratio < 1:
        raise ValueError("residual diagnostic thresholds are invalid")
    if len(residuals) < 4:
        return ValidationCheck(
            name=name,
            status=CheckStatus.WARNING,
            message="too few residuals for dependence/heteroscedasticity diagnostics",
            details={"count": len(residuals)},
        )
    finite = check_finite(residuals, name=name)
    if finite.status is CheckStatus.FAIL:
        return finite

    n = len(residuals)
    mean = sum(residuals) / n
    centered = [value - mean for value in residuals]
    variance = sum(value * value for value in centered) / (n - 1)
    standard_error = math.sqrt(variance / n) if variance > 0 else 0.0
    bias_z = abs(mean) / standard_error if standard_error > 0 else (0.0 if mean == 0 else math.inf)

    denominator = sum(value * value for value in centered)
    lag1 = (
        sum(centered[index] * centered[index - 1] for index in range(1, n)) / denominator
        if denominator > 0
        else 0.0
    )

    midpoint = n // 2
    first = residuals[:midpoint]
    second = residuals[midpoint:]

    def sample_variance(values: Sequence[float]) -> float:
        local_mean = sum(values) / len(values)
        return sum((value - local_mean) ** 2 for value in values) / max(1, len(values) - 1)

    first_var = sample_variance(first)
    second_var = sample_variance(second)
    if first_var <= 1e-24 and second_var <= 1e-24:
        variance_ratio = 1.0
    elif min(first_var, second_var) <= 1e-24:
        variance_ratio = math.inf
    else:
        variance_ratio = max(first_var, second_var) / min(first_var, second_var)

    suspicious = (
        bias_z > max_bias_z or abs(lag1) > max_abs_lag1 or variance_ratio > max_variance_ratio
    )
    return ValidationCheck(
        name=name,
        status=CheckStatus.WARNING if suspicious else CheckStatus.PASS,
        message=(
            "residual diagnostics indicate possible model misspecification"
            if suspicious
            else "residual diagnostics show no configured warning signal"
        ),
        details={
            "bias_z": bias_z,
            "lag1": lag1,
            "variance_ratio": variance_ratio,
            "thresholds": {
                "max_bias_z": max_bias_z,
                "max_abs_lag1": max_abs_lag1,
                "max_variance_ratio": max_variance_ratio,
            },
        },
    )


def check_numerical_stability(
    function: Callable[[float], float],
    point: float,
    *,
    relative_step: float = 1e-7,
    max_relative_change: float = 1e-4,
    name: str = "numerical-stability",
) -> ValidationCheck:
    """Probe local numerical sensitivity using symmetric perturbations."""
    if relative_step <= 0:
        raise ValueError("relative_step must be positive")
    if max_relative_change < 0:
        raise ValueError("max_relative_change must be non-negative")
    step = relative_step * max(1.0, abs(point))
    center = function(point)
    left = function(point - step)
    right = function(point + step)
    finite = check_finite([center, left, right], name=name)
    if finite.status is CheckStatus.FAIL:
        return finite
    spread = max(abs(left - center), abs(right - center))
    scale = max(1.0, abs(center))
    relative_change = spread / scale
    status = CheckStatus.PASS if relative_change <= max_relative_change else CheckStatus.WARNING
    message = (
        "local perturbation check is stable"
        if status is CheckStatus.PASS
        else "result is locally sensitive to small perturbations"
    )
    return ValidationCheck(
        name=name,
        status=status,
        message=message,
        details={
            "point": point,
            "step": step,
            "relative_change": relative_change,
            "threshold": max_relative_change,
        },
    )


def final_audit(checks: Iterable[ValidationCheck]) -> ValidationReport:
    """Assemble the final independent audit report."""
    report = ValidationReport()
    for check in checks:
        report.add(check)
    return report
