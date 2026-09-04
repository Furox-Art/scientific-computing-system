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
    return ValidationCheck(
        name=name,
        status=status,
        message="conservation law satisfied" if status is CheckStatus.PASS else "conservation law violated",
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
    if strict:
        valid_pair = (lambda left, right: left < right) if increasing else (lambda left, right: left > right)
    else:
        valid_pair = (lambda left, right: left <= right) if increasing else (lambda left, right: left >= right)
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
