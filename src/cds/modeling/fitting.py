"""Advanced parameter fitting with replayable out-of-core observations."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal, TypeAlias

from cds.core._numeric import DEFAULT_TOLERANCE, GD_DEFAULT_LR
from cds.modeling.model import MathModel
from cds.optimization import adam, gradient_descent, nelder_mead, projected_gradient_descent

FitLoss = Literal["squared", "absolute", "huber"]
FitOptimizer = Literal["auto", "gradient_descent", "adam", "nelder_mead", "projected_gradient"]
Observation: TypeAlias = tuple[dict[str, float], float]
ObservationFactory: TypeAlias = Callable[[], Iterable[Observation]]
ObservationSource: TypeAlias = Sequence[Observation] | ObservationFactory


@dataclass(frozen=True)
class FitDiagnostics:
    """Diagnostics attached to an advanced parameter fit.

    ``residuals`` is empty when ``store_residuals=False``.  Summary metrics,
    uncertainty normal equations, and identifiability diagnostics are still
    accumulated exactly in bounded memory.
    """

    residuals: tuple[float, ...]
    rmse: float
    mae: float
    r_squared: float
    standard_errors: dict[str, float] | None
    confidence_intervals: dict[str, tuple[float, float]] | None
    identifiable: bool
    condition_number: float | None = None
    identifiability_reason: str = "not evaluated"
    observations: int = 0
    residuals_stored: bool = True


@dataclass(frozen=True)
class AdvancedFitResult:
    """Rich result from :func:`fit_parameters_advanced`."""

    parameters: dict[str, float]
    objective: float
    iterations: int
    converged: bool
    optimizer: str
    optimizer_rationale: str
    starts_tried: int
    diagnostics: FitDiagnostics


def _loss_value(residual: float, loss: FitLoss, huber_delta: float) -> float:
    if loss == "squared":
        return residual * residual
    if loss == "absolute":
        return abs(residual)
    magnitude = abs(residual)
    if magnitude <= huber_delta:
        return 0.5 * residual * residual
    return huber_delta * (magnitude - 0.5 * huber_delta)


def _select_optimizer(
    optimizer: FitOptimizer,
    loss: FitLoss,
    bounds: Sequence[tuple[float, float]] | None,
) -> tuple[str, str]:
    if optimizer != "auto":
        return optimizer, "explicitly selected by caller"
    if bounds is not None:
        return "projected_gradient", "box bounds require projected updates"
    if loss == "absolute":
        return "nelder_mead", "absolute loss is non-smooth, so derivative-free fitting is safer"
    return "adam", "smooth unconstrained objective; Adam is a robust default for parameter scaling"


def _make_starts(
    x0: list[float],
    count: int,
    bounds: Sequence[tuple[float, float]] | None,
    seed: int,
) -> list[list[float]]:
    starts = [list(x0)]
    rng = random.Random(seed)
    for _ in range(1, count):
        if bounds is not None:
            starts.append([rng.uniform(lower, upper) for lower, upper in bounds])
        else:
            starts.append([value + rng.gauss(0.0, 0.25 * max(1.0, abs(value))) for value in x0])
    return starts


def _invert_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Invert a small dense matrix with Gauss-Jordan elimination."""
    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be non-empty")
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")

    augmented = [
        [float(value) for value in row] + [1.0 if row_index == col else 0.0 for col in range(n)]
        for row_index, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row_index: abs(augmented[row_index][col]))
        if abs(augmented[pivot][col]) <= 1e-15:
            raise ValueError("matrix is singular")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        augmented[col] = [value / pivot_value for value in augmented[col]]
        for row_index in range(n):
            if row_index != col:
                factor = augmented[row_index][col]
                augmented[row_index] = [
                    value - factor * pivot_component
                    for value, pivot_component in zip(augmented[row_index], augmented[col])
                ]
    return [row[n:] for row in augmented]


def _matrix_inf_norm(matrix: Sequence[Sequence[float]]) -> float:
    return max((sum(abs(value) for value in row) for row in matrix), default=0.0)


def _condition_number_inf(
    matrix: Sequence[Sequence[float]], inverse: Sequence[Sequence[float]]
) -> float:
    """Estimate conditioning with the exact induced infinity norm."""
    return _matrix_inf_norm(matrix) * _matrix_inf_norm(inverse)


def _as_factory(observed: ObservationSource) -> ObservationFactory:
    if callable(observed):
        return observed

    def replay() -> Iterator[Observation]:
        return iter(observed)

    return replay


def _validated_count(factory: ObservationFactory) -> int:
    count = 0
    for env, observed_value in factory():
        count += 1
        if not math.isfinite(float(observed_value)):
            raise ValueError("observed values must be finite")
        if any(not math.isfinite(float(value)) for value in env.values()):
            raise ValueError("observation environments must contain only finite values")
    if count == 0:
        raise ValueError("observed must contain at least one (env, value) pair")
    return count


def _residual(
    model: MathModel,
    target_label: str,
    env: dict[str, float],
    observed_value: float,
    names: Sequence[str],
    values: Sequence[float],
) -> float:
    params = {**model.parameters, **dict(zip(names, values))}
    result = model.equation(target_label).evaluate({**env, **params}) - observed_value
    if not math.isfinite(result):
        raise ArithmeticError("model produced a non-finite residual")
    return result


def _prediction_residuals(
    model: MathModel,
    observations: Sequence[Observation],
    names: Sequence[str],
    values: Sequence[float],
    target_label: str,
) -> list[float]:
    """Compatibility helper for callers that explicitly request a sequence."""
    return [
        _residual(model, target_label, env, observed_value, names, values)
        for env, observed_value in observations
    ]


def _objective(
    factory: ObservationFactory,
    model: MathModel,
    names: Sequence[str],
    values: Sequence[float],
    target_label: str,
    loss: FitLoss,
    huber_delta: float,
) -> float:
    total = 0.0
    seen = 0
    for env, observed_value in factory():
        seen += 1
        residual = _residual(model, target_label, env, observed_value, names, values)
        total += _loss_value(residual, loss, huber_delta)
    if seen == 0:
        raise ValueError("replayable observation source returned no rows")
    return total


def _normal_equations(
    factory: ObservationFactory,
    model: MathModel,
    names: Sequence[str],
    values: Sequence[float],
    target_label: str,
) -> tuple[list[list[float]], float, int]:
    p = len(names)
    jtj = [[0.0] * p for _ in range(p)]
    rss = 0.0
    count = 0
    params = {**model.parameters, **dict(zip(names, values))}
    target = model.equation(target_label)
    derivatives = [model.gradient(target_label, name) for name in names]
    for env, observed_value in factory():
        bindings = {**env, **params}
        prediction = target.evaluate(bindings)
        residual = prediction - observed_value
        row = [derivative.evaluate(bindings) for derivative in derivatives]
        if not math.isfinite(residual) or any(not math.isfinite(value) for value in row):
            raise ArithmeticError("non-finite value encountered while building fitting diagnostics")
        count += 1
        rss += residual * residual
        for i in range(p):
            for j in range(p):
                jtj[i][j] += row[i] * row[j]
    return jtj, rss, count


def _uncertainty_details(
    model: MathModel,
    factory: ObservationFactory,
    names: Sequence[str],
    values: Sequence[float],
    target_label: str,
    confidence: float,
    condition_limit: float,
) -> tuple[
    dict[str, float] | None,
    dict[str, tuple[float, float]] | None,
    bool,
    float | None,
    str,
]:
    jtj, rss, n = _normal_equations(factory, model, names, values, target_label)
    p = len(names)
    if n <= p:
        return None, None, False, None, "underdetermined: observations must exceed parameters"
    try:
        inverse = _invert_matrix(jtj)
    except ValueError:
        return None, None, False, math.inf, "singular local information matrix"

    condition = _condition_number_inf(jtj, inverse)
    if not math.isfinite(condition) or condition > condition_limit:
        return (
            None,
            None,
            False,
            condition,
            f"practically non-identifiable: condition number exceeds {condition_limit:g}",
        )

    sigma2 = rss / (n - p)
    variances = [max(0.0, sigma2 * inverse[index][index]) for index in range(p)]
    standard_errors = {name: math.sqrt(variance) for name, variance in zip(names, variances)}
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    intervals = {
        name: (value - z * standard_errors[name], value + z * standard_errors[name])
        for name, value in zip(names, values)
    }
    return standard_errors, intervals, True, condition, "locally identifiable"


def _uncertainty(
    model: MathModel,
    observations: Sequence[Observation],
    names: Sequence[str],
    values: Sequence[float],
    target_label: str,
    residuals: Sequence[float],
    confidence: float,
) -> tuple[dict[str, float] | None, dict[str, tuple[float, float]] | None, bool]:
    """Backward-compatible uncertainty helper used by the historical tests."""
    del residuals
    standard_errors, intervals, identifiable, _, _ = _uncertainty_details(
        model,
        _as_factory(observations),
        names,
        values,
        target_label,
        confidence,
        1e10,
    )
    return standard_errors, intervals, identifiable


def _summary_diagnostics(
    factory: ObservationFactory,
    model: MathModel,
    names: Sequence[str],
    values: Sequence[float],
    target_label: str,
    *,
    store_residuals: bool,
) -> tuple[tuple[float, ...], int, float, float, float]:
    residuals: list[float] = []
    count = 0
    squared_error = 0.0
    absolute_error = 0.0
    observed_mean = 0.0
    observed_m2 = 0.0
    for env, observed_value in factory():
        residual = _residual(model, target_label, env, observed_value, names, values)
        count += 1
        squared_error += residual * residual
        absolute_error += abs(residual)
        delta = observed_value - observed_mean
        observed_mean += delta / count
        observed_m2 += delta * (observed_value - observed_mean)
        if store_residuals:
            residuals.append(residual)
    if count == 0:
        raise ValueError("replayable observation source returned no rows")
    rmse = math.sqrt(squared_error / count)
    mae = absolute_error / count
    r_squared = (
        1.0 - squared_error / observed_m2
        if observed_m2 > 0.0
        else (1.0 if squared_error == 0.0 else 0.0)
    )
    return tuple(residuals), count, rmse, mae, r_squared


def fit_parameters_advanced(
    model: MathModel,
    observed: ObservationSource,
    parameter_names: Sequence[str],
    x0: Sequence[float] | None = None,
    *,
    target_label: str | None = None,
    optimizer: FitOptimizer = "auto",
    loss: FitLoss = "squared",
    bounds: Sequence[tuple[float, float]] | None = None,
    multi_start: int = 1,
    seed: int = 0,
    huber_delta: float = 1.0,
    confidence: float = 0.95,
    lr: float = GD_DEFAULT_LR,
    tol: float = DEFAULT_TOLERANCE,
    max_iter: int = 10000,
    store_residuals: bool = True,
    identifiability_condition_limit: float = 1e10,
) -> AdvancedFitResult:
    """Fit parameters from a sequence or replayable out-of-core row factory.

    For large datasets pass a zero-argument callable that opens/replays the
    source on every invocation, for example a CSV/HDF5 batch reader flattened
    to ``(env, observed_value)`` rows.  Optimizer evaluations, final metrics,
    and ``J^T J`` uncertainty diagnostics then use bounded memory.  Set
    ``store_residuals=False`` to avoid retaining the final residual vector.

    Practical identifiability is rejected when the local information matrix is
    singular, underdetermined, or its induced infinity-norm condition number
    exceeds ``identifiability_condition_limit``.
    """
    names = list(parameter_names)
    if not names:
        raise ValueError("parameter_names must list at least one parameter to fit")
    if len(set(names)) != len(names):
        raise ValueError("parameter_names must contain unique names")
    if not model.equations:
        raise ValueError("model must contain at least one equation to fit")
    factory = _as_factory(observed)
    observation_count = _validated_count(factory)
    start = list(x0) if x0 is not None else [0.0] * len(names)
    if len(start) != len(names):
        raise ValueError("x0 length must exactly match parameter_names")
    if multi_start < 1:
        raise ValueError("multi_start must be >= 1")
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if (
        not math.isfinite(identifiability_condition_limit)
        or identifiability_condition_limit <= 1.0
    ):
        raise ValueError("identifiability_condition_limit must be finite and greater than 1")
    if bounds is not None:
        if len(bounds) != len(names):
            raise ValueError("bounds must have the same length as parameter_names")
        if any(lower > upper for lower, upper in bounds):
            raise ValueError("each bound must satisfy lower <= upper")
        if any(value < lower or value > upper for value, (lower, upper) in zip(start, bounds)):
            raise ValueError("x0 must lie inside bounds")
        if optimizer not in ("auto", "projected_gradient"):
            raise ValueError("bounded fitting requires optimizer='auto' or 'projected_gradient'")

    label = target_label if target_label is not None else model.equations[0][0]
    model.equation(label)

    def objective(values: list[float]) -> float:
        return _objective(factory, model, names, values, label, loss, huber_delta)

    chosen, rationale = _select_optimizer(optimizer, loss, bounds)
    starts = _make_starts(start, multi_start, bounds, seed)
    best_values: list[float] | None = None
    best_objective = math.inf
    best_iterations = 0
    best_converged = False

    for candidate in starts:
        if chosen == "gradient_descent":
            gd_result = gradient_descent(objective, candidate, lr=lr, tol=tol, max_iter=max_iter)
            values = list(gd_result.x)
            score = gd_result.value
            iterations = gd_result.iterations
            converged = gd_result.converged
        elif chosen == "adam":
            adam_result = adam(objective, candidate, lr=lr, tol=tol, max_iter=max_iter)
            values = list(adam_result.x)
            score = adam_result.value
            iterations = adam_result.iterations
            converged = adam_result.converged
        elif chosen == "nelder_mead":
            nm_result = nelder_mead(objective, candidate, max_iter=max_iter)
            values = list(nm_result.x)
            score = nm_result.value
            iterations = nm_result.iterations
            converged = nm_result.converged
        else:
            if bounds is None:
                raise ValueError("projected_gradient requires bounds")
            lower = [pair[0] for pair in bounds]
            upper = [pair[1] for pair in bounds]
            pg_result = projected_gradient_descent(
                objective,
                candidate,
                lower,
                upper,
                lr=lr,
                tol=tol,
                max_iter=max_iter,
            )
            values = list(pg_result.x)
            score = pg_result.fun
            iterations = pg_result.iterations
            converged = pg_result.converged

        if math.isfinite(score) and score < best_objective:
            best_values = values
            best_objective = score
            best_iterations = iterations
            best_converged = converged

    if best_values is None:
        raise ArithmeticError("all fitting attempts produced non-finite objective values")

    residuals, counted, rmse, mae, r_squared = _summary_diagnostics(
        factory,
        model,
        names,
        best_values,
        label,
        store_residuals=store_residuals,
    )
    if counted != observation_count:
        raise ValueError("replayable observation source changed row count between passes")

    if loss == "squared":
        standard_errors, intervals, identifiable, condition, reason = _uncertainty_details(
            model,
            factory,
            names,
            best_values,
            label,
            confidence,
            identifiability_condition_limit,
        )
    else:
        standard_errors = None
        intervals = None
        identifiable = False
        condition = None
        reason = f"identifiability covariance is not computed for {loss} loss"

    diagnostics = FitDiagnostics(
        residuals=residuals,
        rmse=rmse,
        mae=mae,
        r_squared=r_squared,
        standard_errors=standard_errors,
        confidence_intervals=intervals,
        identifiable=identifiable,
        condition_number=condition,
        identifiability_reason=reason,
        observations=observation_count,
        residuals_stored=store_residuals,
    )
    return AdvancedFitResult(
        parameters=dict(zip(names, best_values)),
        objective=best_objective,
        iterations=best_iterations,
        converged=best_converged,
        optimizer=chosen,
        optimizer_rationale=rationale,
        starts_tried=len(starts),
        diagnostics=diagnostics,
    )
