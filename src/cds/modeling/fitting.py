"""Advanced parameter fitting with diagnostics, bounds, robust losses, and multi-start."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

from cds.core._numeric import DEFAULT_TOLERANCE, GD_DEFAULT_LR
from cds.modeling.model import MathModel
from cds.optimization import adam, gradient_descent, nelder_mead, projected_gradient_descent

FitLoss = Literal["squared", "absolute", "huber"]
FitOptimizer = Literal["auto", "gradient_descent", "adam", "nelder_mead", "projected_gradient"]
FitUncertainty = Literal["auto", "none", "normal", "bootstrap", "both"]


@dataclass(frozen=True)
class FitDiagnostics:
    """Diagnostics attached to an advanced parameter fit."""

    residuals: tuple[float, ...]
    rmse: float
    mae: float
    r_squared: float
    standard_errors: dict[str, float] | None
    confidence_intervals: dict[str, tuple[float, float]] | None
    identifiable: bool
    uncertainty_method: str = "none"
    bootstrap_confidence_intervals: dict[str, tuple[float, float]] | None = None
    bootstrap_successes: int = 0


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


def _prediction_residuals(
    model: MathModel,
    observations: Sequence[tuple[dict[str, float], float]],
    names: Sequence[str],
    values: Sequence[float],
    target_label: str,
) -> list[float]:
    target = model.equation(target_label)
    params = {**model.parameters, **dict(zip(names, values))}
    return [
        target.evaluate({**env, **params}) - observed_value for env, observed_value in observations
    ]


def _uncertainty(
    model: MathModel,
    observations: Sequence[tuple[dict[str, float], float]],
    names: Sequence[str],
    values: Sequence[float],
    target_label: str,
    residuals: Sequence[float],
    confidence: float,
) -> tuple[dict[str, float] | None, dict[str, tuple[float, float]] | None, bool]:
    """Local linearized covariance/normal-approximation uncertainty."""
    n = len(observations)
    p = len(names)
    if n <= p:
        return None, None, False

    params = {**model.parameters, **dict(zip(names, values))}
    derivatives = [model.gradient(target_label, name) for name in names]
    jacobian = [
        [derivative.evaluate({**env, **params}) for derivative in derivatives]
        for env, _ in observations
    ]
    jtj = [[sum(row[i] * row[j] for row in jacobian) for j in range(p)] for i in range(p)]
    try:
        inverse = _invert_matrix(jtj)
    except ValueError:
        return None, None, False

    rss = sum(residual * residual for residual in residuals)
    sigma2 = rss / (n - p)
    variances = [max(0.0, sigma2 * inverse[i][i]) for i in range(p)]
    standard_errors = {name: math.sqrt(variance) for name, variance in zip(names, variances)}
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    intervals = {
        name: (value - z * standard_errors[name], value + z * standard_errors[name])
        for name, value in zip(names, values)
    }
    return standard_errors, intervals, True


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_uncertainty(
    model: MathModel,
    observations: Sequence[tuple[dict[str, float], float]],
    names: Sequence[str],
    values: Sequence[float],
    *,
    target_label: str,
    optimizer: FitOptimizer,
    loss: FitLoss,
    bounds: Sequence[tuple[float, float]] | None,
    huber_delta: float,
    confidence: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    min_success_fraction: float,
    lr: float,
    tol: float,
    max_iter: int,
) -> tuple[
    dict[str, float] | None,
    dict[str, tuple[float, float]] | None,
    int,
]:
    """Nonparametric case-resampling uncertainty for nonlinear/robust fits."""
    rng = random.Random(bootstrap_seed)
    samples_by_name: dict[str, list[float]] = {name: [] for name in names}
    successes = 0
    n = len(observations)
    for _ in range(bootstrap_samples):
        resampled = [observations[rng.randrange(n)] for _ in range(n)]
        try:
            result = fit_parameters_advanced(
                model,
                resampled,
                names,
                x0=values,
                target_label=target_label,
                optimizer=optimizer,
                loss=loss,
                bounds=bounds,
                multi_start=1,
                seed=rng.randrange(2**31),
                huber_delta=huber_delta,
                confidence=confidence,
                uncertainty="none",
                lr=lr,
                tol=tol,
                max_iter=max_iter,
            )
        except (ArithmeticError, ValueError, ZeroDivisionError):
            continue
        candidate = [result.parameters[name] for name in names]
        if any(not math.isfinite(value) for value in candidate):
            continue
        successes += 1
        for name, value in zip(names, candidate):
            samples_by_name[name].append(value)

    required = max(2, math.ceil(bootstrap_samples * min_success_fraction))
    if successes < required:
        return None, None, successes

    alpha = (1.0 - confidence) / 2.0
    intervals = {
        name: (_percentile(samples, alpha), _percentile(samples, 1.0 - alpha))
        for name, samples in samples_by_name.items()
    }
    standard_errors = {}
    for name, samples in samples_by_name.items():
        mean = sum(samples) / len(samples)
        variance = sum((sample - mean) ** 2 for sample in samples) / (len(samples) - 1)
        standard_errors[name] = math.sqrt(max(0.0, variance))
    return standard_errors, intervals, successes


def fit_parameters_advanced(
    model: MathModel,
    observed: Sequence[tuple[dict[str, float], float]],
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
    uncertainty: FitUncertainty = "auto",
    bootstrap_samples: int = 200,
    bootstrap_seed: int | None = None,
    min_bootstrap_success_fraction: float = 0.8,
    lr: float = GD_DEFAULT_LR,
    tol: float = DEFAULT_TOLERANCE,
    max_iter: int = 10000,
) -> AdvancedFitResult:
    """Fit model parameters with method selection, diagnostics, and optional bootstrap CIs.

    ``uncertainty='normal'`` preserves the fast local Jacobian approximation for
    squared loss. ``'bootstrap'`` provides nonparametric case-resampling CIs
    that remain meaningful for nonlinear and robust-loss fits, while ``'both'``
    exposes both estimates for cross-checking. ``'auto'`` keeps the historical
    fast behavior (normal for squared loss, none for robust losses).
    """
    names = list(parameter_names)
    observations = list(observed)
    if not names:
        raise ValueError("parameter_names must list at least one parameter to fit")
    if len(set(names)) != len(names):
        raise ValueError("parameter_names must contain unique names")
    if not model.equations:
        raise ValueError("model must contain at least one equation to fit")
    if not observations:
        raise ValueError("observed must contain at least one (env, value) pair")
    start = list(x0) if x0 is not None else [0.0] * len(names)
    if len(start) != len(names):
        raise ValueError("x0 length must exactly match parameter_names")
    if multi_start < 1:
        raise ValueError("multi_start must be >= 1")
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if uncertainty not in ("auto", "none", "normal", "bootstrap", "both"):
        raise ValueError("unknown uncertainty method")
    if bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be >= 2")
    if not 0.0 < min_bootstrap_success_fraction <= 1.0:
        raise ValueError("min_bootstrap_success_fraction must lie in (0, 1]")
    if uncertainty in ("normal", "both") and loss != "squared":
        raise ValueError("normal-approximation uncertainty requires squared loss")
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
        residuals = _prediction_residuals(model, observations, names, values, label)
        return sum(_loss_value(residual, loss, huber_delta) for residual in residuals)

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

    residuals = _prediction_residuals(model, observations, names, best_values, label)
    n = len(residuals)
    rmse = math.sqrt(sum(value * value for value in residuals) / n)
    mae = sum(abs(value) for value in residuals) / n
    observed_values = [value for _, value in observations]
    observed_mean = sum(observed_values) / len(observed_values)
    total_variation = sum((value - observed_mean) ** 2 for value in observed_values)
    rss = sum(value * value for value in residuals)
    r_squared = 1.0 - rss / total_variation if total_variation > 0 else (1.0 if rss == 0 else 0.0)

    resolved_uncertainty: FitUncertainty
    if uncertainty == "auto":
        resolved_uncertainty = "normal" if loss == "squared" else "none"
    else:
        resolved_uncertainty = uncertainty

    normal_errors: dict[str, float] | None = None
    normal_intervals: dict[str, tuple[float, float]] | None = None
    normal_identifiable = False
    if resolved_uncertainty in ("normal", "both"):
        normal_errors, normal_intervals, normal_identifiable = _uncertainty(
            model,
            observations,
            names,
            best_values,
            label,
            residuals,
            confidence,
        )

    bootstrap_errors: dict[str, float] | None = None
    bootstrap_intervals: dict[str, tuple[float, float]] | None = None
    bootstrap_successes = 0
    if resolved_uncertainty in ("bootstrap", "both"):
        bootstrap_errors, bootstrap_intervals, bootstrap_successes = _bootstrap_uncertainty(
            model,
            observations,
            names,
            best_values,
            target_label=label,
            optimizer=optimizer,
            loss=loss,
            bounds=bounds,
            huber_delta=huber_delta,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=seed if bootstrap_seed is None else bootstrap_seed,
            min_success_fraction=min_bootstrap_success_fraction,
            lr=lr,
            tol=tol,
            max_iter=max_iter,
        )

    if resolved_uncertainty == "bootstrap":
        standard_errors = bootstrap_errors
        intervals = bootstrap_intervals
        identifiable = bootstrap_intervals is not None
    else:
        standard_errors = normal_errors
        intervals = normal_intervals
        identifiable = normal_identifiable
        if resolved_uncertainty == "both":
            identifiable = normal_identifiable and bootstrap_intervals is not None

    diagnostics = FitDiagnostics(
        residuals=tuple(residuals),
        rmse=rmse,
        mae=mae,
        r_squared=r_squared,
        standard_errors=standard_errors,
        confidence_intervals=intervals,
        identifiable=identifiable,
        uncertainty_method=resolved_uncertainty,
        bootstrap_confidence_intervals=bootstrap_intervals,
        bootstrap_successes=bootstrap_successes,
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
