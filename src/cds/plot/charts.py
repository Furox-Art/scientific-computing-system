"""High-level plotting helpers built on the optional matplotlib backend.

Every public function:

* accepts plain Python sequences (``list[float]``), matching CDS core APIs;
* returns a matplotlib ``Figure`` (typed as ``Any`` so mypy stays clean without
  matplotlib stubs in the default environment);
* never imports matplotlib at module import time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cds.plot._backend import require_matplotlib
from cds.signals.processing import power_spectrum as _power_spectrum
from cds.stats.regression import linear_regression
from cds.stats.time_series import (
    autocorrelation_function,
    partial_autocorrelation,
    seasonal_decompose,
)

_TEAL = "#0d9488"
_SLATE = "#94a3b8"
_GREEN = "#22c55e"
_RED = "#ef4444"
_AMBER = "#f59e0b"
_INDIGO = "#6366f1"
_PALETTE = (_TEAL, _INDIGO, _AMBER, _RED, _GREEN, "#8b5cf6", "#06b6d4", "#f97316")


def _as_floats(values: Sequence[float], *, name: str) -> list[float]:
    data = [float(v) for v in values]
    if not data:
        raise ValueError(f"{name} must be non-empty")
    return data


def _style_axes(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_series(
    y: Sequence[float],
    *,
    x: Sequence[float] | None = None,
    title: str = "Series",
    xlabel: str = "index",
    ylabel: str = "value",
    color: str = _TEAL,
) -> Any:
    """Line plot of a 1-D series.

    Args:
        y: y-values.
        x: optional x-values (defaults to ``0..n-1``).
        title: figure title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        color: line color.

    Returns:
        A matplotlib ``Figure``.

    Raises:
        ImportError: if matplotlib is not installed.
        ValueError: if ``y`` is empty or ``x`` length mismatches ``y``.
    """
    ys = _as_floats(y, name="y")
    if x is None:
        xs: list[float] = [float(i) for i in range(len(ys))]
    else:
        xs = _as_floats(x, name="x")
        if len(xs) != len(ys):
            raise ValueError(f"x and y must have the same length (got {len(xs)} vs {len(ys)})")

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(xs, ys, color=color, linewidth=1.5)
    _style_axes(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def plot_multi_series(
    series: Mapping[str, Sequence[float]] | Sequence[tuple[str, Sequence[float]]],
    *,
    x: Sequence[float] | None = None,
    title: str = "Series",
    xlabel: str = "index",
    ylabel: str = "value",
) -> Any:
    """Overlay multiple labeled 1-D series on one axes.

    Args:
        series: either a ``{label: y_values}`` mapping or a sequence of
            ``(label, y_values)`` pairs. All series must share the same length.
        x: optional shared x-axis (defaults to ``0..n-1``).
        title: figure title.
        xlabel: x-axis label.
        ylabel: y-axis label.

    Returns:
        A matplotlib ``Figure``.
    """
    if isinstance(series, Mapping):
        items: list[tuple[str, Sequence[float]]] = list(series.items())
    else:
        items = list(series)
    if not items:
        raise ValueError("series must be non-empty")

    named: list[tuple[str, list[float]]] = []
    n0: int | None = None
    for label, vals in items:
        ys = _as_floats(vals, name=str(label))
        if n0 is None:
            n0 = len(ys)
        elif len(ys) != n0:
            raise ValueError(
                f"all series must share the same length (got {len(ys)} for {label!r}, expected {n0})"
            )
        named.append((str(label), ys))
    assert n0 is not None

    if x is None:
        xs = [float(i) for i in range(n0)]
    else:
        xs = _as_floats(x, name="x")
        if len(xs) != n0:
            raise ValueError(f"x length must match series length ({len(xs)} vs {n0})")

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    for i, (label, ys) in enumerate(named):
        ax.plot(xs, ys, color=_PALETTE[i % len(_PALETTE)], linewidth=1.5, label=label)
    _style_axes(ax, title, xlabel, ylabel)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_scatter(
    x: Sequence[float],
    y: Sequence[float],
    *,
    title: str = "Scatter",
    xlabel: str = "x",
    ylabel: str = "y",
    color: str = _TEAL,
) -> Any:
    """Scatter plot of paired ``(x, y)`` points."""
    xs = _as_floats(x, name="x")
    ys = _as_floats(y, name="y")
    if len(xs) != len(ys):
        raise ValueError(f"x and y must have the same length (got {len(xs)} vs {len(ys)})")

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, color=color, s=28, alpha=0.85, edgecolors="white", linewidths=0.4)
    _style_axes(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def plot_regression(
    x: Sequence[float],
    y: Sequence[float],
    *,
    title: str = "Linear regression",
    xlabel: str = "x",
    ylabel: str = "y",
) -> Any:
    """Scatter of ``(x, y)`` with OLS fit line from :func:`cds.stats.linear_regression`.

    The legend shows slope, intercept, and R².
    """
    xs = _as_floats(x, name="x")
    ys = _as_floats(y, name="y")
    if len(xs) != len(ys):
        raise ValueError(f"x and y must have the same length (got {len(xs)} vs {len(ys)})")
    fit = linear_regression(xs, ys)
    x_line = [min(xs), max(xs)]
    y_line = [fit.predict(xv) for xv in x_line]

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, color=_TEAL, s=28, alpha=0.85, edgecolors="white", linewidths=0.4, zorder=2)
    label = f"y = {fit.slope:.4g}x + {fit.intercept:.4g}  (R²={fit.r_squared:.3f})"
    ax.plot(x_line, y_line, color=_RED, linewidth=1.8, label=label, zorder=3)
    _style_axes(ax, title, xlabel, ylabel)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_histogram(
    values: Sequence[float],
    *,
    bins: int = 20,
    title: str = "Histogram",
    xlabel: str = "value",
    ylabel: str = "count",
    density: bool = False,
) -> Any:
    """Histogram of a sample.

    Args:
        values: sample values.
        bins: number of bins (must be >= 1).
        title: figure title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        density: if ``True``, normalize to a probability density.
    """
    data = _as_floats(values, name="values")
    if bins < 1:
        raise ValueError("bins must be >= 1")

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    ax.hist(
        data,
        bins=bins,
        color=_TEAL,
        edgecolor="white",
        alpha=0.9,
        density=density,
    )
    y_label = "density" if density else ylabel
    _style_axes(ax, title, xlabel, y_label)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_waveform(
    signal: Sequence[float],
    *,
    sample_rate: float | None = None,
    title: str = "Waveform",
) -> Any:
    """Time-domain waveform plot for signal-processing demos."""
    ys = _as_floats(signal, name="signal")
    if sample_rate is not None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        xs = [i / sample_rate for i in range(len(ys))]
        xlabel = "time (s)"
    else:
        xs = [float(i) for i in range(len(ys))]
        xlabel = "sample"
    return plot_series(ys, x=xs, title=title, xlabel=xlabel, ylabel="amplitude")


def plot_spectrum(
    magnitudes: Sequence[float],
    *,
    sample_rate: float | None = None,
    title: str = "Spectrum",
) -> Any:
    """Magnitude spectrum (e.g. from ``abs(fft(...))``)."""
    mags = _as_floats(magnitudes, name="magnitudes")
    n = len(mags)
    if sample_rate is not None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        freqs = [k * sample_rate / n for k in range(n)]
        xlabel = "frequency (Hz)"
    else:
        freqs = [float(k) for k in range(n)]
        xlabel = "bin"
    return plot_series(mags, x=freqs, title=title, xlabel=xlabel, ylabel="magnitude")


def plot_power_spectrum(
    signal: Sequence[float],
    *,
    sample_rate: float | None = None,
    half: bool = True,
    title: str = "Power spectrum",
) -> Any:
    """Power spectrum of a time-domain signal via :func:`cds.signals.power_spectrum`.

    Args:
        signal: time-domain samples.
        sample_rate: if given, x-axis is Hz over ``[0, sample_rate)`` (or Nyquist
            when ``half=True``).
        half: if ``True``, plot only the non-negative half of the spectrum.
        title: figure title.
    """
    data = _as_floats(signal, name="signal")
    # power_spectrum accepts list[float | complex]; list is invariant so copy.
    signal_in: list[float | complex] = list(data)
    power = _power_spectrum(signal_in)
    if half:
        n = len(power)
        power = power[: max(1, n // 2)]
        if sample_rate is not None:
            # bins span [0, sample_rate/2) for the half spectrum
            return plot_spectrum(power, sample_rate=sample_rate / 2.0, title=title)
    return plot_spectrum(power, sample_rate=sample_rate, title=title)


def plot_acf(
    data: Sequence[float],
    *,
    max_lag: int | None = None,
    title: str = "Autocorrelation (ACF)",
    partial: bool = False,
) -> Any:
    """Stem plot of the sample ACF (or PACF) via :mod:`cds.stats`."""
    series = _as_floats(data, name="data")
    if len(series) < 2:
        raise ValueError("data must have at least 2 observations")

    if partial:
        coeffs = partial_autocorrelation(series, max_lag=max_lag)
    else:
        coeffs = autocorrelation_function(series, max_lag=max_lag)

    lags = list(range(len(coeffs)))
    band = 1.96 / math.sqrt(len(series))

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(band, color=_SLATE, linestyle="--", linewidth=1.0, label="≈95% band")
    ax.axhline(-band, color=_SLATE, linestyle="--", linewidth=1.0)
    markerline, stemlines, baseline = ax.stem(lags, coeffs, basefmt=" ")
    plt.setp(markerline, color=_TEAL, markersize=4)
    plt.setp(stemlines, color=_TEAL, linewidth=1.2)
    plt.setp(baseline, visible=False)
    ax.set_title(title)
    ax.set_xlabel("lag")
    ax.set_ylabel("correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_seasonal_decompose(
    data: Sequence[float],
    period: int,
    *,
    title: str = "Seasonal decomposition",
) -> Any:
    """4-panel classical additive seasonal decomposition.

    Uses :func:`cds.stats.seasonal_decompose` for trend / seasonal / residual.
    """
    series = _as_floats(data, name="data")
    trend, seasonal, residual = seasonal_decompose(series, period)
    xs = [float(i) for i in range(len(series))]

    plt = require_matplotlib()
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 7))
    panels = (
        (series, "observed", _TEAL),
        (trend, "trend", _INDIGO),
        (seasonal, "seasonal", _AMBER),
        (residual, "residual", _RED),
    )
    for ax, (vals, name, color) in zip(axes, panels):
        ax.plot(xs, vals, color=color, linewidth=1.2)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"{title} (period={period})")
    axes[-1].set_xlabel("index")
    fig.tight_layout()
    return fig


def plot_heatmap(
    matrix: Sequence[Sequence[float]],
    *,
    title: str = "Heatmap",
    xlabel: str = "column",
    ylabel: str = "row",
    cmap: str = "viridis",
) -> Any:
    """2-D heatmap of a rectangular numeric matrix."""
    if not matrix:
        raise ValueError("matrix must be non-empty")
    rows = [list(map(float, row)) for row in matrix]
    n_cols = len(rows[0])
    if n_cols == 0:
        raise ValueError("matrix rows must be non-empty")
    if any(len(r) != n_cols for r in rows):
        raise ValueError("all matrix rows must have the same length")

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    im = ax.imshow(rows, aspect="auto", cmap=cmap, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _style_axes(ax, title, xlabel, ylabel)
    fig.tight_layout()
    return fig


def plot_loss(
    values: Sequence[float],
    *,
    title: str = "Loss curve",
    xlabel: str = "iteration",
    ylabel: str = "loss",
) -> Any:
    """Plot an optimizer / training loss history against iteration index."""
    return plot_series(values, title=title, xlabel=xlabel, ylabel=ylabel, color=_INDIGO)


def plot_optimization_path(
    history: Sequence[Sequence[float]],
    *,
    title: str = "Optimization trajectory",
    xlabel: str = "x₀",
    ylabel: str = "x₁",
) -> Any:
    """2-D path of iterates ``[(x0, x1), ...]`` from an optimizer."""
    if not history:
        raise ValueError("history must be non-empty")
    xs: list[float] = []
    ys: list[float] = []
    for i, point in enumerate(history):
        if len(point) < 2:
            raise ValueError(f"history[{i}] must have at least 2 coordinates")
        xs.append(float(point[0]))
        ys.append(float(point[1]))

    plt = require_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(xs, ys, color=_TEAL, linewidth=1.5, marker="o", markersize=3)
    ax.scatter([xs[0]], [ys[0]], color=_GREEN, s=40, zorder=3, label="start")
    ax.scatter([xs[-1]], [ys[-1]], color=_RED, s=40, zorder=3, label="end")
    _style_axes(ax, title, xlabel, ylabel)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _save_metadata(suffix: str) -> dict[str, Any] | None:
    """Drop writer-injected timestamps so a figure saves to identical bytes."""
    if suffix == ".pdf":
        return {"CreationDate": None, "ModDate": None}
    if suffix == ".svg":
        return {"Date": None}
    return None


def save_figure(fig: Any, path: str, *, dpi: int = 120, close: bool = False) -> str:
    """Save a figure to ``path`` and return ``path``.

    Convenience wrapper so demos/CLI do not need to call matplotlib APIs.
    When ``close`` is True, the figure is closed after save (frees memory in
    long demos / test suites).

    PDF and SVG exports are byte-reproducible: writer-injected timestamps are
    dropped and SVG element ids derive from a fixed hash salt, so saving the
    same figure twice yields identical bytes.
    """
    if dpi < 1:
        raise ValueError("dpi must be >= 1")
    plt = require_matplotlib()
    suffix = Path(path).suffix.lower()
    metadata = _save_metadata(suffix)
    with plt.rc_context({"svg.hashsalt": "cds"} if suffix == ".svg" else None):
        if metadata is None:
            fig.savefig(path, dpi=dpi)
        else:
            fig.savefig(path, dpi=dpi, metadata=metadata)
    if close:
        plt.close(fig)
    return path
