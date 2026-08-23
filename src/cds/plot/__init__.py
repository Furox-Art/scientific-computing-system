"""Optional matplotlib plotting helpers for CDS numerical modules.

This package is **optional**: matplotlib is imported lazily inside each
plotting call, so ``import cds`` and ``import cds.plot`` stay free of
heavy dependencies. Install with::

    pip install scientific-computing-system[plot]

Design mirrors :mod:`cds.data_analysis.pandas_io`: clear install-hint
``ImportError`` when matplotlib is missing, pure-Python inputs
(``list[float]`` / lag arrays from ``cds.stats``), and Agg-friendly
figures suitable for scripts, notebooks, and headless CI.
"""

from __future__ import annotations

from cds.plot.charts import (
    plot_acf,
    plot_heatmap,
    plot_histogram,
    plot_loss,
    plot_multi_series,
    plot_optimization_path,
    plot_power_spectrum,
    plot_regression,
    plot_scatter,
    plot_seasonal_decompose,
    plot_series,
    plot_spectrum,
    plot_waveform,
    save_figure,
)

__all__ = [
    "plot_series",
    "plot_multi_series",
    "plot_scatter",
    "plot_regression",
    "plot_histogram",
    "plot_waveform",
    "plot_spectrum",
    "plot_power_spectrum",
    "plot_acf",
    "plot_seasonal_decompose",
    "plot_heatmap",
    "plot_loss",
    "plot_optimization_path",
    "save_figure",
]
