#!/usr/bin/env python3
"""Demo: optional ``cds[plot]`` matplotlib helpers.

Requires: ``pip install scientific-computing-system[plot]``

Saves figures under ``examples/_plot_out/``.
"""

from __future__ import annotations

import math
from pathlib import Path

from cds.plot import (
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
    plot_waveform,
    save_figure,
)


def main() -> None:
    out = Path(__file__).resolve().parent / "_plot_out"
    out.mkdir(exist_ok=True)

    y = [math.sin(2 * math.pi * i / 40) + 0.1 * math.sin(2 * math.pi * i / 7) for i in range(120)]
    save_figure(plot_series(y, title="Noisy sine"), str(out / "series.png"))
    save_figure(plot_histogram(y, bins=24, title="Amplitude histogram"), str(out / "hist.png"))
    save_figure(
        plot_multi_series(
            {
                "raw": y,
                "abs": [abs(v) for v in y],
            },
            title="Multi-series",
        ),
        str(out / "multi.png"),
    )

    n, fs = 128, 128.0
    signal = [math.sin(2 * math.pi * 8 * t / fs) for t in range(n)]
    save_figure(plot_waveform(signal, sample_rate=fs, title="8 Hz tone"), str(out / "wave.png"))
    save_figure(
        plot_power_spectrum(signal, sample_rate=fs, title="Power spectrum"),
        str(out / "power.png"),
    )

    ar = [0.0]
    for i in range(1, 100):
        ar.append(0.75 * ar[-1] + (0.2 if i % 3 == 0 else -0.05))
    save_figure(plot_acf(ar, max_lag=20, title="ACF"), str(out / "acf.png"))

    seasonal = [1.0, 2.0, 3.0, 2.0, 1.0, 0.5] * 5
    save_figure(
        plot_seasonal_decompose(seasonal, period=6, title="Seasonal"),
        str(out / "seasonal.png"),
    )

    xs = [float(i) for i in range(1, 11)]
    ys = [2.0 * x + 1.0 + (0.3 if i % 2 else -0.2) for i, x in enumerate(xs)]
    save_figure(plot_scatter(xs, ys, title="Scatter"), str(out / "scatter.png"))
    save_figure(plot_regression(xs, ys, title="OLS fit"), str(out / "regression.png"))

    mat = [[math.sin((i + 1) * (j + 1) / 5) for j in range(8)] for i in range(6)]
    save_figure(plot_heatmap(mat, title="Heatmap"), str(out / "heatmap.png"))

    save_figure(plot_loss([1.0 / (i + 1) for i in range(40)], title="Loss"), str(out / "loss.png"))
    path = [(math.cos(t / 8) * (1 - t / 40), math.sin(t / 8) * (1 - t / 40)) for t in range(40)]
    save_figure(plot_optimization_path(path, title="Spiral"), str(out / "opt_path.png"))

    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
