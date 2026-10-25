"""Tests for ``cds.plot`` chart helpers (skipped without matplotlib)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from cds.plot import (  # noqa: E402
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


def test_plot_series_basic() -> None:
    fig = plot_series([1.0, 2.0, 3.0, 2.5], title="demo")
    assert fig is not None
    assert len(fig.axes) == 1


def test_plot_series_with_x() -> None:
    fig = plot_series([1.0, 4.0, 9.0], x=[1.0, 2.0, 3.0])
    assert fig is not None


def test_plot_series_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_series([])


def test_plot_series_x_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_series([1.0, 2.0], x=[1.0])


def test_plot_multi_series_dict_and_pairs() -> None:
    a = [1.0, 2.0, 3.0]
    b = [3.0, 2.0, 1.0]
    fig1 = plot_multi_series({"a": a, "b": b}, title="dict")
    fig2 = plot_multi_series([("a", a), ("b", b)], x=[0.0, 1.0, 2.0])
    assert fig1 is not None and fig2 is not None


def test_plot_multi_series_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_multi_series({})


def test_plot_multi_series_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_multi_series({"a": [1.0, 2.0], "b": [1.0]})


def test_plot_multi_series_x_mismatch() -> None:
    with pytest.raises(ValueError, match="x length"):
        plot_multi_series({"a": [1.0, 2.0]}, x=[1.0])


def test_plot_scatter() -> None:
    fig = plot_scatter([1.0, 2.0, 3.0], [1.0, 4.0, 9.0])
    assert fig is not None


def test_plot_scatter_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_scatter([1.0, 2.0], [1.0])


def test_plot_regression() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.1, 3.9, 6.2, 7.8, 10.1]
    fig = plot_regression(x, y)
    assert fig is not None


def test_plot_regression_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_regression([1.0, 2.0], [1.0])


def test_plot_histogram() -> None:
    fig = plot_histogram([1.0, 1.1, 2.0, 2.2, 3.0, 3.5, 3.6], bins=5)
    assert fig is not None


def test_plot_histogram_density() -> None:
    fig = plot_histogram([1.0, 2.0, 2.0, 3.0], bins=3, density=True)
    assert fig is not None


def test_plot_histogram_bad_bins() -> None:
    with pytest.raises(ValueError, match="bins"):
        plot_histogram([1.0, 2.0], bins=0)


def test_plot_waveform_index_and_time() -> None:
    sig = [math.sin(2 * math.pi * 5 * t / 64) for t in range(64)]
    fig1 = plot_waveform(sig)
    fig2 = plot_waveform(sig, sample_rate=64.0)
    assert fig1 is not None and fig2 is not None


def test_plot_waveform_bad_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        plot_waveform([0.0, 1.0], sample_rate=0.0)


def test_plot_spectrum() -> None:
    mags = [abs(math.sin(i / 3)) for i in range(32)]
    fig = plot_spectrum(mags, sample_rate=32.0)
    assert fig is not None


def test_plot_spectrum_bin_axis() -> None:
    fig = plot_spectrum([1.0, 0.5, 0.25, 0.1])
    assert fig is not None


def test_plot_spectrum_bad_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        plot_spectrum([1.0, 2.0], sample_rate=-1.0)


def test_plot_power_spectrum() -> None:
    n = 64
    sig = [math.sin(2 * math.pi * 4 * t / n) for t in range(n)]
    fig_half = plot_power_spectrum(sig, sample_rate=float(n), half=True)
    fig_half_no_rate = plot_power_spectrum(sig, half=True)
    fig_full = plot_power_spectrum(sig, half=False)
    assert fig_half is not None and fig_half_no_rate is not None and fig_full is not None


def test_plot_acf_and_pacf() -> None:
    data = [0.0]
    for i in range(1, 80):
        data.append(0.7 * data[-1] + (0.1 if i % 2 == 0 else -0.05))
    fig_acf = plot_acf(data, max_lag=15)
    fig_pacf = plot_acf(data, max_lag=15, partial=True, title="PACF")
    assert fig_acf is not None and fig_pacf is not None


def test_plot_acf_too_short() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        plot_acf([1.0])


def test_plot_seasonal_decompose() -> None:
    # 4 periods of length 6
    period = 6
    base = [1.0, 2.0, 3.0, 2.0, 1.0, 0.5]
    data = base * 4
    fig = plot_seasonal_decompose(data, period=period)
    assert fig is not None
    assert len(fig.axes) == 4


def test_plot_heatmap() -> None:
    mat = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    fig = plot_heatmap(mat)
    assert fig is not None


def test_plot_heatmap_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_heatmap([])


def test_plot_heatmap_empty_row() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_heatmap([[]])


def test_plot_heatmap_ragged() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_heatmap([[1.0, 2.0], [3.0]])


def test_plot_loss() -> None:
    fig = plot_loss([1.0, 0.5, 0.25, 0.1])
    assert fig is not None


def test_plot_optimization_path() -> None:
    path = [(0.0, 0.0), (0.5, 0.2), (0.8, 0.5), (1.0, 1.0)]
    fig = plot_optimization_path(path)
    assert fig is not None


def test_plot_optimization_path_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_optimization_path([])


def test_plot_optimization_path_bad_point() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        plot_optimization_path([(1.0,)])


def test_save_figure(tmp_path: Path) -> None:
    path = tmp_path / "out.png"
    fig = plot_series([1.0, 2.0, 3.0])
    out = save_figure(fig, str(path), dpi=80, close=False)
    assert out == str(path)
    assert path.is_file()
    assert path.stat().st_size > 0
    # close=True frees the figure handle (covers the close branch)
    path2 = tmp_path / "out2.png"
    fig2 = plot_series([1.0, 2.0])
    save_figure(fig2, str(path2), close=True)
    assert path2.is_file()


def test_save_figure_bad_dpi() -> None:
    fig = plot_series([1.0, 2.0])
    with pytest.raises(ValueError, match="dpi"):
        save_figure(fig, "x.png", dpi=0)


def test_save_figure_pdf_has_no_write_timestamps(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    save_figure(plot_series([1.0, 2.0, 3.0]), str(first), close=True)
    save_figure(plot_series([1.0, 2.0, 3.0]), str(second), close=True)
    first_bytes = first.read_bytes()
    assert b"CreationDate" not in first_bytes
    assert b"ModDate" not in first_bytes
    assert first_bytes == second.read_bytes()


def test_save_figure_svg_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.svg"
    second = tmp_path / "second.svg"
    save_figure(plot_series([1.0, 2.0, 3.0]), str(first), close=True)
    save_figure(plot_series([1.0, 2.0, 3.0]), str(second), close=True)
    first_text = first.read_text(encoding="utf-8")
    assert "dc:date" not in first_text
    assert first_text == second.read_text(encoding="utf-8")


def test_require_matplotlib_respects_mplbackend(monkeypatch: pytest.MonkeyPatch) -> None:
    import cds.plot._backend as backend

    monkeypatch.setenv("MPLBACKEND", "Agg")
    plt = backend.require_matplotlib()
    assert plt is not None


def test_require_matplotlib_sets_agg_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    import cds.plot._backend as backend

    monkeypatch.delenv("MPLBACKEND", raising=False)
    plt = backend.require_matplotlib()
    assert plt is not None
