"""Tests for bounded-memory streaming statistics and linear fitting."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from cds.data_io import (
    OnlineMoments,
    StreamingLinearAccumulator,
    fit_linear_csv_streaming,
)


def test_online_moments_matches_known_sample_statistics() -> None:
    moments = OnlineMoments()
    moments.extend([1.0, 2.0, 3.0, 4.0])
    assert moments.count == 4
    assert moments.mean == pytest.approx(2.5)
    assert moments.variance == pytest.approx(5.0 / 3.0)


def test_online_moments_merge_matches_single_pass() -> None:
    left = OnlineMoments()
    right = OnlineMoments()
    all_values = OnlineMoments()
    left.extend([1.0, 2.0])
    right.extend([3.0, 4.0, 5.0])
    all_values.extend([1.0, 2.0, 3.0, 4.0, 5.0])
    left.merge(right)
    assert left.count == all_values.count
    assert left.mean == pytest.approx(all_values.mean)
    assert left.variance == pytest.approx(all_values.variance)


def test_online_moments_edge_paths() -> None:
    empty = OnlineMoments()
    assert empty.variance is None
    one = OnlineMoments()
    one.update(2.0)
    assert one.variance is None
    empty.merge(OnlineMoments())
    empty.merge(one)
    assert empty.count == 1
    with pytest.raises(ValueError, match="finite"):
        empty.update(math.inf)


def test_streaming_linear_fit_recovers_exact_coefficients() -> None:
    accumulator = StreamingLinearAccumulator()
    for x in range(-20, 21):
        accumulator.update([float(x), float(x * x)], 1.5 + 2.0 * x - 0.25 * x * x)
    fit = accumulator.fit()
    assert fit.rows_seen == 41
    assert fit.intercept == pytest.approx(1.5, abs=1e-9)
    assert fit.coefficients == pytest.approx((2.0, -0.25), abs=1e-9)
    assert fit.rmse == pytest.approx(0.0, abs=1e-6)


def test_streaming_linear_fit_validates_shape_and_identifiability() -> None:
    accumulator = StreamingLinearAccumulator(max_features=2)
    with pytest.raises(ValueError, match="empty"):
        accumulator.fit()
    with pytest.raises(ValueError, match="at least one feature"):
        accumulator.update([], 1.0)
    with pytest.raises(ValueError, match="max_features"):
        accumulator.update([1.0, 2.0, 3.0], 1.0)
    accumulator.update([1.0, 2.0], 1.0)
    with pytest.raises(ValueError, match="same feature count"):
        accumulator.update([1.0], 1.0)
    with pytest.raises(ValueError, match="finite"):
        accumulator.update([1.0, math.nan], 1.0)
    with pytest.raises(ValueError, match="insufficient rows"):
        accumulator.fit()


def test_streaming_linear_fit_detects_singular_design() -> None:
    accumulator = StreamingLinearAccumulator()
    for x in range(5):
        accumulator.update([float(x), float(2 * x)], float(x))
    with pytest.raises(ValueError, match="singular"):
        accumulator.fit()


def test_streaming_csv_fit_never_materializes_full_dataset(tmp_path: Path) -> None:
    path = tmp_path / "large.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x1", "x2", "y"])
        for index in range(5000):
            x1 = float(index % 101)
            x2 = float((index // 101) % 37)
            writer.writerow([x1, x2, 3.0 + 1.25 * x1 - 0.5 * x2])

    fit = fit_linear_csv_streaming(
        path,
        ["x1", "x2"],
        "y",
        batch_size=17,
    )
    assert fit.rows_seen == 5000
    assert fit.intercept == pytest.approx(3.0, abs=1e-8)
    assert fit.coefficients == pytest.approx((1.25, -0.5), abs=1e-8)


def test_streaming_csv_fit_validates_schema_and_numeric_values(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("x,y\n1,nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        fit_linear_csv_streaming(path, ["x"], "y")
    with pytest.raises(ValueError, match="missing required column"):
        fit_linear_csv_streaming(path, ["missing"], "y")
    with pytest.raises(ValueError, match="non-empty"):
        fit_linear_csv_streaming(path, [], "y")
    with pytest.raises(ValueError, match="unique"):
        fit_linear_csv_streaming(path, ["x", "x"], "y")
    with pytest.raises(ValueError, match="target_column"):
        fit_linear_csv_streaming(path, ["x"], "")
    with pytest.raises(ValueError, match="also be a feature"):
        fit_linear_csv_streaming(path, ["x"], "x")


def test_streaming_accumulator_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="max_features"):
        StreamingLinearAccumulator(max_features=0)
