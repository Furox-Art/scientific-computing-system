"""Memory-bounded local file readers and optional scientific data adapters."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import math
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class FileProfile:
    """Basic local file metadata useful for planning memory-bounded analysis."""

    path: str
    size_bytes: int
    recommended_block_size: int


@dataclass
class OnlineMoments:
    """Numerically stable one-pass mean/variance accumulator.

    The state has constant memory and can therefore summarize arbitrarily large
    streams. ``merge`` implements the Chan parallel-combination formula so
    independently processed chunks can be combined without materializing rows.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("online moments require finite values")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.update(value)

    def merge(self, other: OnlineMoments) -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total

    @property
    def variance(self) -> float | None:
        """Return unbiased sample variance, or ``None`` before two observations."""
        return None if self.count < 2 else self.m2 / (self.count - 1)


@dataclass(frozen=True)
class StreamingLinearFit:
    """OLS fit obtained from bounded-memory sufficient statistics."""

    intercept: float
    coefficients: tuple[float, ...]
    rows_seen: int
    rmse: float


@dataclass
class StreamingLinearAccumulator:
    """Accumulate OLS normal-equation statistics in O(p²) memory.

    The number of observations does not affect memory use. This is intended for
    large local datasets where holding every row in RAM is neither necessary
    nor acceptable. The feature count is intentionally bounded because the
    dense normal matrix itself scales quadratically in ``p``.
    """

    max_features: int = 128
    rows_seen: int = 0
    _feature_count: int | None = None
    _xtx: list[list[float]] = field(default_factory=list)
    _xty: list[float] = field(default_factory=list)
    _yty: float = 0.0

    def __post_init__(self) -> None:
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")

    def update(self, features: Sequence[float], target: float) -> None:
        values = [float(value) for value in features]
        y = float(target)
        if not values:
            raise ValueError("streaming linear fit requires at least one feature")
        if len(values) > self.max_features:
            raise ValueError("feature count exceeds max_features")
        if any(not math.isfinite(value) for value in values) or not math.isfinite(y):
            raise ValueError("streaming linear fit requires finite values")

        if self._feature_count is None:
            self._feature_count = len(values)
            dimension = len(values) + 1
            self._xtx = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
            self._xty = [0.0 for _ in range(dimension)]
        elif len(values) != self._feature_count:
            raise ValueError("all streaming rows must have the same feature count")

        design = [1.0, *values]
        for row, left in enumerate(design):
            self._xty[row] += left * y
            for column in range(row, len(design)):
                contribution = left * design[column]
                self._xtx[row][column] += contribution
                if row != column:
                    self._xtx[column][row] += contribution
        self._yty += y * y
        self.rows_seen += 1

    def extend(self, rows: Iterable[tuple[Sequence[float], float]]) -> None:
        for features, target in rows:
            self.update(features, target)

    def fit(self) -> StreamingLinearFit:
        if self.rows_seen == 0 or self._feature_count is None:
            raise ValueError("cannot fit an empty stream")
        dimension = self._feature_count + 1
        if self.rows_seen < dimension:
            raise ValueError("insufficient rows for an identifiable linear fit")
        beta = _solve_dense(self._xtx, self._xty)
        rss = max(0.0, self._yty - sum(value * rhs for value, rhs in zip(beta, self._xty)))
        return StreamingLinearFit(
            intercept=beta[0],
            coefficients=tuple(beta[1:]),
            rows_seen=self.rows_seen,
            rmse=math.sqrt(rss / self.rows_seen),
        )


def _solve_dense(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve a small dense system with partial-pivot Gaussian elimination."""
    n = len(matrix)
    if n == 0 or len(rhs) != n or any(len(row) != n for row in matrix):
        raise ValueError("normal equation dimensions are inconsistent")
    augmented = [
        [float(value) for value in row] + [float(rhs[index])] for index, row in enumerate(matrix)
    ]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-14:
            raise ValueError("streaming linear fit is singular or non-identifiable")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for item in range(column, n + 1):
            augmented[column][item] /= pivot_value
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            for item in range(column, n + 1):
                augmented[row][item] -= factor * augmented[column][item]
    return [augmented[row][-1] for row in range(n)]


def fit_linear_csv_streaming(
    path: str | os.PathLike[str],
    feature_columns: Sequence[str],
    target_column: str,
    *,
    batch_size: int = 10_000,
    encoding: str = "utf-8",
    delimiter: str = ",",
    max_features: int = 128,
) -> StreamingLinearFit:
    """Fit OLS directly from CSV batches without materializing the dataset."""
    columns = tuple(feature_columns)
    if not columns or any(not column.strip() for column in columns):
        raise ValueError("feature_columns must contain non-empty names")
    if len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be unique")
    if not target_column.strip():
        raise ValueError("target_column must not be empty")
    if target_column in columns:
        raise ValueError("target_column must not also be a feature column")

    accumulator = StreamingLinearAccumulator(max_features=max_features)
    for batch in iter_csv_batches(
        path,
        batch_size=batch_size,
        encoding=encoding,
        delimiter=delimiter,
    ):
        for row in batch:
            try:
                features = [float(row[column]) for column in columns]
                target = float(row[target_column])
            except KeyError as exc:
                raise ValueError(f"CSV is missing required column {exc.args[0]!r}") from exc
            except (TypeError, ValueError) as exc:
                raise ValueError("CSV contains a non-numeric required value") from exc
            accumulator.update(features, target)
    return accumulator.fit()


def _validate_block_size(block_size: int) -> None:
    if block_size <= 0:
        raise ValueError("block_size must be positive")


def iter_file_blocks(
    path: str | os.PathLike[str],
    *,
    block_size: int = 8 * 1024 * 1024,
) -> Iterator[bytes]:
    """Yield a file in bounded byte blocks without loading it into memory."""
    _validate_block_size(block_size)
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            yield block


def profile_file(
    path: str | os.PathLike[str],
    *,
    memory_budget_bytes: int = 256 * 1024 * 1024,
    min_block_size: int = 1024 * 1024,
    max_block_size: int = 64 * 1024 * 1024,
) -> FileProfile:
    """Profile a file and recommend a conservative streaming block size."""
    if memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    if min_block_size <= 0 or max_block_size <= 0:
        raise ValueError("block sizes must be positive")
    if min_block_size > max_block_size:
        raise ValueError("min_block_size must not exceed max_block_size")
    if min_block_size > memory_budget_bytes:
        raise ValueError("min_block_size must not exceed memory_budget_bytes")

    source = Path(path)
    size = source.stat().st_size
    budget_block = max(1, memory_budget_bytes // 8)
    recommended = min(max_block_size, max(min_block_size, budget_block))
    if size > 0:
        recommended = min(recommended, max(min_block_size, size))
    return FileProfile(str(source), size, recommended)


def iter_csv_batches(
    path: str | os.PathLike[str],
    *,
    batch_size: int = 10_000,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> Iterator[list[dict[str, str]]]:
    """Yield CSV rows in bounded batches using only the standard library."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(delimiter) != 1:
        raise ValueError("delimiter must be exactly one character")
    with Path(path).open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return
        batch: list[dict[str, str]] = []
        for row in reader:
            batch.append(dict(row))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def _load_optional(module_name: str, distribution: str) -> ModuleType:
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(
            f"optional scientific I/O backend {module_name!r} is not installed; "
            f"install {distribution!r}"
        )
    return importlib.import_module(module_name)


def open_hdf5(
    path: str | os.PathLike[str],
    mode: str = "r",
    **kwargs: Any,
) -> Any:
    """Open HDF5 lazily through optional ``h5py`` without a core dependency."""
    h5py = _load_optional("h5py", "h5py")
    file_type = getattr(h5py, "File")
    return file_type(path, mode, **kwargs)


def open_netcdf(
    path: str | os.PathLike[str],
    mode: str = "r",
    **kwargs: Any,
) -> Any:
    """Open NetCDF lazily through optional ``netCDF4`` without a core dependency."""
    netcdf4 = _load_optional("netCDF4", "netCDF4")
    dataset_type = getattr(netcdf4, "Dataset")
    return dataset_type(path, mode, **kwargs)
