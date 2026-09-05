"""Memory-bounded local file readers and optional scientific data adapters."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class FileProfile:
    """Basic local file metadata useful for planning memory-bounded analysis."""

    path: str
    size_bytes: int
    recommended_block_size: int


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
    if memory_budget_bytes < min_block_size:
        raise ValueError("memory_budget_bytes must be at least min_block_size")

    source = Path(path)
    size = source.stat().st_size
    budget_block = max(1, memory_budget_bytes // 8)
    recommended = min(max_block_size, max(min_block_size, budget_block))
    # The explicit memory budget is a hard ceiling even when max_block_size is larger.
    recommended = min(recommended, memory_budget_bytes)
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
