"""Memory-bounded pipelines for HDF5, NetCDF, and array-like scientific data."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast

from cds.data_io.streaming import open_hdf5, open_netcdf


class SliceableArray(Protocol):
    """Minimal protocol implemented by HDF5/NetCDF arrays and similar objects."""

    @property
    def shape(self) -> Sequence[int]: ...

    @property
    def dtype(self) -> object: ...

    def __getitem__(self, key: tuple[slice, ...]) -> object: ...


@dataclass(frozen=True)
class ScientificArrayProfile:
    """Shape and bounded-chunk recommendation for one scientific array."""

    shape: tuple[int, ...]
    dtype: str
    itemsize: int | None
    total_elements: int
    estimated_bytes: int | None
    axis: int
    recommended_chunk_size: int


@dataclass(frozen=True)
class ArrayChunk:
    """One bounded slice from a scientific array."""

    index: int
    start: int
    stop: int
    data: object


StateT = TypeVar("StateT")


def _array_shape(array: object) -> tuple[int, ...]:
    raw = getattr(array, "shape", None)
    if raw is None:
        raise TypeError("scientific array must expose a shape")
    shape = tuple(int(value) for value in raw)
    if not shape:
        raise ValueError("scientific array must have at least one dimension")
    if any(value < 0 for value in shape):
        raise ValueError("scientific array dimensions must be non-negative")
    return shape


def _itemsize(array: object) -> int | None:
    dtype = getattr(array, "dtype", None)
    raw = getattr(dtype, "itemsize", None)
    if raw is None:
        return None
    value = int(raw)
    return value if value > 0 else None


def _dtype_name(array: object) -> str:
    dtype = getattr(array, "dtype", None)
    return "unknown" if dtype is None else str(dtype)


def _validate_axis(shape: Sequence[int], axis: int) -> None:
    if axis < 0 or axis >= len(shape):
        raise ValueError("axis is out of range for scientific array")


def _recommended_chunk_size(
    shape: Sequence[int],
    *,
    axis: int,
    itemsize: int | None,
    memory_budget_bytes: int,
) -> int:
    if memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    _validate_axis(shape, axis)
    bytes_per_value = itemsize if itemsize is not None else 8
    values_per_slice = math.prod(shape[index] for index in range(len(shape)) if index != axis)
    bytes_per_slice = max(1, values_per_slice * bytes_per_value)
    return max(1, memory_budget_bytes // bytes_per_slice)


def profile_scientific_array(
    array: object,
    *,
    axis: int = 0,
    memory_budget_bytes: int = 256 * 1024 * 1024,
) -> ScientificArrayProfile:
    """Profile an array-like object and choose a memory-bounded chunk size."""
    shape = _array_shape(array)
    size = _itemsize(array)
    chunk_size = _recommended_chunk_size(
        shape,
        axis=axis,
        itemsize=size,
        memory_budget_bytes=memory_budget_bytes,
    )
    total = math.prod(shape)
    return ScientificArrayProfile(
        shape=shape,
        dtype=_dtype_name(array),
        itemsize=size,
        total_elements=total,
        estimated_bytes=None if size is None else total * size,
        axis=axis,
        recommended_chunk_size=chunk_size,
    )


def iter_array_chunks(
    array: object,
    *,
    axis: int = 0,
    chunk_size: int | None = None,
    memory_budget_bytes: int = 256 * 1024 * 1024,
) -> Iterator[ArrayChunk]:
    """Yield bounded slices without materialising an entire scientific array."""
    shape = _array_shape(array)
    _validate_axis(shape, axis)
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided")
    rows = chunk_size
    if rows is None:
        rows = _recommended_chunk_size(
            shape,
            axis=axis,
            itemsize=_itemsize(array),
            memory_budget_bytes=memory_budget_bytes,
        )

    source = cast(SliceableArray, array)
    for index, start in enumerate(range(0, shape[axis], rows)):
        stop = min(shape[axis], start + rows)
        key = [slice(None)] * len(shape)
        key[axis] = slice(start, stop)
        yield ArrayChunk(index=index, start=start, stop=stop, data=source[tuple(key)])


def reduce_chunks(
    chunks: Iterable[ArrayChunk],
    reducer: Callable[[StateT, object], StateT],
    initial: StateT,
) -> StateT:
    """Reduce a chunk stream while keeping only caller-defined bounded state."""
    state = initial
    for chunk in chunks:
        state = reducer(state, chunk.data)
    return state


def iter_hdf5_chunks(
    path: str | os.PathLike[str],
    dataset: str,
    *,
    axis: int = 0,
    chunk_size: int | None = None,
    memory_budget_bytes: int = 256 * 1024 * 1024,
    **open_kwargs: Any,
) -> Iterator[ArrayChunk]:
    """Open an HDF5 file and stream one dataset through the CDS chunk pipeline."""
    if not dataset:
        raise ValueError("dataset must not be empty")
    with open_hdf5(path, "r", **open_kwargs) as handle:
        array = handle[dataset]
        yield from iter_array_chunks(
            array,
            axis=axis,
            chunk_size=chunk_size,
            memory_budget_bytes=memory_budget_bytes,
        )


def iter_netcdf_chunks(
    path: str | os.PathLike[str],
    variable: str,
    *,
    axis: int = 0,
    chunk_size: int | None = None,
    memory_budget_bytes: int = 256 * 1024 * 1024,
    **open_kwargs: Any,
) -> Iterator[ArrayChunk]:
    """Open a NetCDF file and stream one variable through the CDS chunk pipeline."""
    if not variable:
        raise ValueError("variable must not be empty")
    with open_netcdf(path, "r", **open_kwargs) as handle:
        variables = getattr(handle, "variables")
        array = variables[variable]
        yield from iter_array_chunks(
            array,
            axis=axis,
            chunk_size=chunk_size,
            memory_budget_bytes=memory_budget_bytes,
        )


__all__ = [
    "ArrayChunk",
    "ScientificArrayProfile",
    "iter_array_chunks",
    "iter_hdf5_chunks",
    "iter_netcdf_chunks",
    "profile_scientific_array",
    "reduce_chunks",
]
