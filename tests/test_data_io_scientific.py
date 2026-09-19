from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from cds.data_io import (
    ArrayChunk,
    iter_array_chunks,
    iter_hdf5_chunks,
    iter_netcdf_chunks,
    profile_scientific_array,
    reduce_chunks,
)


class FakeDType:
    def __init__(self, itemsize: int, name: str = "float32") -> None:
        self.itemsize = itemsize
        self.name = name

    def __str__(self) -> str:
        return self.name


class FakeArray:
    def __init__(
        self,
        rows: list[list[float]],
        *,
        itemsize: int | None = 4,
        dtype_name: str = "float32",
    ) -> None:
        self._rows = rows
        width = len(rows[0]) if rows else 2
        self.shape = (len(rows), width)
        if itemsize is not None:
            self.dtype = FakeDType(itemsize, dtype_name)

    def __getitem__(self, key: tuple[slice, ...]) -> object:
        row_slice, column_slice = key
        selected = self._rows[row_slice]
        return [row[column_slice] for row in selected]


class FakeHandle:
    def __init__(self, entries: dict[str, object], *, netcdf: bool = False) -> None:
        self._entries = entries
        if netcdf:
            self.variables = entries
        self.closed = False

    def __enter__(self) -> FakeHandle:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def __getitem__(self, name: str) -> object:
        return self._entries[name]


def _rows() -> list[list[float]]:
    return [[float(i), float(i + 10)] for i in range(5)]


def test_profile_and_automatic_chunks_are_memory_bounded() -> None:
    array = FakeArray(_rows())
    profile = profile_scientific_array(array, memory_budget_bytes=16)

    assert profile.shape == (5, 2)
    assert profile.dtype == "float32"
    assert profile.itemsize == 4
    assert profile.total_elements == 10
    assert profile.estimated_bytes == 40
    assert profile.recommended_chunk_size == 2

    chunks = list(iter_array_chunks(array, memory_budget_bytes=16))
    assert [(chunk.index, chunk.start, chunk.stop) for chunk in chunks] == [
        (0, 0, 2),
        (1, 2, 4),
        (2, 4, 5),
    ]
    assert chunks[0].data == [[0.0, 10.0], [1.0, 11.0]]
    assert chunks[-1].data == [[4.0, 14.0]]


def test_explicit_chunk_size_axis_and_empty_axis() -> None:
    array = FakeArray(_rows())
    chunks = list(iter_array_chunks(array, axis=0, chunk_size=3))
    assert [(chunk.start, chunk.stop) for chunk in chunks] == [(0, 3), (3, 5)]

    empty = FakeArray([])
    assert list(iter_array_chunks(empty, chunk_size=2)) == []


def test_unknown_or_invalid_dtype_itemsize_uses_conservative_fallback() -> None:
    unknown = FakeArray(_rows(), itemsize=None)
    profile = profile_scientific_array(unknown, memory_budget_bytes=32)
    assert profile.dtype == "unknown"
    assert profile.itemsize is None
    assert profile.estimated_bytes is None
    assert profile.recommended_chunk_size == 2

    zero = FakeArray(_rows(), itemsize=0, dtype_name="zero")
    zero_profile = profile_scientific_array(zero, memory_budget_bytes=32)
    assert zero_profile.dtype == "zero"
    assert zero_profile.itemsize is None


def test_shape_axis_and_budget_validation() -> None:
    with pytest.raises(TypeError, match="expose a shape"):
        profile_scientific_array(object())

    scalar = SimpleNamespace(shape=(), dtype=FakeDType(8))
    with pytest.raises(ValueError, match="at least one dimension"):
        profile_scientific_array(scalar)

    negative = SimpleNamespace(shape=(2, -1), dtype=FakeDType(8))
    with pytest.raises(ValueError, match="non-negative"):
        profile_scientific_array(negative)

    array = FakeArray(_rows())
    with pytest.raises(ValueError, match="axis is out of range"):
        profile_scientific_array(array, axis=2)
    with pytest.raises(ValueError, match="memory_budget_bytes"):
        profile_scientific_array(array, memory_budget_bytes=0)
    with pytest.raises(ValueError, match="chunk_size"):
        list(iter_array_chunks(array, chunk_size=0))
    with pytest.raises(ValueError, match="axis is out of range"):
        list(iter_array_chunks(array, axis=-1, chunk_size=1))


def test_reduce_chunks_keeps_only_reducer_state() -> None:
    chunks = (
        ArrayChunk(index=0, start=0, stop=1, data=[1.0, 2.0]),
        ArrayChunk(index=1, start=1, stop=2, data=[3.0]),
    )

    def reducer(total: float, data: object) -> float:
        values = data
        assert isinstance(values, list)
        return total + sum(float(value) for value in values)

    assert reduce_chunks(chunks, reducer, 0.0) == 6.0


def test_hdf5_wrapper_opens_streams_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    array = FakeArray(_rows())
    handle = FakeHandle({"measurements": array})
    calls: list[tuple[object, ...]] = []

    def fake_open(*args: object, **kwargs: Any) -> FakeHandle:
        calls.append((*args, kwargs))
        return handle

    monkeypatch.setattr("cds.data_io.scientific.open_hdf5", fake_open)
    chunks = list(
        iter_hdf5_chunks(
            "data.h5",
            "measurements",
            chunk_size=2,
            libver="latest",
        )
    )
    assert len(chunks) == 3
    assert calls == [("data.h5", "r", {"libver": "latest"})]
    assert handle.closed

    with pytest.raises(ValueError, match="dataset"):
        list(iter_hdf5_chunks("data.h5", ""))


def test_netcdf_wrapper_opens_streams_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    array = FakeArray(_rows())
    handle = FakeHandle({"temperature": array}, netcdf=True)
    calls: list[tuple[object, ...]] = []

    def fake_open(*args: object, **kwargs: Any) -> FakeHandle:
        calls.append((*args, kwargs))
        return handle

    monkeypatch.setattr("cds.data_io.scientific.open_netcdf", fake_open)
    chunks = list(iter_netcdf_chunks("data.nc", "temperature", chunk_size=4, diskless=True))
    assert [(chunk.start, chunk.stop) for chunk in chunks] == [(0, 4), (4, 5)]
    assert calls == [("data.nc", "r", {"diskless": True})]
    assert handle.closed

    with pytest.raises(ValueError, match="variable"):
        list(iter_netcdf_chunks("data.nc", ""))


def test_chunk_iterator_type_is_lazy() -> None:
    iterator = iter_array_chunks(FakeArray(_rows()), chunk_size=2)
    assert isinstance(iterator, Iterator)
