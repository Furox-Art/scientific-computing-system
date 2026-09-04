"""Tests for memory-bounded file readers and optional scientific I/O backends."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from cds.data_io import iter_csv_batches, iter_file_blocks, open_hdf5, open_netcdf, profile_file


def test_iter_file_blocks_streams_bounded_chunks(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abcdefghij")
    assert list(iter_file_blocks(path, block_size=4)) == [b"abcd", b"efgh", b"ij"]

    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert list(iter_file_blocks(empty, block_size=4)) == []

    with pytest.raises(ValueError, match="block_size"):
        list(iter_file_blocks(path, block_size=0))


def test_profile_file_recommends_memory_bounded_blocks(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"x" * 20)

    profile = profile_file(
        path,
        memory_budget_bytes=80,
        min_block_size=4,
        max_block_size=64,
    )
    assert profile.path == str(path)
    assert profile.size_bytes == 20
    assert profile.recommended_block_size == 10

    small = profile_file(
        path,
        memory_budget_bytes=8,
        min_block_size=4,
        max_block_size=64,
    )
    assert small.recommended_block_size == 4

    capped = profile_file(
        path,
        memory_budget_bytes=10_000,
        min_block_size=4,
        max_block_size=8,
    )
    assert capped.recommended_block_size == 8

    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    empty_profile = profile_file(
        empty,
        memory_budget_bytes=80,
        min_block_size=4,
        max_block_size=64,
    )
    assert empty_profile.recommended_block_size == 10

    with pytest.raises(ValueError, match="memory_budget_bytes"):
        profile_file(path, memory_budget_bytes=0)
    with pytest.raises(ValueError, match="block sizes"):
        profile_file(path, min_block_size=0)
    with pytest.raises(ValueError, match="block sizes"):
        profile_file(path, max_block_size=0)
    with pytest.raises(ValueError, match="must not exceed"):
        profile_file(path, min_block_size=8, max_block_size=4)


def test_iter_csv_batches_preserves_headers_and_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,x\n2,y\n3,z\n", encoding="utf-8")

    batches = list(iter_csv_batches(path, batch_size=2))
    assert batches == [
        [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}],
        [{"a": "3", "b": "z"}],
    ]

    semicolon = tmp_path / "semi.csv"
    semicolon.write_text("a;b\n1;x\n", encoding="utf-8")
    assert list(iter_csv_batches(semicolon, delimiter=";")) == [[{"a": "1", "b": "x"}]]

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert list(iter_csv_batches(empty)) == []

    header_only = tmp_path / "header.csv"
    header_only.write_text("a,b\n", encoding="utf-8")
    assert list(iter_csv_batches(header_only)) == []

    with pytest.raises(ValueError, match="batch_size"):
        list(iter_csv_batches(path, batch_size=0))
    with pytest.raises(ValueError, match="delimiter"):
        list(iter_csv_batches(path, delimiter=""))


def test_optional_hdf5_backend_is_lazy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[object, ...]] = []
    fake = ModuleType("h5py")

    def fake_file(path: object, mode: str, **kwargs: object) -> tuple[object, ...]:
        result = (path, mode, kwargs)
        calls.append(result)
        return result

    fake.File = fake_file  # type: ignore[attr-defined]
    monkeypatch.setattr("cds.data_io.streaming.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr("cds.data_io.streaming.importlib.import_module", lambda _name: fake)

    path = tmp_path / "data.h5"
    result = open_hdf5(path, "a", libver="latest")
    assert result == (path, "a", {"libver": "latest"})
    assert calls == [result]


def test_optional_netcdf_backend_is_lazy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = ModuleType("netCDF4")

    def fake_dataset(path: object, mode: str, **kwargs: object) -> tuple[object, ...]:
        return (path, mode, kwargs)

    fake.Dataset = fake_dataset  # type: ignore[attr-defined]
    monkeypatch.setattr("cds.data_io.streaming.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr("cds.data_io.streaming.importlib.import_module", lambda _name: fake)

    path = tmp_path / "data.nc"
    assert open_netcdf(path, format="NETCDF4") == (path, "r", {"format": "NETCDF4"})


def test_optional_backend_missing_has_clear_install_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("cds.data_io.streaming.importlib.util.find_spec", lambda _name: None)
    with pytest.raises(ModuleNotFoundError, match="install 'h5py'"):
        open_hdf5(tmp_path / "data.h5")
    with pytest.raises(ModuleNotFoundError, match="install 'netCDF4'"):
        open_netcdf(tmp_path / "data.nc")
