from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from cds.data_io import iter_hdf5_chunks, iter_netcdf_chunks


def _tolist(value: object) -> list[list[float]]:
    converted = cast(Any, value).tolist()
    return [[float(item) for item in row] for row in converted]


def test_real_hdf5_chunk_pipeline(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "measurements.h5"
    with cast(Any, h5py).File(path, "w") as handle:
        handle.create_dataset(
            "measurements",
            data=[[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]],
            chunks=(2, 2),
        )

    chunks = list(iter_hdf5_chunks(path, "measurements", chunk_size=2))
    assert [(chunk.start, chunk.stop) for chunk in chunks] == [(0, 2), (2, 4), (4, 5)]
    assert _tolist(chunks[0].data) == [[0.0, 10.0], [1.0, 11.0]]
    assert _tolist(chunks[-1].data) == [[4.0, 14.0]]


def test_real_netcdf_chunk_pipeline(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "measurements.nc"
    with cast(Any, netcdf4).Dataset(path, "w") as handle:
        handle.createDimension("row", 5)
        handle.createDimension("column", 2)
        variable = handle.createVariable("temperature", "f8", ("row", "column"))
        variable[:] = [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]]

    chunks = list(iter_netcdf_chunks(path, "temperature", chunk_size=3))
    assert [(chunk.start, chunk.stop) for chunk in chunks] == [(0, 3), (3, 5)]
    assert _tolist(chunks[0].data) == [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]]
    assert _tolist(chunks[-1].data) == [[3.0, 13.0], [4.0, 14.0]]
