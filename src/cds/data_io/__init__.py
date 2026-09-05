"""Streaming file I/O and lazy optional scientific data backends."""

from cds.data_io.streaming import (
    FileProfile,
    OnlineMoments,
    StreamingLinearAccumulator,
    StreamingLinearFit,
    fit_linear_csv_streaming,
    iter_csv_batches,
    iter_file_blocks,
    open_hdf5,
    open_netcdf,
    profile_file,
)

__all__ = [
    "FileProfile",
    "OnlineMoments",
    "StreamingLinearAccumulator",
    "StreamingLinearFit",
    "fit_linear_csv_streaming",
    "iter_csv_batches",
    "iter_file_blocks",
    "open_hdf5",
    "open_netcdf",
    "profile_file",
]
