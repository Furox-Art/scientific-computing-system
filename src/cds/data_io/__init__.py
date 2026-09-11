"""Streaming file I/O and lazy optional scientific data backends."""

from cds.data_io.scientific import (
    ArrayChunk,
    ScientificArrayProfile,
    iter_array_chunks,
    iter_hdf5_chunks,
    iter_netcdf_chunks,
    profile_scientific_array,
    reduce_chunks,
)
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
    "ArrayChunk",
    "FileProfile",
    "OnlineMoments",
    "ScientificArrayProfile",
    "StreamingLinearAccumulator",
    "StreamingLinearFit",
    "fit_linear_csv_streaming",
    "iter_array_chunks",
    "iter_csv_batches",
    "iter_file_blocks",
    "iter_hdf5_chunks",
    "iter_netcdf_chunks",
    "open_hdf5",
    "open_netcdf",
    "profile_file",
    "profile_scientific_array",
    "reduce_chunks",
]
