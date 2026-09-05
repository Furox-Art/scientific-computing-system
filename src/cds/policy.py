"""Shared execution-location and data-handling policy primitives."""

from __future__ import annotations

from enum import Enum


class ExecutionLocation(str, Enum):
    """Where a scientific method or tool executes."""

    LOCAL = "local"
    REMOTE = "remote"


class DataHandling(str, Enum):
    """How a method or tool treats input data."""

    LOCAL_ONLY = "local_only"
    NO_RETENTION = "no_retention"
    EXTERNAL_UNSPECIFIED = "external_unspecified"
