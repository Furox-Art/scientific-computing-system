"""Capability discovery for optional scientific computing backends."""

from cds.tools.registry import (
    ToolRegistry,
    ToolSpec,
    ToolStatus,
    default_registry,
)

__all__ = ["ToolRegistry", "ToolSpec", "ToolStatus", "default_registry"]
