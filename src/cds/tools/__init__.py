"""Capability discovery and adapters for optional scientific computing backends."""

from cds.tools.adapters import (
    OptimizationResult,
    Satisfiability,
    scipy_minimize,
    sympy_verify_identity,
    z3_satisfiability,
)
from cds.tools.registry import (
    ToolLocality,
    ToolRegistry,
    ToolSpec,
    ToolStatus,
    default_registry,
)

__all__ = [
    "OptimizationResult",
    "Satisfiability",
    "ToolLocality",
    "ToolRegistry",
    "ToolSpec",
    "ToolStatus",
    "default_registry",
    "scipy_minimize",
    "sympy_verify_identity",
    "z3_satisfiability",
]
