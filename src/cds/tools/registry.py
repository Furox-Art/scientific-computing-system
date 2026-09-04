"""Lazy discovery and loading of optional scientific backends."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from importlib import metadata
from types import ModuleType


@dataclass(frozen=True)
class ToolSpec:
    """Description of one optional scientific backend."""

    name: str
    module: str
    distribution: str
    capabilities: tuple[str, ...]
    purpose: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.module.strip() or not self.distribution.strip():
            raise ValueError("tool name, module, and distribution must not be empty")
        if not self.capabilities:
            raise ValueError("tool must declare at least one capability")


@dataclass(frozen=True)
class ToolStatus:
    """Installed/available state without importing the backend."""

    spec: ToolSpec
    available: bool
    version: str | None


class ToolRegistry:
    """Registry that keeps optional dependencies out of the CDS core import path."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a backend under a unique logical name."""
        if spec.name in self._specs:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    def spec(self, name: str) -> ToolSpec:
        """Return a registered backend specification."""
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def names(self) -> tuple[str, ...]:
        """Return registered tool names in deterministic order."""
        return tuple(sorted(self._specs))

    def status(self, name: str) -> ToolStatus:
        """Check import availability and package version without importing the tool."""
        spec = self.spec(name)
        available = importlib.util.find_spec(spec.module) is not None
        if not available:
            return ToolStatus(spec=spec, available=False, version=None)
        try:
            version = metadata.version(spec.distribution)
        except metadata.PackageNotFoundError:
            version = None
        return ToolStatus(spec=spec, available=True, version=version)

    def statuses(self) -> tuple[ToolStatus, ...]:
        """Probe every registered backend."""
        return tuple(self.status(name) for name in self.names())

    def recommend(self, capability: str, *, installed_only: bool = True) -> tuple[ToolStatus, ...]:
        """Return backends that advertise a capability, installed tools first."""
        matches = [
            self.status(name)
            for name in self.names()
            if capability in self._specs[name].capabilities
        ]
        if installed_only:
            matches = [status for status in matches if status.available]
        return tuple(sorted(matches, key=lambda status: (not status.available, status.spec.name)))

    def load(self, name: str) -> ModuleType:
        """Import an explicitly requested backend or raise a clear installation error."""
        status = self.status(name)
        if not status.available:
            raise ModuleNotFoundError(
                f"optional tool {name!r} is not installed; install its CDS extra or "
                f"the {status.spec.distribution!r} distribution"
            )
        return importlib.import_module(status.spec.module)


def default_registry() -> ToolRegistry:
    """Return CDS's standard optional scientific backend registry."""
    registry = ToolRegistry()
    for spec in (
        ToolSpec(
            name="numpy",
            module="numpy",
            distribution="numpy",
            capabilities=("arrays", "linear-algebra", "numerics"),
            purpose="vectorized arrays and dense numerical kernels",
        ),
        ToolSpec(
            name="scipy",
            module="scipy",
            distribution="scipy",
            capabilities=("optimization", "integration", "signal", "sparse", "statistics"),
            purpose="reference scientific algorithms and sparse numerical methods",
        ),
        ToolSpec(
            name="statsmodels",
            module="statsmodels",
            distribution="statsmodels",
            capabilities=("statistics", "regression", "time-series", "diagnostics"),
            purpose="statistical models, inference, and model diagnostics",
        ),
        ToolSpec(
            name="sklearn",
            module="sklearn",
            distribution="scikit-learn",
            capabilities=("machine-learning", "validation", "preprocessing"),
            purpose="machine-learning estimators and validation utilities",
        ),
        ToolSpec(
            name="sympy",
            module="sympy",
            distribution="sympy",
            capabilities=("symbolic", "algebra", "calculus", "verification"),
            purpose="symbolic mathematics and algebraic verification",
        ),
        ToolSpec(
            name="z3",
            module="z3",
            distribution="z3-solver",
            capabilities=("formal-verification", "constraints", "logic"),
            purpose="SMT-based logical and constraint verification",
        ),
    ):
        registry.register(spec)
    return registry
