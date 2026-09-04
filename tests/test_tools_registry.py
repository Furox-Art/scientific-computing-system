"""Tests for optional scientific backend capability discovery."""

from __future__ import annotations

from importlib import metadata
from types import ModuleType

import pytest

from cds.tools import ToolRegistry, ToolSpec, default_registry


def _spec(name: str = "demo") -> ToolSpec:
    return ToolSpec(
        name=name,
        module=f"{name}_module",
        distribution=f"{name}-dist",
        capabilities=("numerics",),
        purpose="test backend",
    )


def test_tool_spec_and_registry_validation() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ToolSpec("", "module", "dist", ("x",), "purpose")
    with pytest.raises(ValueError, match="must not be empty"):
        ToolSpec("name", "", "dist", ("x",), "purpose")
    with pytest.raises(ValueError, match="must not be empty"):
        ToolSpec("name", "module", "", ("x",), "purpose")
    with pytest.raises(ValueError, match="capability"):
        ToolSpec("name", "module", "dist", (), "purpose")

    registry = ToolRegistry()
    registry.register(_spec())
    assert registry.names() == ("demo",)
    assert registry.spec("demo").purpose == "test backend"

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_spec())
    with pytest.raises(KeyError, match="unknown tool"):
        registry.spec("missing")


def test_status_available_unavailable_and_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry()
    registry.register(_spec("missing"))
    registry.register(_spec("installed"))
    registry.register(_spec("noversion"))

    def fake_find_spec(module: str) -> object | None:
        return None if module == "missing_module" else object()

    def fake_version(distribution: str) -> str:
        if distribution == "noversion-dist":
            raise metadata.PackageNotFoundError(distribution)
        return "1.2.3"

    monkeypatch.setattr("cds.tools.registry.importlib.util.find_spec", fake_find_spec)
    monkeypatch.setattr("cds.tools.registry.metadata.version", fake_version)

    missing = registry.status("missing")
    assert not missing.available
    assert missing.version is None

    installed = registry.status("installed")
    assert installed.available
    assert installed.version == "1.2.3"

    noversion = registry.status("noversion")
    assert noversion.available
    assert noversion.version is None

    statuses = registry.statuses()
    assert tuple(status.spec.name for status in statuses) == ("installed", "missing", "noversion")


def test_recommend_filters_and_orders_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec("b", "b_module", "b-dist", ("fit",), "b"))
    registry.register(ToolSpec("a", "a_module", "a-dist", ("fit", "other"), "a"))
    registry.register(ToolSpec("c", "c_module", "c-dist", ("other",), "c"))

    monkeypatch.setattr(
        "cds.tools.registry.importlib.util.find_spec",
        lambda module: object() if module == "b_module" else None,
    )
    monkeypatch.setattr("cds.tools.registry.metadata.version", lambda _distribution: "1")

    installed = registry.recommend("fit")
    assert tuple(status.spec.name for status in installed) == ("b",)

    all_matches = registry.recommend("fit", installed_only=False)
    assert tuple(status.spec.name for status in all_matches) == ("b", "a")

    assert registry.recommend("unknown") == ()


def test_load_requires_installation_and_imports_on_explicit_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry()
    registry.register(_spec("backend"))

    monkeypatch.setattr("cds.tools.registry.importlib.util.find_spec", lambda _module: None)
    with pytest.raises(ModuleNotFoundError, match="optional tool"):
        registry.load("backend")

    module = ModuleType("backend_module")
    monkeypatch.setattr("cds.tools.registry.importlib.util.find_spec", lambda _module: object())
    monkeypatch.setattr("cds.tools.registry.metadata.version", lambda _distribution: "2.0")
    monkeypatch.setattr("cds.tools.registry.importlib.import_module", lambda _module: module)
    assert registry.load("backend") is module


def test_default_registry_declares_scientific_backends() -> None:
    registry = default_registry()
    assert registry.names() == (
        "h5py",
        "netcdf4",
        "numpy",
        "scipy",
        "sklearn",
        "statsmodels",
        "sympy",
        "z3",
    )
    assert "optimization" in registry.spec("scipy").capabilities
    assert "formal-verification" in registry.spec("z3").capabilities
    assert "hdf5" in registry.spec("h5py").capabilities
    assert "netcdf" in registry.spec("netcdf4").capabilities
