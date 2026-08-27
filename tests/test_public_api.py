"""Public API guard: every ``__all__`` entry must actually resolve.

A name listed in ``__all__`` but never imported into the package body is
invisible to the test suite whenever tests import from the *submodule*
directly (``from cds.stats.bootstrap import bootstrap_ci``) instead of the
package (``from cds.stats import bootstrap_ci``). The declared public API
then silently diverges from the importable one:

    from cds.stats import bootstrap_ci   -> ImportError
    from cds.stats import *              -> AttributeError

That is exactly how ``cds.stats`` shipped ``BootstrapResult`` /
``bootstrap_ci`` / ``bootstrap_diff_ci`` in ``__all__`` without importing
them. This module walks every public ``cds`` module and pins the invariant
so the whole class of mistake fails CI instead of reaching users.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import cds


def _public_modules() -> list[str]:
    """Every importable ``cds`` module whose path has no private component."""
    names = ["cds"]
    for info in pkgutil.walk_packages(cds.__path__, prefix="cds."):
        parts = info.name.split(".")
        if any(part.startswith("_") for part in parts):
            continue
        names.append(info.name)
    return sorted(names)


PUBLIC_MODULES = _public_modules()


def test_public_modules_were_discovered() -> None:
    """Guard the guard: a broken walk must not vacuously pass the suite."""
    assert len(PUBLIC_MODULES) > 50
    assert "cds.stats" in PUBLIC_MODULES


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_all_entries_resolve(module_name: str) -> None:
    """Each name in ``__all__`` must be an attribute of its own module."""
    module = importlib.import_module(module_name)
    declared = getattr(module, "__all__", None)
    if declared is None:
        pytest.skip(f"{module_name} declares no __all__")
    missing = [name for name in declared if not hasattr(module, name)]
    assert not missing, f"{module_name}.__all__ names nothing: {missing}"


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_star_import_succeeds(module_name: str) -> None:
    """``from <module> import *`` must not raise on a declared ``__all__``."""
    module = importlib.import_module(module_name)
    if getattr(module, "__all__", None) is None:
        pytest.skip(f"{module_name} declares no __all__")
    namespace: dict[str, object] = {}
    exec(f"from {module_name} import *", namespace)  # noqa: S102


def test_stats_reexports_bootstrap() -> None:
    """Regression pin for the ``cds.stats`` bootstrap re-export gap."""
    from cds.stats import BootstrapResult, bootstrap_ci, bootstrap_diff_ci

    assert BootstrapResult.__module__ == "cds.stats.bootstrap"
    assert bootstrap_ci.__module__ == "cds.stats.bootstrap"
    assert bootstrap_diff_ci.__module__ == "cds.stats.bootstrap"
