"""C-3: ``cds.pde`` and ``cds.interpolate`` must be importable as ``cds`` attributes.

``docs/index.md`` advertises ``cds.interpolate`` and ``cds.pde`` in the module
table and ``docs/api.md`` renders both with ``::: cds.pde`` / ``::: cds.interpolate``.
The flat package layout meant ``import cds`` never bound those attributes, so
``import cds; cds.pde`` raised ``AttributeError`` while the documentation
promised it worked.

These tests assert identity — the ``cds`` attribute is the documented module
object itself — plus a value from each module's public API, so a stub or a
differently-named object cannot satisfy them.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

import cds

DOCUMENTED_MODULES = ("pde", "interpolate")


class TestLazySubpackageExposure:
    @pytest.mark.parametrize("module_name", DOCUMENTED_MODULES)
    def test_attribute_exists_without_eager_import(self, module_name: str) -> None:
        """``cds.pde`` resolves on attribute access, not only via ``import cds.pde``."""
        assert hasattr(cds, module_name)
        resolved = getattr(cds, module_name)
        assert resolved.__name__ == f"cds.{module_name}"

    @pytest.mark.parametrize("module_name", DOCUMENTED_MODULES)
    def test_attribute_is_the_documented_module(self, module_name: str) -> None:
        """The attribute must be the very object ``import cds.pde`` yields."""
        expected = importlib.import_module(f"cds.{module_name}")
        assert getattr(cds, module_name) is expected

    @pytest.mark.parametrize("module_name", DOCUMENTED_MODULES)
    def test_attribute_is_cached_after_first_lookup(
        self, module_name: str, tmp_path: object
    ) -> None:
        """The lazy import must resolve once, not on every attribute lookup.

        Run in a subprocess so the answer reflects a cold interpreter: a
        plain ``import cds`` must *not* have pulled the subpackage in yet,
        and accessing it once must bind it into ``cds.__dict__``.
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        script = (
            "import cds, sys\n"
            f"name = {module_name!r}\n"
            "qualified = 'cds.' + name\n"
            "print(name in vars(cds), qualified in sys.modules)\n"
            "getattr(cds, name)\n"
            "print(name in vars(cds), qualified in sys.modules)\n"
            "print(getattr(cds, name) is getattr(cds, name))\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root / "src")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = completed.stdout.strip().splitlines()
        assert lines[0] == "False False", (
            f"cds.{module_name} was imported eagerly; lazy loading is not in effect"
        )
        assert lines[1] == "True True", (
            f"accessing cds.{module_name} did not import and bind the subpackage"
        )
        assert lines[2] == "True"

    @pytest.mark.parametrize("module_name", DOCUMENTED_MODULES)
    def test_documented_public_api_is_reachable(self, module_name: str) -> None:
        """Each module's ``__all__`` resolves through the ``cds.<name>`` attribute."""
        module = getattr(cds, module_name)
        declared = list(getattr(module, "__all__", ()))
        assert declared, f"cds.{module_name} declares no public API"
        for exported in declared:
            assert getattr(module, exported) is not None

    @pytest.mark.parametrize("module_name", DOCUMENTED_MODULES)
    def test_from_import_and_star_import(self, module_name: str) -> None:
        """``from cds import <name>`` and ``from cds import *`` both expose it."""
        namespace: dict[str, ModuleType] = {}
        exec(f"from cds import {module_name}", namespace)  # noqa: S102
        assert namespace[module_name].__name__ == f"cds.{module_name}"

        star_namespace: dict[str, ModuleType] = {}
        exec("from cds import *", star_namespace)  # noqa: S102
        assert module_name in star_namespace
        assert star_namespace[module_name].__name__ == f"cds.{module_name}"
        assert module_name in cds.__all__

    def test_unknown_attribute_still_raises(self) -> None:
        """The lazy loader must not swallow genuine AttributeErrors."""
        with pytest.raises(AttributeError, match="no attribute"):
            getattr(cds, "definitely_not_a_subpackage")

    def test_dir_lists_the_lazy_subpackages(self) -> None:
        for module_name in DOCUMENTED_MODULES:
            assert module_name in dir(cds)


class TestDocumentedValues:
    """The exposed modules must return their documented numbers, not just exist."""

    def test_interp1d_endpoint_and_midpoint(self) -> None:
        interpolate = cds.interpolate
        curve = interpolate.interp1d([0.0, 1.0, 2.0], [0.0, 10.0, 20.0], kind="linear")
        # Endpoint knots reproduce the samples exactly.
        assert curve(0.0) == 0.0
        assert curve(2.0) == 20.0
        # The midpoint of the first segment is the average of its endpoints.
        assert curve(0.5) == pytest.approx(5.0)

    def test_interp2d_bilinear_blend(self) -> None:
        interpolate = cds.interpolate
        grid = interpolate.interp2d([0.0, 1.0], [0.0, 1.0], [[0.0, 1.0], [2.0, 3.0]])
        # z[i][j] at the corners; the centre of the unit square averages to 1.5.
        assert grid(0.0, 0.0) == 0.0
        assert grid(1.0, 1.0) == 3.0
        assert grid(0.5, 0.5) == pytest.approx(1.5)

    def test_pde_heat_energy_decays(self) -> None:
        pde = cds.pde
        result = pde.solve_heat([1.0, 0.0, 0.0], alpha=0.1, length=1.0, t_final=0.05, nx=3)
        # Dirichlet-pinned ends stay at their initial level; the interior
        # relaxes toward them, so the total energy can only fall.
        assert result.u_final[0] == 1.0
        assert result.u_final[-1] == 0.0
        assert 0.0 <= result.u_final[1] <= 1.0
        assert result.n_steps >= 1

    def test_pde_wave_amplitude_stays_bounded(self) -> None:
        pde = cds.pde
        result = pde.solve_wave(
            [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], c=1.0, length=1.0, t_final=0.05, nx=3
        )
        assert len(result.u_final) == 3
        assert all(abs(value) <= 1.0 + 1e-9 for value in result.u_final)
