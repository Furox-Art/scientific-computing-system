"""Smoke test for the benchmark suite.

Verifies structure (return types, keys) without timing the actual functions,
so CI stays fast. Timing correctness is validated manually via
``python benchmarks/run_benchmarks.py``.
"""

import os
from collections import OrderedDict
from importlib import import_module
from pathlib import Path
from typing import cast

import pytest


class TestBenchmarkStructure:
    """Each bench_* function must return OrderedDict[str, str]."""

    def _import_and_call(self, func_name: str, module_path: str) -> OrderedDict[str, str]:
        mod = import_module(module_path)
        return cast(OrderedDict[str, str], getattr(mod, func_name)())

    @pytest.mark.parametrize(
        "func, module",
        [
            ("bench_linalg", "benchmarks.run_benchmarks"),
            ("bench_linalg_intelligence", "benchmarks.run_benchmarks"),
            ("bench_montecarlo", "benchmarks.run_benchmarks"),
            ("bench_quantum", "benchmarks.run_benchmarks"),
            ("bench_signals", "benchmarks.run_benchmarks"),
            ("bench_numerical_integration", "benchmarks.run_benchmarks"),
        ],
    )
    def test_returns_ordered_dict_of_strings(self, func: str, module: str) -> None:
        result = self._import_and_call(func, module)
        assert isinstance(result, OrderedDict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    @pytest.mark.parametrize(
        "func, module, required_key",
        [
            ("bench_linalg", "benchmarks.run_benchmarks", "CDS Matrix Mul (100x100)"),
            ("bench_montecarlo", "benchmarks.run_benchmarks", "Pi (100k samples)"),
            ("bench_quantum", "benchmarks.run_benchmarks", "Intelligence Speedup"),
            ("bench_signals", "benchmarks.run_benchmarks", "Algorithmic speedup"),
            ("bench_numerical_integration", "benchmarks.run_benchmarks", "Romberg (auto tol)"),
        ],
    )
    def test_contains_expected_keys(self, func: str, module: str, required_key: str) -> None:
        result = self._import_and_call(func, module)
        assert required_key in result

    def test_run_all_generates_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_all writes a valid markdown report with all six sections.

        Output is redirected to ``tmp_path`` (``output_dir`` for the markdown,
        ``json_dir`` for ``results.json``) so the committed
        ``benchmarks/results.json`` and ``docs/benchmarks.md`` are never
        clobbered by a test run.
        """
        mod = import_module("benchmarks.run_benchmarks")
        monkeypatch.setattr(mod, "_bench", lambda _func, number=1, repeat=1: 0.0)
        mod.run_all(output_dir=tmp_path, json_dir=tmp_path)
        report = (tmp_path / "docs" / "benchmarks.md").read_text(encoding="utf-8")
        assert "# CDS Performance & Intelligence Report" in report
        assert "Signal Processing" in report
        assert "Numerical Integration" in report
        assert "Algorithmic Intelligence" in report
        assert "Quantum Intelligence" in report
        assert "Convergence" in report
        # results.json lands in json_dir (tmp_path here), not in benchmarks/.
        assert (tmp_path / "results.json").exists()

    def test_write_json_default_dir_is_benchmarks(self, tmp_path: Path) -> None:
        """_write_json defaults to the script's own dir (benchmarks/).

        Regression guard for the bug where ``run_all`` passed the repo root as
        the output dir, writing ``results.json`` to ``./results.json`` instead
        of the canonical ``benchmarks/results.json`` that the workflow diffs
        and the report references. The default ``json_dir=None`` MUST resolve
        to the directory containing ``run_benchmarks.py`` (``benchmarks/``),
        independent of the current working directory.

        The committed ``benchmarks/results.json`` is backed up and restored so
        this test leaves the working tree clean.
        """
        mod = import_module("benchmarks.run_benchmarks")
        # __file__ is typed as str | None (a module could be a namespace pkg),
        # but run_benchmarks.py is a regular file module — assert for mypy,
        # which then narrows it to str with no cast needed.
        assert mod.__file__ is not None
        expected_dir = Path(mod.__file__).resolve().parent
        canonical = expected_dir / "results.json"
        backup = canonical.read_bytes() if canonical.exists() else None

        prev = os.getcwd()
        os.chdir(tmp_path)
        try:
            out = mod._write_json({"git_sha": "deadbeef", "metrics": {}})
        finally:
            os.chdir(prev)
            if backup is not None:
                canonical.write_bytes(backup)
        assert out == canonical
        # The cwd (tmp_path) must NOT have received a stray results.json.
        assert not (tmp_path / "results.json").exists()
