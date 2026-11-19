"""Smoke test for the benchmark suite.

Verifies structure (return types, keys) without timing the actual functions,
so CI stays fast. Timing correctness is validated manually via
``python benchmarks/run_benchmarks.py``.
"""

import json
import os
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[1]


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
            ("bench_quantum", "benchmarks.run_benchmarks", "Speedup vs extrapolation"),
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
        assert "# CDS Performance Report" in report
        assert "Signal Processing" in report
        assert "Numerical Integration" in report
        assert "Extrapolated Brute Force" in report
        assert "Visual comparison: quantum sampling" in report
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

    def test_run_all_splices_between_markers_preserving_prose(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A curated page keeps its hand-written half; only the block is replaced.

        The old writer overwrote the whole file, which would have destroyed the
        provenance/analysis sections on the next scheduled regeneration.
        """
        mod = import_module("benchmarks.run_benchmarks")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        page = docs_dir / "benchmarks.md"
        page.write_text(
            "# CDS Performance Report\n\nHAND-WRITTEN PROVENANCE\n\n"
            + mod.BEGIN_GENERATED
            + "\nSTALE BLOCK\n"
            + mod.END_GENERATED
            + "\n\nHAND-WRITTEN ANALYSIS\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "_bench", lambda _func, number=1, repeat=1: 0.0)
        mod.run_all(output_dir=tmp_path, json_dir=tmp_path)
        result = page.read_text(encoding="utf-8")
        assert "HAND-WRITTEN PROVENANCE" in result
        assert "HAND-WRITTEN ANALYSIS" in result
        assert "STALE BLOCK" not in result
        assert "Linear Algebra (Pure Python vs NumPy)" in result
        assert result.count(mod.BEGIN_GENERATED) == 1
        assert result.count(mod.END_GENERATED) == 1

    def test_run_all_without_markers_writes_full_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An existing page without markers is replaced by the full report."""
        mod = import_module("benchmarks.run_benchmarks")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        page = docs_dir / "benchmarks.md"
        page.write_text("legacy page without markers\n", encoding="utf-8")
        monkeypatch.setattr(mod, "_bench", lambda _func, number=1, repeat=1: 0.0)
        mod.run_all(output_dir=tmp_path, json_dir=tmp_path)
        result = page.read_text(encoding="utf-8")
        assert result.startswith("# CDS Performance Report")
        assert "legacy page without markers" not in result


class TestCommittedArtifactHonesty:
    """Pin the committed artifact to the data and the docs block to the artifact.

    The committed results.json used to carry marketing headers that the data
    contradicted ("Approaching C-Speed" above a 1154.8x-slower measurement,
    "CPU Cores Saturated" that was a bare cpu_count(), an "Intelligence
    Speedup" extrapolated from a single circuit run), and the docs page's
    "written verbatim" block had been hand-edited away from the JSON. These
    tests make both failures impossible to reintroduce silently.
    """

    BANNED = (
        "Approaching C-Speed",
        "Hardware Saturation",
        "Algorithmic Intelligence",
        "Intelligence Speedup",
        "CPU Cores Saturated",
        "Parallel Pi",
    )

    def _artifact(self) -> dict[str, object]:
        payload = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))
        assert isinstance(payload, dict)  # noqa: S101 - JSON object contract
        return payload

    def _generated_block(self) -> str:
        mod = import_module("benchmarks.run_benchmarks")
        text = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
        begin = text.index(mod.BEGIN_GENERATED) + len(mod.BEGIN_GENERATED)
        end = text.index(mod.END_GENERATED)
        return text[begin:end]

    def _tables(self, block: str) -> dict[str, list[tuple[str, str]]]:
        sections: dict[str, list[tuple[str, str]]] = {}
        current: str | None = None
        for line in block.splitlines():
            if line.startswith("#### "):
                current = line[len("#### ") :]
                sections[current] = []
            elif line.startswith("| ") and current is not None:
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if cells[0] == "Metric":
                    continue
                assert len(cells) == 2, f"malformed table row: {line!r}"  # noqa: S101
                sections[current].append((cells[0], cells[1]))
        return sections

    def test_no_marketing_headers_in_the_artifact(self) -> None:
        artifact = self._artifact()
        metrics = artifact["metrics"]
        assert isinstance(metrics, dict)  # noqa: S101 - JSON object contract
        labels = list(metrics)
        for rows in metrics.values():
            assert isinstance(rows, dict)  # noqa: S101 - JSON object contract
            labels.extend(rows)
        for label in labels:
            for banned in self.BANNED:
                assert banned not in label, f"marketing label survived: {label!r}"

    def test_docs_generated_block_matches_json_row_for_row(self) -> None:
        """The 'verbatim' claim must be true: every JSON metric, same order/value."""
        mod = import_module("benchmarks.run_benchmarks")
        coerce = cast("Callable[[str], float | str]", mod._coerce_metric)
        artifact = self._artifact()
        metrics = artifact["metrics"]
        assert isinstance(metrics, dict)  # noqa: S101 - JSON object contract
        tables = self._tables(self._generated_block())
        visual = "Visual comparison: quantum sampling (extrapolated)"
        assert set(tables) == set(metrics) | {visual}, (
            f"docs sections {sorted(tables)} differ from JSON categories {sorted(metrics)}"
        )
        for category, rows in metrics.items():
            assert isinstance(rows, dict)  # noqa: S101 - JSON object contract
            documented = tables[category]
            assert [key for key, _ in documented] == list(rows), (
                f"{category}: docs rows differ from the JSON metrics"
            )
            for (key, displayed), expected in zip(documented, rows.values()):
                assert coerce(displayed) == expected, (
                    f"{category} / {key}: docs shows {displayed!r}, JSON records {expected!r}"
                )

    def test_docs_block_provenance_matches_json(self) -> None:
        artifact = self._artifact()
        block = self._generated_block()
        sha = artifact["git_sha"]
        stamp = datetime.fromisoformat(str(artifact["timestamp_utc"])).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        assert f"commit `{sha}` at {stamp}" in block

    def test_quantum_speedup_is_one_consistent_extrapolation(self) -> None:
        """Table and conclusion quote the same ratio, derived from displayed times."""
        artifact = self._artifact()
        metrics = artifact["metrics"]
        assert isinstance(metrics, dict)  # noqa: S101 - JSON object contract
        quantum = metrics["Quantum (O(1) Sampling vs Extrapolated Brute Force)"]
        assert isinstance(quantum, dict)  # noqa: S101 - JSON object contract
        measured = float(quantum["O(1) Sampling (measured)"])
        extrapolated = float(quantum["Naive Brute Force (extrapolated)"])
        expected = round(extrapolated / measured, 1)
        assert quantum["Speedup vs extrapolation"] == pytest.approx(expected)
        block = self._generated_block()
        assert f"({extrapolated:.2f}s)" in block
        assert f"({measured:.4f}s)" in block
        assert f"~{expected:.1f}x" in block
        assert "extrapolated, not measured" in block
