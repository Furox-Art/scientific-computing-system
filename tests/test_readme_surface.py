"""Pin the README's structural claims to the package as it actually is.

The module-surface text in ``README.md`` drifted out of sync with ``src/cds/``
for several releases without CI noticing: the README said "34 subpackages", which
silently omitted the two top-level public modules (``causal.py``,
``sensitivity.py``), and it described ``cds modules`` as a "live list" when that
command prints a curated subset. Nothing failed, because nothing compared the
prose to the package.

These checks derive the numbers from ``src/cds/`` and from the installed package
itself, so a new module, a renamed export, or a changed count turns the build red
instead of quietly making the README wrong.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# Modules that are package surface but not importable as `cds.<name>` subpackages.
TOP_LEVEL_PUBLIC = ("causal", "sensitivity")
# The `cds` entry point: a console script, not a feature module.
NOT_FEATURE_MODULES = frozenset({"cli", "_paths"})


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _subpackages() -> list[str]:
    root = ROOT / "src" / "cds"
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "__pycache__")


def _feature_modules() -> list[str]:
    names = set(_subpackages()) | set(TOP_LEVEL_PUBLIC)
    return sorted(names - NOT_FEATURE_MODULES)


def _export_counts() -> tuple[int, int]:
    """Return (functions, classes) across every feature module's ``__all__``."""
    functions = classes = 0
    for name in _feature_modules():
        module = importlib.import_module(f"cds.{name}")
        for exported in getattr(module, "__all__", None) or []:
            obj = getattr(module, exported, None)
            if inspect.isclass(obj):
                classes += 1
            elif callable(obj):
                functions += 1
    return functions, classes


def test_readme_subpackage_count_matches_src() -> None:
    """The subpackage count must come from src/cds, not from prose."""
    expected = len(_subpackages())
    match = re.search(r"\*\*(\d+) subpackages", _readme())
    assert match is not None, "README no longer states a subpackage count"
    assert int(match.group(1)) == expected, (
        f"README says {match.group(1)} subpackages but src/cds/ has {expected}"
    )


def test_readme_names_the_two_top_level_public_modules() -> None:
    """A bare subpackage count hides top-level public modules; README must say so."""
    readme = _readme()
    for name in TOP_LEVEL_PUBLIC:
        assert f"{name}.py" in readme, f"README omits the public module {name}.py"


def test_readme_export_count_matches_package() -> None:
    """The advertised export total must equal what the package actually exports."""
    functions, classes = _export_counts()
    total = functions + classes
    expected = f"**{total} names** ({functions} functions, {classes} classes)"
    assert expected in _readme(), (
        f"README does not carry the verified export line {expected!r}; "
        f"package exports {functions} functions and {classes} classes"
    )


def test_readme_module_table_covers_every_feature_module() -> None:
    """Every public feature module must appear in the README's area table.

    Matching is boundary-aware on purpose. A plain substring test would accept
    ``cds.graphX`` as evidence that ``cds.graph`` is documented, which is exactly
    the silent-drift failure this file exists to prevent.
    """
    readme = _readme()
    missing = []
    for name in _feature_modules():
        pattern = rf"(?<![\w.])cds\.{re.escape(name)}(?![\w])"
        if re.search(pattern, readme) is None:
            missing.append(name)
    assert not missing, f"README module table omits: {missing}"


def test_readme_does_not_call_cds_modules_a_live_list() -> None:
    """`cds modules` is a curated subset. Calling it live or authoritative is wrong."""
    readme = _readme()
    for phrase in ("live module catalog", "live list", "authoritative if this page"):
        assert phrase not in readme, f"README still calls `cds modules` exhaustive: {phrase!r}"


def test_readme_benchmark_claim_matches_the_committed_artifact() -> None:
    """The performance figure must match benchmarks/results.json, and be hedged."""
    import json

    artifact = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))
    linear = artifact["metrics"]["Linear Algebra (Approaching C-Speed)"]
    status = str(linear["Speed Status"])
    match = re.search(r"CDS is ([\d.]+)x slower than NumPy", status)
    assert match is not None, f"unparsable Speed Status: {status!r}"

    readme = _readme()
    ratio = float(match.group(1))
    # ~1155x is the rounded form of 1154.8; accept a small rounding window only.
    claimed = re.search(r"about \*\*(\d+)× slower\*\*", readme)
    assert claimed is not None, "README no longer carries a benchmark ratio claim"
    assert abs(int(claimed.group(1)) - round(ratio / 5) * 5) <= 5, (
        f"README claims ~{claimed.group(1)}x but results.json says {ratio}x"
    )

    # The artifact records no environment, so the README must not imply it does.
    assert "not recorded" in (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8"), (
        "benchmarks.md lost its 'not recorded' provenance warning; the README's "
        "hedged wording would then be unsupported"
    )
    assert "were not recorded" in readme, (
        "README must state that the benchmark environment was not recorded"
    )


def test_readme_pins_no_hand_written_download_metric() -> None:
    """Belt-and-braces: no invented popularity numbers creep into the prose."""
    match = re.search(r"\d[\d,._]*\s*(?:/month|/day|/week|downloads?\b)", _readme(), re.IGNORECASE)
    assert match is None, f"README carries a download metric: {match.group(0)!r}"
