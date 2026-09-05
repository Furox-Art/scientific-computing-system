"""Architectural layering guard.

Enforces package dependency directions with a plain AST walk.  The modeling
layer is explicitly allowed to depend on ``units`` because dimensional
contracts are now part of scientific model/fitting validation.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "cds"

_ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "math_utils": {"core"},
    "probability": {"core", "math_utils"},
    "optimization": {"core", "math_utils"},
    "signals": {"core", "math_utils", "probability"},
    "scientific": {"core"},
    "graph": {"core"},
    "quantum": {"core"},
    "montecarlo": {"core", "math_utils", "probability"},
    "numerical_integration": {"core", "math_utils"},
    "diffeq": {"core", "math_utils", "optimization"},
    "stats": {"core", "math_utils", "probability"},
    "ml": {"core", "math_utils", "optimization", "stats", "probability"},
    "data_analysis": {"core", "math_utils", "probability", "stats"},
    "hypothesis": {"core", "stats"},
    "knowledge": set(),
    "modeling": {"core", "optimization", "units"},
    "nlp": {"core", "math_utils"},
    "plot": {"core", "math_utils", "signals", "stats"},
    "cli": {
        "<root>",
        "core",
        "data_analysis",
        "hypothesis",
        "numerical_integration",
        "plot",
        "probability",
        "scientific",
        "stats",
    },
}


def _cds_imports(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cds" or alias.name.startswith("cds."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module == "cds" or module.startswith("cds.")):
                found.add(module)
    return found


def test_no_layering_violations() -> None:
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        parts = rel.parts
        pkg = parts[0] if len(parts) > 1 else ""
        if pkg not in _ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _cds_imports(tree):
            target = imported.split(".")[1] if "." in imported else "<root>"
            if target != pkg and target not in _ALLOWED[pkg]:
                violations.append(f"cds/{rel} -> {imported}")
    assert not violations, "layering violations:\n" + "\n".join(violations)


def test_probability_does_not_import_stats() -> None:
    for path in (SRC / "probability").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bad = [m for m in _cds_imports(tree) if m.startswith("cds.stats")]
        assert not bad, f"{path.name} imports {bad}"
