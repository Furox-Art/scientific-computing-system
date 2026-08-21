"""Architectural layering guard.

Enforces the package dependency directions with a plain AST walk — no extra
tooling, runs anywhere pytest does. Every subpackage is constrained to its
measured import surface; growing one requires a conscious edit here.

    core            → (nothing)
    math_utils      → core
    probability     → core, math_utils
    optimization    → core, math_utils
    signals         → core, math_utils, probability
    stats           → core, math_utils, probability
    diffeq          → core, math_utils, optimization
    ml              → core, math_utils, optimization, stats, probability
    data_analysis   → core, math_utils, probability, stats
    hypothesis      → core, stats
    knowledge       → (nothing)
    modeling        → core, optimization
    nlp             → core, math_utils
    plot            → core, math_utils, signals, stats
    cli             → today's orchestrator surface (pinned)

The guard fails the suite if a new import ever inverts these directions —
e.g. ``probability`` reaching back into ``stats`` was exactly the violation
this test exists to prevent.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "cds"

# Allowed cds.* dependencies per top-level subpackage. Absence from the map
# means "may import any other cds subpackage" (high-level orchestration code).
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
    # Orchestrator/domain packages — constrained to their measured imports.
    "data_analysis": {"core", "math_utils", "probability", "stats"},
    "hypothesis": {"core", "stats"},
    "knowledge": set(),
    "modeling": {"core", "optimization"},
    "nlp": {"core", "math_utils"},
    "plot": {"core", "math_utils", "signals", "stats"},
    # The CLI is the top orchestrator; the entry pins today's surface and
    # grows only consciously. ``<root>`` allows ``from cds import __version__``.
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
    """Collect every ``cds.X`` / ``cds`` root imported by one module."""
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
    """Every low-level subpackage must only import its allowed set."""
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        parts = rel.parts
        # Top-level modules like cli.py / __init__.py are orchestrators.
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
    """Regression pin: probability must stay below stats in the layer stack."""
    for path in (SRC / "probability").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bad = [m for m in _cds_imports(tree) if m.startswith("cds.stats")]
        assert not bad, f"{path.name} imports {bad}"
