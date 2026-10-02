"""Regression guard for the shipped runtime dependency graph.

CDS sells a zero-runtime-dependency core: ``pip install scientific-computing-system``
must resolve to the package plus ``pip`` and nothing else. That property is the
foundation of the project's supply-chain posture -- every advisory-free claim
about the core rests on the unconditional graph staying empty, with all third
party science stacks reachable only through extras.

An empty graph is easy to break silently. Adding one unconditional entry to
``[project] dependencies`` pulls it into every downstream install with no
upper bound, and no existing test would notice: the CI audit jobs resolve
extras environments, and ``tests/test_release_contract.py`` only covers
workflow wiring. These tests make the invariant fail loudly instead.

What is asserted:

* ``[project] dependencies`` stays empty;
* the runtime ``requirements.txt`` mirror declares no package;
* a package with an open advisory never appears outside the
  ``[project.optional-dependencies]`` table, so it cannot reach a plain install;
* the built wheel emits no unconditional ``Requires-Dist``;
* importing ``cds`` pulls in no third-party distribution.

``pyproject.toml`` is read with the same regex approach as
``tests/test_release_contract.py`` rather than :mod:`tomllib`: the project's
mypy target is ``python_version = "3.10"`` and ``tomllib`` is 3.11+, so a
``tomllib`` import would fail ``mypy --strict`` on the tests target.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYPROJECT = ROOT / "pyproject.toml"

# Packages with open advisories. Each belongs to a dev/optional extra and must
# never appear outside ``[project.optional-dependencies]``. Kept as data so a
# future advisory adds one line here instead of a new bespoke assertion.
VULNERABLE_IF_UNCONDITIONAL = frozenset({"jupyterlab", "streamlit"})

_REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][\w.-]*)(?=[^\S\n]*(?:[<>=!~;\[]|$))")


def _pyproject() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def _table(text: str, header: str) -> str:
    """Return the body of a TOML table, up to the next table header."""
    match = re.search(
        rf"^\[{re.escape(header)}\][^\S\n]*$(.*?)(?=^\[|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"pyproject.toml is missing the [{header}] table"
    return match.group(1)


def _string_array(body: str, key: str) -> list[str]:
    """Return the string literals of an inline ``key = ["a", "b"]`` array."""
    match = re.search(
        rf"^{re.escape(key)}[^\S\n]*=[^\S\n]*\[(.*?)\]",
        body,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing inline array for {key!r}"
    return re.findall(r'"([^"]+)"', match.group(1))


def _requirement_name(requirement: str) -> str | None:
    """Normalize a requirement string to its distribution name, or ``None``.

    ``None`` means the string is not requirement-shaped -- ``">=3.10"`` from
    ``requires-python``, or a description field. Only requirement-shaped
    strings can name a distribution, so anything else is skipped rather than
    failing the guard.
    """
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        return None
    return match.group(1).lower().replace("_", "-")


def _src_env() -> dict[str, str]:
    """A child-process env with ``src`` importable, mirroring pytest's pythonpath."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), existing]) if existing else str(SRC)
    return env


def test_declared_runtime_dependencies_are_empty() -> None:
    body = _table(_pyproject(), "project")
    assert _string_array(body, "dependencies") == []


def test_runtime_requirements_mirror_declares_no_package() -> None:
    mirror = ROOT / "requirements.txt"
    if not mirror.is_file():
        # The sdist allowlist ships the lock files but not this convenience
        # mirror. The wheel-METADATA test below still guards the shipped
        # artifact, so there is nothing to assert against here.
        pytest.skip("requirements.txt is not shipped in this distribution tree")
    text = mirror.read_text(encoding="utf-8")
    packages = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert packages == [], f"runtime mirror must stay empty, found: {packages}"


def test_vulnerable_packages_appear_only_inside_extras() -> None:
    """No advisory-bearing package may sit in the unconditional core graph.

    Parsed structurally: the optional-dependencies table is located by its TOML
    header and everything outside that span is the unconditional surface.
    """
    text = _pyproject()
    extras = _table(text, "project.optional-dependencies")
    inside_start = text.index(extras)
    inside_end = inside_start + len(extras)

    for match in re.finditer(r'"([^"]+)"', text):
        span_start, span_end = match.span()
        inside_extras = inside_start <= span_start and span_end <= inside_end
        name = _requirement_name(match.group(1))
        if name is not None and name in VULNERABLE_IF_UNCONDITIONAL:
            assert inside_extras, (
                f"{name} must stay under [project.optional-dependencies]; "
                f"found unconditionally at offset {span_start}"
            )


def test_extras_table_stays_populated_and_parsed() -> None:
    """Sanity guard: the structural parse above must not go vacuous.

    If the extras table were renamed, emptied, or reformatted, the span-based
    containment test in the previous case would trivially pass. Pin the shape so
    a parsing regression fails loudly here instead.
    """
    extras = _table(_pyproject(), "project.optional-dependencies")
    blocks = list(
        re.finditer(
            r"^([A-Za-z0-9][\w.-]*)[^\S\n]*=[^\S\n]*\[(.*?)\]",
            extras,
            re.MULTILINE | re.DOTALL,
        )
    )
    names = {match.group(1) for match in blocks}
    assert len(names) >= 10, f"expected a populated extras table, parsed: {sorted(names)}"
    for expected in ("all", "dashboard", "pandas", "scientific", "io", "test"):
        assert expected in names, f"extras table lost the {expected!r} group"

    requirements = [
        requirement
        for match in blocks
        for requirement in re.findall(r'"([^"]+)"', match.group(2))
        if _requirement_name(requirement) is not None
    ]
    assert len(requirements) >= 30, "expected the extras table to keep its requirement set"


def test_resolved_runtime_graph_contains_no_third_party_package(tmp_path: Path) -> None:
    """Build the wheel; every emitted requirement must be extras-marked.

    This is the end-to-end form of the invariant: it reads the metadata the
    build actually emits rather than the source table, so a backend mistake
    (a requirement landing outside an extras group) is caught even though the
    source-level assertions above would still pass.
    """
    pytest.importorskip("build", reason="the `build` backend is required to build a wheel")

    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        cwd=ROOT,
        env=_src_env(),
        check=True,
        capture_output=True,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    with zipfile.ZipFile(wheels[0]) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        text = archive.read(metadata_name).decode("utf-8")

    requires = [
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith("Requires-Dist:")
    ]
    assert requires, "expected extras requirements to be present"

    bare = [req for req in requires if "extra ==" not in req]
    assert bare == [], f"wheel ships unconditional requirements: {bare}"

    # An advisory-bearing package may appear in the wheel only behind an extra.
    # (`bare` above already proves nothing ships unconditionally; this pins the
    # specific packages by name so the failure message names the culprit.)
    for requirement in requires:
        name = _requirement_name(requirement)
        if name is not None and name in VULNERABLE_IF_UNCONDITIONAL:
            assert "extra ==" in requirement, (
                f"advisory-bearing package reached the shipped wheel unconditionally: {requirement}"
            )


def test_core_import_needs_no_third_party_package() -> None:
    """Importing ``cds`` must not pull a third-party distribution into sys.modules."""
    probe = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import cds\n"
        "import cds.cli, cds.provenance, cds.knowledge, cds.data_io\n"
        "print(repr(sorted({n.split('.')[0] for n in set(sys.modules) - before})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        env=_src_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    added = set(re.findall(r"'([^']+)'", result.stdout))
    third_party = sorted(
        name
        for name in added
        if name
        and not name.startswith("_")
        and name not in sys.stdlib_module_names
        and name != "cds"
    )
    assert third_party == [], f"core import pulled third-party packages: {third_party}"
