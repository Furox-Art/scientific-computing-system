#!/usr/bin/env python3
"""Assert the exact contents of the built wheel and sdist.

A distribution that builds cleanly is not evidence that it contains what it
should, or that it excludes what it should not. Both failure modes are silent:
a wheel missing ``py.typed`` still imports, still runs, and silently degrades
every downstream type checker; an sdist carrying ``site/``, ``node_modules/`` or
promo media still installs, just several orders of magnitude larger.

This script is the packaging content gate. It fails on:

* **missing** ``cds/py.typed`` (PEP 561 marker)
* **missing** console entry point metadata (``cds = cds.cli:main``)
* **wrong** distribution name / version in ``METADATA`` and ``PKG-INFO``
* **extra** top-level importable packages in the wheel (the wheel must ship
  exactly ``cds``; anything else can shadow user imports)
* **extra** top-level directories in the sdist (only an explicit allowlist of
  documentation, tests, examples and metadata files may ship)
* any attempt to ship build residue: ``site/``, ``node_modules/``, ``dist/``,
  ``__pycache__/``, ``*.pyc``, ``.coverage``, ``.pytest_cache/``, ``.git/``

Usage
-----

    python scripts/check_distribution_contents.py dist
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import NoReturn

DIST_NAME = "scientific-computing-system"
WHEEL_DIST = "scientific_computing_system"

# Top-level entries allowed inside the sdist. Anything else is either build
# residue, a secret, or a multi-megabyte media tree.
SDIST_ALLOWED_TOP_LEVEL = frozenset(
    {
        ".github",
        ".gitignore",
        "CHANGELOG.md",
        "CITATION.cff",
        "codemeta.json",
        "CONTRIBUTING.md",
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "SECURITY.md",
        "paper.bib",
        "paper.md",
        "benchmarks",
        "bin",
        "dashboard",
        "docs",
        "examples",
        "index.js",
        "mkdocs.yml",
        "npm.test.js",
        "package.json",
        "pyproject.toml",
        "requirements-build.lock",
        "requirements-dev.lock",
        "requirements.lock",
        "scripts",
        "src",
        "tests",
    }
)

# Path fragments that must never appear in a published artifact.
FORBIDDEN_FRAGMENTS = (
    "site/",
    "node_modules/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".git/",
    ".coverage",
    ".env",
    "dist/",
)

FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".so.orig", ".rej")

# Required wheel members beyond the package itself.
REQUIRED_WHEEL_MEMBERS = (
    "cds/__init__.py",
    "cds/py.typed",
    "DIST_INFO/METADATA",
    "DIST_INFO/entry_points.txt",
    "DIST_INFO/WHEEL",
    "DIST_INFO/licenses/LICENSE",
)

REQUIRED_ENTRY_POINT = "cds = cds.cli:main"

CLASS_AUDIT = (
    "Wheel/Sdist contents",
    "dist/",
    "The built wheel must ship exactly the `cds` package (plus its .dist-info) "
    "and the PEP 561 `py.typed` marker, with no build residue; the sdist must "
    "ship only the explicit allowlist of source, tests, docs and metadata.",
)


class ContentError(Exception):
    """Raised when a built artifact violates its content contract."""


def _fail(problems: list[str]) -> NoReturn:
    # NoReturn lets mypy narrow types after a failure branch (for example the
    # `IO[bytes] | None` returned by `TarFile.extractfile`).
    raise ContentError("\n  - ".join([""] + problems))


def _forbidden_hits(names: list[str]) -> list[str]:
    hits = []
    for name in names:
        if any(fragment in name for fragment in FORBIDDEN_FRAGMENTS):
            hits.append(f"{name} (matches a forbidden build-residue fragment)")
        elif name.endswith(FORBIDDEN_SUFFIXES):
            hits.append(f"{name} (forbidden file suffix)")
    return hits


def check_wheel(wheel: Path, expected_version: str) -> list[str]:
    """Verify the wheel ships exactly the ``cds`` package plus its metadata."""
    dist_info = f"{WHEEL_DIST}-{expected_version}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

        if not any(n.startswith(f"{dist_info}/") for n in names):
            _fail([f"{wheel.name}: no {dist_info}/ entries found"])

        for member in REQUIRED_WHEEL_MEMBERS:
            resolved = member.replace("DIST_INFO", dist_info, 1)
            if resolved not in names:
                _fail([f"{wheel.name}: required member missing: {resolved}"])

        top_level = sorted({n.split("/")[0] for n in names if "/" in n})
        unexpected = [entry for entry in top_level if entry not in {"cds", dist_info}]
        if unexpected:
            _fail(
                [
                    f"{wheel.name}: unexpected top-level entries {unexpected}; "
                    "the wheel must ship only `cds` (an extra top-level package "
                    "can shadow imports in the user's environment)"
                ]
            )

        residue = _forbidden_hits(names)
        if residue:
            _fail([f"{wheel.name}: build residue present: {sorted(residue)}"])

        if not any(n.startswith("cds/__init__.py") for n in names):
            _fail([f"{wheel.name}: cds/__init__.py missing"])

        entry_points = archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        if REQUIRED_ENTRY_POINT not in entry_points:
            _fail(
                [
                    f"{wheel.name}: entry_points.txt does not declare "
                    f"{REQUIRED_ENTRY_POINT!r}; found: {entry_points.strip()!r}"
                ]
            )

        metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
        fields = _fields(metadata)
        if fields.get("Name") != DIST_NAME:
            _fail(
                [f"{wheel.name}: METADATA Name is {fields.get('Name')!r}, expected {DIST_NAME!r}"]
            )
        if fields.get("Version") != expected_version:
            _fail(
                [
                    f"{wheel.name}: METADATA Version is {fields.get('Version')!r}, "
                    f"expected {expected_version!r}"
                ]
            )
        if not fields.get("Requires-Python"):
            _fail([f"{wheel.name}: METADATA has no Requires-Python"])
        if "License-Expression: MIT" not in metadata and "License: MIT" not in metadata:
            _fail([f"{wheel.name}: METADATA does not declare the MIT license"])

    return [
        f"{wheel.name}: {len(names)} members, py.typed present, entry point declared",
    ]


def check_sdist(sdist: Path, expected_version: str) -> list[str]:
    """Verify the sdist root, allowlist and metadata."""
    prefix = f"{WHEEL_DIST}-{expected_version}/"
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()

        bad_roots = sorted(
            {n for n in names if not (n == prefix.rstrip("/") or n.startswith(prefix))}
        )
        if bad_roots:
            _fail(
                [
                    f"{sdist.name}: entries outside the {prefix} root: {bad_roots[:5]}",
                ]
            )

        top_level = sorted({n[len(prefix) :].split("/")[0] for n in names if len(n) > len(prefix)})
        unexpected = [entry for entry in top_level if entry not in SDIST_ALLOWED_TOP_LEVEL]
        if unexpected:
            _fail(
                [
                    f"{sdist.name}: unexpected top-level entries {unexpected}; "
                    f"allowed: {sorted(SDIST_ALLOWED_TOP_LEVEL)}"
                ]
            )

        residue = _forbidden_hits(names)
        if residue:
            _fail([f"{sdist.name}: build residue present: {sorted(residue)}"])

        required = (
            f"{prefix}pyproject.toml",
            f"{prefix}PKG-INFO",
            f"{prefix}README.md",
            f"{prefix}LICENSE",
            f"{prefix}src/cds/__init__.py",
            f"{prefix}src/cds/py.typed",
            f"{prefix}tests/conftest.py",
            f"{prefix}.github/workflows/release.yml",
            f"{prefix}.github/workflows/tests.yml",
            f"{prefix}requirements-build.lock",
            # Locked version surfaces: the test suite reads all of them from the
            # repository root, so omitting any one makes the suite fail with
            # FileNotFoundError when run from an unpacked sdist.
            f"{prefix}codemeta.json",
            f"{prefix}package.json",
        )
        absent = [member for member in required if member not in names]
        if absent:
            _fail([f"{sdist.name}: required members missing: {absent}"])

        handle = archive.extractfile(f"{prefix}PKG-INFO")
        if handle is None:
            _fail([f"{sdist.name}: PKG-INFO unreadable"])
        with handle:
            fields = _fields(handle.read().decode("utf-8"))
        if fields.get("Version") != expected_version:
            _fail(
                [
                    f"{sdist.name}: PKG-INFO Version is {fields.get('Version')!r}, "
                    f"expected {expected_version!r}"
                ]
            )

    return [
        f"{sdist.name}: {len(names)} entries, allowlist clean, tests + .github/workflows shipped",
    ]


def _fields(metadata: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in metadata.splitlines():
        if not line.strip():
            break
        key, separator, value = line.partition(": ")
        if separator:
            fields.setdefault(key, value.strip())
    return fields


def expected_version_from_pyproject(root: Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise ContentError("pyproject.toml must declare a static project.version")
    return match.group(1)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path, help="directory containing the built artifacts")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root used to read the declared version (default: script parent)",
    )
    args = parser.parse_args(argv)

    try:
        version = expected_version_from_pyproject(args.root)
        dist = args.dist_dir
        wheels = sorted(dist.glob("*.whl"))
        sdists = sorted(dist.glob("*.tar.gz"))
        if len(wheels) != 1:
            _fail([f"expected exactly one wheel in {dist}, found {len(wheels)}"])
        if len(sdists) != 1:
            _fail([f"expected exactly one sdist in {dist}, found {len(sdists)}"])

        for line in check_wheel(wheels[0], version):
            print(f"PASS: {line}")
        for line in check_sdist(sdists[0], version):
            print(f"PASS: {line}")
        print(f"Built artifact contents verified against version {version}.")
    except ContentError as exc:
        print(f"DISTRIBUTION CONTENTS CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
