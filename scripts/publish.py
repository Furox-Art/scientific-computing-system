#!/usr/bin/env python3
"""Local release preflight for scientific-computing-system.

Usage:
    python scripts/publish.py                 # run tests, build, verify version
    python scripts/publish.py --skip-tests    # skip the pytest step

GitHub Actions publishes: ``release.yml`` starts from a successful CI run on
``main`` and is the sole authority for the PyPI upload, the release tag, and the
GitHub Release; ``tests.yml`` deploys the docs. This script only verifies
locally what CI verifies again -- a clean tree on ``main``, the tests, a clean
build, and an artifact whose version matches ``pyproject.toml``.

Versioning: static. ``version`` lives in ``pyproject.toml`` and is mirrored in
``package.json``, ``codemeta.json``, ``src/cds/_version.py``, ``CITATION.cff``
and ``CHANGELOG.md``. ``scripts/check_version_lockstep.py`` enforces that all six
agree.

Requirements (install once):
    pip install build
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, inheriting stdout/stderr. Raise on non-zero exit.

    ``**kwargs`` is typed ``Any`` rather than ``object`` because it forwards
    into :func:`subprocess.run`, whose keyword parameters span many distinct
    types (``cwd: str | None``, ``env: Mapping``, ``check: bool`` …). A
    narrower annotation would reject valid forwards.
    """
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def run_capture(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, capture output, return result without raising."""
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def git(*args: str) -> str:
    return run_capture(["git", *args], cwd=PROJECT_ROOT).stdout.strip()


def declared_version() -> str:
    """Return the authoritative ``[project] version`` from pyproject.toml."""
    payload = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = payload["project"]["version"]
    if not isinstance(version, str):
        raise SystemExit("pyproject.toml must contain a string [project] version")
    return version


def version_from_wheel(path: Path) -> str | None:
    """Extract the version from a built wheel filename.

    ``scientific_computing_system-1.0.2-py3-none-any.whl`` -> ``1.0.2``
    ``scientific_computing_system-1.0.2.post1-py3-none-any.whl`` -> ``1.0.2.post1``
    """
    m = re.search(r"-([\w.+]+)-py\d", path.name)
    return m.group(1) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest step")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # 1. Sanity: clean working tree, on main. ``src/cds/_version.py`` is a
    #    committed static file, so a forgotten bump would otherwise pass here
    #    and fail in CI.
    status = git("status", "--porcelain")
    if status:
        print("Working tree is dirty. Commit/stash first:\n", status, file=sys.stderr)
        return 1

    branch = git("branch", "--show-current")
    if branch != "main":
        print(f"Not on main (currently on {branch!r}). Run this from main.", file=sys.stderr)
        return 1

    expected_version = declared_version()
    print(f"Declared version: {expected_version}", flush=True)

    # 2. Tests
    if not args.skip_tests:
        print("\n=== Running tests ===", flush=True)
        run([sys.executable, "-m", "pytest", "-q"])

    # 3. Clean dist and build. CI rebuilds independently from a clean checkout;
    #    this build only proves the tree produces an artifact locally.
    dist = PROJECT_ROOT / "dist"
    if dist.exists():
        shutil.rmtree(dist)

    print("\n=== Building sdist + wheel ===", flush=True)
    run([sys.executable, "-m", "build"])

    artifacts = sorted(dist.iterdir())
    print("\nBuilt artifacts:", flush=True)
    for path in artifacts:
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name} ({size_kb:.1f} kB)")

    # 4. Verify the built version matches the version CI will release.
    wheel = next((p for p in artifacts if p.suffix == ".whl"), None)
    if not wheel:
        print("No wheel produced. Aborting.", file=sys.stderr)
        return 1
    built_version = version_from_wheel(wheel)
    if built_version != expected_version:
        print(
            f"Version mismatch: built={built_version}, declared={expected_version}.\n"
            "Check the version surfaces listed in docs/maintenance.md.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nPreflight passed for {expected_version}. Nothing was pushed or uploaded.\n"
        "Merge to main; CI and the Release workflow publish from there.",
        flush=True,
    )
    print(
        "Watch: https://github.com/Furox-Art/scientific-computing-system/actions/workflows/release.yml",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
