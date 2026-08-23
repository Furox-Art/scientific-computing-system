#!/usr/bin/env python3
"""Local release driver for scientific-computing-system.

Usage:
    python scripts/publish.py                   # use latest git tag as version
    python scripts/publish.py --tag v1.1.6      # use a specific tag
    python scripts/publish.py --dry-run         # build + verify, but push nothing
    python scripts/publish.py --skip-tests      # skip the pytest step
    python scripts/publish.py --skip-docs       # skip mkdocs gh-deploy

Role of this script: it is the LOCAL half of a two-stage release. It
verifies a clean tree on ``main``, runs the test suite, builds the sdist
+ wheel, confirms the built version matches the tag, then **pushes the
tag**. Pushing a ``v*`` tag triggers ``.github/workflows/release.yml``,
which is the SOLE authority for the PyPI upload and the GitHub Release —
this script does not touch PyPI or create the release, so there is no
double-upload and no long-lived token handled locally.

Docs are deployed here (``mkdocs gh-deploy``) rather than in CI, because
the canonical docs build lives on the local toolchain.

Versioning: static. ``version`` lives in ``pyproject.toml`` and is mirrored
in ``src/cds/_version.py`` (kept in lockstep — bump both before tagging).
The script reads the version from the built wheel filename (e.g.
``cognitive_discovery_system-1.0.2-py3-none-any.whl``) and uses it for
verification and tag creation. (Previously we tried hatch-vcs, but 0.5.0
silently ignored the version-scheme override; static versioning is what
shipped — see ``pyproject.toml`` release checklist.)

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


def latest_tag() -> str | None:
    """Return the latest ``v*`` tag, or None if there is none."""
    out = git("tag", "--list", "v*", "--sort=-v:refname")
    return out.splitlines()[0] if out else None


def version_from_wheel(path: Path) -> str | None:
    """Extract the version from a built wheel filename.

    ``cognitive_discovery_system-1.0.2-py3-none-any.whl`` -> ``1.0.2``
    ``cognitive_discovery_system-1.0.2.post1-py3-none-any.whl`` -> ``1.0.2.post1``
    """
    m = re.search(r"-([\w.+]+)-py\d", path.name)
    return m.group(1) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--tag",
        help="Git tag to publish (e.g. v0.9.0b1). Defaults to the latest v* tag.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build + verify, but push nothing")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest step")
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip mkdocs gh-deploy (Pages not updated after the tag push).",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    # 1. Sanity: clean working tree, on main. ``src/cds/_version.py`` is a
    #    committed static file (lockstepped with ``pyproject.toml`` version),
    #    so a dirty tree there MUST block publish — a forgotten bump would
    #    otherwise ship a version mismatch.
    status = git("status", "--porcelain")
    if status:
        print("Working tree is dirty. Commit/stash first:\n", status, file=sys.stderr)
        return 1

    branch = git("branch", "--show-current")
    if branch != "main":
        print(f"Not on main (currently on {branch!r}). Publish from main only.", file=sys.stderr)
        return 1

    # 2. Determine the tag/version
    if args.tag:
        tag = args.tag
    else:
        tag = latest_tag()

    if not tag:
        print("No --tag given and no v* tags found locally. Use --tag vX.Y.Z.", file=sys.stderr)
        return 1

    if not tag.startswith("v"):
        print(f"Tag {tag!r} does not start with 'v'. Rename and retry.", file=sys.stderr)
        return 1

    print(f"Version: {tag} (from {'--tag' if args.tag else 'latest'})", flush=True)

    # 3. Tests
    if not args.skip_tests:
        print("\n=== Running tests ===", flush=True)
        run([sys.executable, "-m", "pytest", "-q"])

    # 4. Clean dist, build (the wheel is needed to verify the version the
    #    tag will publish; CI rebuilds independently from a clean checkout).
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

    # 5. Verify the built version matches the requested tag
    wheel = next((p for p in artifacts if p.suffix == ".whl"), None)
    if not wheel:
        print("No wheel produced. Aborting.", file=sys.stderr)
        return 1
    built_version = version_from_wheel(wheel)
    expected_version = tag.lstrip("v")
    if built_version != expected_version:
        print(
            f"Version mismatch: built={built_version}, expected={expected_version}.\n"
            f"  Did you forget to create the tag? `git tag -a {tag} -m 'Release {tag}'`",
            file=sys.stderr,
        )
        return 1

    # 6. Stop here on a dry run — nothing is pushed, no release is triggered.
    if args.dry_run:
        print("\nDry run: skipping tag push and docs deploy.", flush=True)
        return 0

    # 7. Push tag. This triggers release.yml, which uploads to PyPI and
    #    creates the GitHub Release. The local script stays out of both.
    print(f"\n=== Pushing tag {tag} (triggers CI release.yml) ===", flush=True)
    run(["git", "push", "origin", tag])

    # 8. Deploy docs to GitHub Pages (legacy mode: push to gh-pages branch).
    #    Kept local on purpose — docs deploy is not part of release.yml.
    if not args.skip_docs:
        print("\n=== Deploying docs to GitHub Pages ===", flush=True)
        try:
            run([sys.executable, "-m", "mkdocs", "gh-deploy", "--force", "--no-history"])
        except subprocess.CalledProcessError as exc:
            print(
                f"  mkdocs gh-deploy failed (exit {exc.returncode}). "
                "The tag push already succeeded and release.yml is running; "
                "you can retry the docs deploy later with --skip-tests.",
                file=sys.stderr,
            )

    print(
        f"\nDone. {tag} pushed; CI release.yml will publish to PyPI and create the GitHub Release.",
        flush=True,
    )
    print(
        "Watch: https://github.com/Furox-Art/scientific-computing-system/actions/workflows/release.yml",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
