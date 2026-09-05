#!/usr/bin/env python3
"""Fail CI when package code changes without a synchronized version bump."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _run_git(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout.strip()


def _semver(value: str, *, source: str) -> tuple[int, int, int]:
    match = SEMVER.fullmatch(value)
    if match is None:
        raise ValueError(f"{source} version must be X.Y.Z, got {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _pyproject_version(text: str) -> str:
    payload = tomllib.loads(text)
    project = payload.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("version"), str):
        raise ValueError("pyproject.toml must contain project.version")
    return project["version"]


def _python_version(text: str) -> str:
    match = re.search(r'^__version__\s*=\s*version\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if match is None:
        raise ValueError("src/cds/_version.py must define __version__ = version = X.Y.Z")
    return match.group(1)


def _citation_versions(text: str) -> tuple[str, str]:
    matches = re.findall(r"^\s*version:\s*[\"']?([^\s\"']+)[\"']?\s*$", text, re.MULTILINE)
    if len(matches) != 2:
        raise ValueError("CITATION.cff must contain top-level and preferred-citation version fields")
    return matches[0], matches[1]


def current_versions(root: Path = ROOT) -> tuple[str, str, str, str]:
    """Return pyproject, Python, CFF top-level, and preferred-citation versions."""
    pyproject = _pyproject_version((root / "pyproject.toml").read_text(encoding="utf-8"))
    python = _python_version((root / "src/cds/_version.py").read_text(encoding="utf-8-sig"))
    citation, preferred = _citation_versions((root / "CITATION.cff").read_text(encoding="utf-8"))
    return pyproject, python, citation, preferred


def assert_metadata_sync(root: Path = ROOT) -> str:
    """Require every public version source to declare the exact same semver."""
    versions = current_versions(root)
    if len(set(versions)) != 1:
        raise ValueError(
            "version metadata drift: "
            f"pyproject={versions[0]}, _version.py={versions[1]}, "
            f"CITATION.cff={versions[2]}, preferred-citation={versions[3]}"
        )
    _semver(versions[0], source="current")
    return versions[0]


def _base_version(base_ref: str) -> str:
    text = _run_git("show", f"{base_ref}:pyproject.toml")
    return _pyproject_version(text)


def _changed_paths(base_ref: str) -> tuple[str, ...]:
    output = _run_git("diff", "--name-only", base_ref, "HEAD")
    return tuple(line for line in output.splitlines() if line)


def check_version_discipline(base_ref: str) -> None:
    """Require a monotonic synchronized bump for package-affecting changes."""
    current = assert_metadata_sync()
    changed = _changed_paths(base_ref)
    package_changed = any(path.startswith("src/cds/") or path == "pyproject.toml" for path in changed)
    if not package_changed:
        print(f"Version metadata synchronized at {current}; no package-affecting change detected.")
        return

    base = _base_version(base_ref)
    if _semver(current, source="current") <= _semver(base, source="base"):
        raise ValueError(
            "package-affecting changes require a monotonic version bump: "
            f"base={base}, current={current}"
        )
    required = {"pyproject.toml", "src/cds/_version.py", "CITATION.cff"}
    missing = sorted(required.difference(changed))
    if missing:
        raise ValueError(
            "package-affecting changes must update all version metadata files: "
            + ", ".join(missing)
        )
    print(f"Version discipline passed: {base} -> {current}; {len(changed)} changed paths checked.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="HEAD^1")
    args = parser.parse_args(argv)
    try:
        check_version_discipline(args.base_ref)
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"VERSION DISCIPLINE FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
