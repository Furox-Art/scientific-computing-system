#!/usr/bin/env python3
"""Fail CI when package code changes without a synchronized version bump.

Comparison base
---------------

The diff base must be the *merge base* with the target branch, never ``HEAD^1``.
``HEAD^1`` is the parent of the current commit, which under a rebase merge (and
for any multi-commit push inspected at its tip) is the previous commit of the
same branch. The diff then covers only the final commit, so a pull request that
changed ``src/cds/`` in one commit and added an unrelated follow-up commit passed
this gate with no version bump at all. ``resolve_base_ref`` defaults to
``git merge-base origin/main HEAD``; CI passes the PR merge-base explicitly.

Built artifacts
---------------

``--dist-dir`` additionally requires the built wheel and sdist to declare the
same version as the metadata sources, so a release candidate cannot pass the
discipline gate on source metadata while the artifacts that would actually be
published disagree.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def _run_git(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
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
        raise ValueError(
            "CITATION.cff must contain top-level and preferred-citation version fields"
        )
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


def resolve_base_ref(base_ref: str | None) -> str:
    """Resolve the comparison base for the discipline diff.

    ``HEAD^1`` is only correct for a single-commit or merge-commit update. Under
    a *rebase* merge, or for any multi-commit push evaluated on its last commit,
    ``HEAD^1`` is the previous commit of the same branch, so the diff covers only
    the final commit. A pull request could then change ``src/cds/`` in one
    commit, add an unrelated follow-up commit, and pass this gate with no version
    bump at all:

        commit A  feat: change package code, no version bump   <- invisible
        commit B  docs: unrelated follow-up                    <- only diff seen

    The comparison base must therefore be the *merge base* with the target
    branch, which is stable across merge, squash and rebase strategies. CI passes
    an explicit ``--base-ref``; this fallback covers local and release use.
    """
    if base_ref is not None:
        return base_ref
    for candidate in ("origin/main", "main"):
        try:
            merge_base = _run_git("merge-base", candidate, "HEAD")
        except subprocess.CalledProcessError:
            continue
        if merge_base:
            return merge_base
    raise ValueError(
        "could not resolve a comparison base: pass --base-ref explicitly "
        "(CI uses the PR merge-base against the base branch)"
    )


def check_version_discipline(base_ref: str) -> None:
    """Require a monotonic synchronized bump for package-affecting changes."""
    current = assert_metadata_sync()
    changed = _changed_paths(base_ref)
    package_changed = any(
        path.startswith("src/cds/") or path == "pyproject.toml" for path in changed
    )
    if not package_changed:
        print(
            f"Version metadata synchronized at {current}; no package-affecting "
            f"change detected in {len(changed)} path(s) against {base_ref}."
        )
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
    parser.add_argument(
        "--base-ref",
        default=None,
        help=(
            "comparison base for the diff; defaults to the merge-base with "
            "origin/main. Do not use HEAD^1: it under-reports the change set for "
            "multi-commit and rebased updates."
        ),
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=None,
        help=(
            "additionally require built wheel/sdist metadata to agree with the "
            "declared version (delegates to scripts/check_version_lockstep.py)"
        ),
    )
    args = parser.parse_args(argv)
    try:
        base_ref = resolve_base_ref(args.base_ref)
        check_version_discipline(base_ref)
        if args.dist_dir is not None:
            lockstep = ROOT / "scripts" / "check_version_lockstep.py"
            command = [sys.executable, str(lockstep), "--dist-dir", str(args.dist_dir)]
            completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
            if completed.returncode != 0:
                raise ValueError("built artifacts do not match the declared version")
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"VERSION DISCIPLINE FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
