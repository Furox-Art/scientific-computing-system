"""``codemeta.json`` is a locked version surface, and drift must fail CI.

``codemeta.json`` was added to the publication surface in the same commit that
made this repository "the front door". It was not added to the version-lockstep
check, and it drifted: after the package moved to ``2.2.0`` the CodeMeta record
still declared ``2.1.0``.

That drift is silent in both directions. Nothing in the test suite read
codemeta.json, so no local run and no CI job failed; and CodeMeta is consumed
by machines -- indexers, registries, aggregators -- so a stale ``version`` field
advertises the previous release to every automated reader while
``import cds; cds.__version__`` reports a different number.

The tests below pin both halves of the contract:

* the *live* state -- every surface in the repository, codemeta.json included,
  currently agrees, and any divergence from the declared version is detected;
* the *failure* state -- the lockstep script itself reports drift in codemeta.json
  with a non-zero exit and a message naming the file.

The second half is what makes the first meaningful. A test that only asserted
"the files agree right now" would pass on a repository where the lockstep script
had simply forgotten about codemeta.json.

A note on Python 3.10
---------------------

``scripts/check_version_lockstep.py`` imports :mod:`tomllib`, which is
Python 3.11+. The ``version_discipline`` CI job runs on 3.12, so the script is
never executed on 3.10 -- but this test module runs in the full matrix, so the
subprocess tests are skipped below 3.11. The read-only assertions, including the
one that reproduces the original drift, still run everywhere.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_version_lockstep.py"
CODEMETA = ROOT / "codemeta.json"

# `scripts/check_version_lockstep.py` reads tomllib, added in Python 3.11.
LOCKSTEP_SCRIPT_MIN_PYTHON = (3, 11)

# Every file the script reads, relative to the repository root.
#
# The whole `src/cds` package is copied, not just `_version.py`, because the
# script imports `cds` to verify the `cds.__version__` re-export behaviourally.
# Copying only `__init__.py` produced a sandbox whose partial package raised a
# circular import, which failed the run for the wrong reason and would have made
# the drift assertion below pass for the wrong reason too.
LOCKED_FILES = (
    "pyproject.toml",
    "package.json",
    "codemeta.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "src/cds",
)

PY_VERSION = re.compile(r'^__version__\s*=\s*version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
CFF_VERSION = re.compile(r"^\s*version:\s*[\"']?([^\s\"']+)[\"']?\s*$", re.MULTILINE)
CHANGELOG_HEADING = re.compile(r"^## \[v?(\d+\.\d+\.\d+)\]", re.MULTILINE)
PROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def _codemeta() -> dict[str, object]:
    payload = json.loads(CODEMETA.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)  # noqa: S101 - JSON object contract
    return payload


def _declared_version() -> str:
    match = PROJECT_VERSION.search((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert match is not None, "pyproject.toml must declare a static project.version"  # noqa: S101
    return match.group(1)


def _locked_versions() -> dict[str, str]:
    """Every locked surface read directly, using no 3.11+-only stdlib."""
    codemeta_version = _codemeta().get("version")
    assert isinstance(codemeta_version, str)  # noqa: S101 - JSON contract

    python_version = PY_VERSION.search(
        (ROOT / "src" / "cds" / "_version.py").read_text(encoding="utf-8-sig")
    )
    assert python_version is not None, "src/cds/_version.py must define __version__"  # noqa: S101

    citation = CFF_VERSION.findall((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert len(citation) == 2, "CITATION.cff must carry two version fields"  # noqa: S101

    package_version = json.loads((ROOT / "package.json").read_text(encoding="utf-8")).get("version")
    assert isinstance(package_version, str)  # noqa: S101 - JSON contract

    return {
        "pyproject.toml [project].version": _declared_version(),
        "package.json version": package_version,
        "codemeta.json version": codemeta_version,
        "src/cds/_version.py __version__": python_version.group(1),
        "CITATION.cff top-level version": citation[0],
        "CITATION.cff preferred-citation version": citation[1],
    }


def _clone_locked_files(destination: Path) -> None:
    """Copy the metadata files and the package the script reads into ``destination``."""
    for relative in LOCKED_FILES:
        source = ROOT / relative
        target = destination / relative
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _run_script_in(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the lockstep script with its ROOT pointed at ``root``.

    The script derives ROOT from its own location (``parents[1]``), so the only
    way to execute it against a mutated tree is to copy it into that tree.
    """
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_copy = scripts_dir / SCRIPT.name
    shutil.copy2(SCRIPT, script_copy)
    return subprocess.run(
        [sys.executable, str(script_copy)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


needs_lockstep_script = pytest.mark.skipif(
    sys.version_info < LOCKSTEP_SCRIPT_MIN_PYTHON,
    reason=(
        "scripts/check_version_lockstep.py imports tomllib "
        f"(Python {LOCKSTEP_SCRIPT_MIN_PYTHON[0]}.{LOCKSTEP_SCRIPT_MIN_PYTHON[1]}+); "
        "the version_discipline CI job runs on a newer interpreter"
    ),
)


def test_codemeta_is_a_locked_version_surface() -> None:
    """The script must read codemeta.json, not silently ignore it."""
    source = SCRIPT.read_text(encoding="utf-8")

    assert "_codemeta_version" in source, (
        "the lockstep script must define a codemeta.json version reader"
    )
    assert '"codemeta.json version"' in source, (
        "codemeta.json must appear in the script's locked-surface mapping"
    )


def test_every_locked_surface_agrees_in_the_repository() -> None:
    """The committed metadata files currently agree on one version."""
    versions = _locked_versions()
    declared = versions["pyproject.toml [project].version"]
    mismatched = {source: value for source, value in versions.items() if value != declared}

    assert not mismatched, f"version drift against {declared}: {mismatched}"
    assert declared == _codemeta()["version"], (
        "codemeta.json must agree with pyproject.toml [project].version"
    )
    assert declared in CHANGELOG_HEADING.findall(
        (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    ), f"CHANGELOG.md must document v{declared}"


def test_codemeta_drift_is_detectable() -> None:
    """The original defect -- a stale codemeta version -- must be caught.

    Reproduces the exact bug this change fixes: every surface agrees except
    codemeta.json, which is left at the previous release. The mismatch has to be
    visible without executing anything, so this runs on every supported Python.
    """
    versions = _locked_versions()
    declared = versions["pyproject.toml [project].version"]

    stale = dict(versions, **{"codemeta.json version": "0.0.1"})
    mismatched = {source: value for source, value in stale.items() if value != declared}

    assert mismatched == {"codemeta.json version": "0.0.1"}, (
        "a stale codemeta.json version must be reported as drift against the "
        f"declared version {declared}"
    )


@needs_lockstep_script
def test_drift_in_codemeta_fails_the_lockstep_check(tmp_path: Path) -> None:
    """A stale codemeta.json version must make the CI check exit non-zero."""
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    _clone_locked_files(sandbox)

    baseline = _run_script_in(sandbox)
    assert baseline.returncode == 0, (
        f"the copied tree should be internally consistent before mutation: {baseline.stderr}"
    )

    codemeta_path = sandbox / "codemeta.json"
    payload = json.loads(codemeta_path.read_text(encoding="utf-8"))
    payload["version"] = "0.0.1"
    codemeta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    drifted = _run_script_in(sandbox)

    assert drifted.returncode != 0, (
        "a stale codemeta.json version must fail scripts/check_version_lockstep.py; "
        f"the script exited 0 with output: {drifted.stdout}"
    )
    assert "codemeta.json version" in drifted.stderr, (
        f"the failure message must name the drifting surface, got: {drifted.stderr}"
    )


@needs_lockstep_script
def test_missing_codemeta_fails_the_lockstep_check(tmp_path: Path) -> None:
    """Deleting codemeta.json must fail rather than silently reduce the surface set."""
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    _clone_locked_files(sandbox)
    (sandbox / "codemeta.json").unlink()

    result = _run_script_in(sandbox)

    assert result.returncode != 0, "a missing codemeta.json must fail the lockstep check"
    assert "codemeta.json is missing" in result.stderr, result.stderr
