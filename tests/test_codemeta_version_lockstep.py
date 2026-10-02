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
  currently agrees;
* the *failure* state -- introducing drift in codemeta.json makes
  ``scripts/check_version_lockstep.py`` exit non-zero with a message naming the
  offending file.

The second half is what makes the first meaningful. A test that only asserted
"the files agree right now" would pass on a repository where the lockstep script
had simply forgotten about codemeta.json; these tests execute the script against
a mutated copy of the real metadata files and require a non-zero exit.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_version_lockstep.py"
CODEMETA = ROOT / "codemeta.json"

# Every file check_version_lockstep.py reads, relative to the repository root.
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


def _load_script() -> ModuleType:
    """Import ``scripts/check_version_lockstep.py`` as a module.

    The script lives outside ``src/`` and outside the installed package, so it
    cannot be imported normally. ``spec_from_file_location`` loads it in place
    without copying it into the tree.
    """
    spec = importlib.util.spec_from_file_location("check_version_lockstep", SCRIPT)
    assert spec is not None and spec.loader is not None  # noqa: S101 - import contract
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_codemeta_is_a_locked_version_surface() -> None:
    """The script must read codemeta.json, not silently ignore it."""
    module = _load_script()
    versions = module.collected_versions()

    assert "codemeta.json version" in versions, (
        f"codemeta.json must be a locked version surface; locked surfaces are {sorted(versions)}"
    )


def test_every_locked_surface_agrees_in_the_repository() -> None:
    """The committed metadata files currently agree on one version."""
    module = _load_script()
    versions = module.collected_versions()

    declared = versions["pyproject.toml [project].version"]
    mismatched = {source: value for source, value in versions.items() if value != declared}

    assert not mismatched, f"version drift against {declared}: {mismatched}"
    assert declared == json.loads(CODEMETA.read_text(encoding="utf-8"))["version"], (
        "codemeta.json must agree with pyproject.toml [project].version"
    )


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


def test_missing_codemeta_fails_the_lockstep_check(tmp_path: Path) -> None:
    """Deleting codemeta.json must fail rather than silently reduce the surface set."""
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    _clone_locked_files(sandbox)
    (sandbox / "codemeta.json").unlink()

    result = _run_script_in(sandbox)

    assert result.returncode != 0, "a missing codemeta.json must fail the lockstep check"
    assert "codemeta.json is missing" in result.stderr, result.stderr
