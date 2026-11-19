"""``__version_tuple__`` is a locked version surface, and drift must fail CI.

``src/cds/_version.py`` exports both a version string (``__version__``) and an
integer tuple (``__version_tuple__``). The tuple sat at ``(2, 2, 21)`` while
every string surface declared ``2.2.22``: the lockstep gate only read the
string, so nothing failed. Downstream code that compares
``cds._version.__version_tuple__`` against a minimum version saw the wrong
release.

These tests pin both halves of the contract, mirroring
``test_codemeta_version_lockstep.py``:

* the *live* state -- the committed tuple spells out the declared version,
  textually and at runtime;
* the *failure* state -- the lockstep script itself reports a stale, missing
  or malformed tuple with a non-zero exit and a message naming the surface.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_version_lockstep.py"
VERSION_FILE = ROOT / "src" / "cds" / "_version.py"

PY_VERSION = re.compile(r'^__version__\s*=\s*version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
PY_VERSION_TUPLE = re.compile(
    r"^__version_tuple__\s*=\s*version_tuple\s*=\s*\(([^)]*)\)", re.MULTILINE
)
PROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)

# Every file the lockstep script reads, relative to the repository root. The
# whole package is copied because the script imports ``cds`` behaviourally.
LOCKED_FILES = (
    "pyproject.toml",
    "package.json",
    "codemeta.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "src/cds",
)


def _declared_version() -> str:
    match = PROJECT_VERSION.search((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert match is not None, "pyproject.toml must declare a static project.version"  # noqa: S101
    return match.group(1)


def _clone_locked_files(destination: Path) -> None:
    for relative in LOCKED_FILES:
        source = ROOT / relative
        target = destination / relative
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _run_script_in(root: Path) -> subprocess.CompletedProcess[str]:
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


def _mutate_tuple(sandbox: Path, replacement: str) -> None:
    """Replace (or delete, with ``replacement=""``) the tuple line in the sandbox."""
    path = sandbox / "src" / "cds" / "_version.py"
    text = path.read_text(encoding="utf-8")
    mutated = re.sub(r"^__version_tuple__.*$", replacement, text, flags=re.MULTILINE)
    assert mutated != text, "the sandbox _version.py must carry a tuple line to mutate"  # noqa: S101
    path.write_text(mutated, encoding="utf-8")


def test_lockstep_script_reads_the_version_tuple() -> None:
    """Guard the guard: the script must parse the tuple, not just the string."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "_python_version_tuple" in source
    assert '"src/cds/_version.py __version_tuple__"' in source


def test_version_tuple_matches_declared_version_in_repository() -> None:
    """The committed tuple must spell out the pyproject version exactly."""
    text = VERSION_FILE.read_text(encoding="utf-8-sig")
    string_match = PY_VERSION.search(text)
    tuple_match = PY_VERSION_TUPLE.search(text)
    assert string_match is not None, "_version.py must define __version__"  # noqa: S101
    assert tuple_match is not None, "_version.py must define __version_tuple__"  # noqa: S101
    parts = tuple(int(part.strip()) for part in tuple_match.group(1).split(","))
    expected = tuple(int(part) for part in _declared_version().split("."))
    assert parts == expected, f"__version_tuple__ {parts} lags the declared version {expected}"
    assert string_match.group(1) == _declared_version()


def test_runtime_version_tuple_matches_version_string() -> None:
    """Behavioural pin: the imported tuple agrees with the imported string."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import cds
        from cds import _version as cds_version
    finally:
        sys.path.pop(0)
    assert cds_version.__version_tuple__ == tuple(
        int(part) for part in cds_version.__version__.split(".")
    )
    assert cds_version.version_tuple == cds_version.__version_tuple__
    assert cds.__version__ == cds_version.__version__


def test_drift_in_version_tuple_fails_the_lockstep_check(tmp_path: Path) -> None:
    """A stale tuple -- the original defect -- must make the CI check exit non-zero."""
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    _clone_locked_files(sandbox)

    baseline = _run_script_in(sandbox)
    assert baseline.returncode == 0, (
        f"the copied tree should be internally consistent before mutation: {baseline.stderr}"
    )

    _mutate_tuple(sandbox, "__version_tuple__ = version_tuple = (0, 0, 1)")
    drifted = _run_script_in(sandbox)
    assert drifted.returncode != 0, (
        "a stale __version_tuple__ must fail scripts/check_version_lockstep.py; "
        f"the script exited 0 with output: {drifted.stdout}"
    )
    assert "__version_tuple__" in drifted.stderr, drifted.stderr


def test_missing_version_tuple_fails_the_lockstep_check(tmp_path: Path) -> None:
    """Deleting the tuple must fail rather than silently reduce the surface set."""
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    _clone_locked_files(sandbox)
    _mutate_tuple(sandbox, "")
    result = _run_script_in(sandbox)
    assert result.returncode != 0
    assert "must define __version_tuple__" in result.stderr, result.stderr


def test_malformed_version_tuple_fails_the_lockstep_check(tmp_path: Path) -> None:
    """Non-integer or wrong-arity tuples must fail with a naming message."""
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    _clone_locked_files(sandbox)
    _mutate_tuple(sandbox, "__version_tuple__ = version_tuple = (2, two, 22)")
    result = _run_script_in(sandbox)
    assert result.returncode != 0
    assert "three non-negative integers" in result.stderr, result.stderr


def test_wrong_arity_version_tuple_fails_the_lockstep_check(tmp_path: Path) -> None:
    sandbox = tmp_path / "repo"
    sandbox.mkdir()
    _clone_locked_files(sandbox)
    _mutate_tuple(sandbox, "__version_tuple__ = version_tuple = (2, 2)")
    result = _run_script_in(sandbox)
    assert result.returncode != 0
    assert "three non-negative integers" in result.stderr, result.stderr
