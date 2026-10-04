"""The release toolchain must be pinned, and the pin must match what it validates.

A sibling repository failed a real PyPI release because ``twine`` was unpinned.
twine<= 6.2.0 replaces ``packaging.metadata._VALID_METADATA_VERSIONS`` at import
time with a list ending at 2.4, so ``twine check --strict`` rejects a valid
artifact with ``InvalidDistribution: '2.5' is not a valid metadata version``. The
package was fine; the validator was wrong.

Every test here is paired with a mutation that must turn the suite red. An
assertion nobody checked the teeth of is not a guard.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import zipfile
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_release_toolchain.py"
BUILD_LOCK = ROOT / "requirements-build.lock"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"

# What the guard reads, relative to the repository root.
SANDBOX_FILES = (
    "requirements-build.lock",
    ".github/workflows",
)


def _load_guard() -> ModuleType:
    """Import the guard by path.

    ``scripts/`` is not a package and is deliberately absent from the mypy
    search path, so a plain ``import check_release_toolchain`` is an
    ``import-not-found`` under ``mypy tests/``. Other suites in this repo drive
    ``scripts/`` through a subprocess; this one also needs the functions directly
    to test the compatibility boundary, so the module is loaded explicitly and
    typed as ``ModuleType``.
    """
    location = ROOT / "scripts" / SCRIPT.name
    spec = importlib.util.spec_from_file_location("_check_release_toolchain", location)
    assert spec is not None, f"cannot build an import spec for {location}"
    assert spec.loader is not None, f"no loader for {location}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard: ModuleType = _load_guard()


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _lock_pins() -> dict[str, str]:
    return {name: str(entry["version"]) for name, entry in guard.parse_lock(BUILD_LOCK).items()}


def _clone(destination: Path) -> Path:
    """Copy the files the guard reads, plus the guard itself, into ``destination``."""
    for relative in SANDBOX_FILES:
        source = ROOT / relative
        target = destination / relative
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    scripts = destination / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, scripts / SCRIPT.name)
    return destination


def _run_guard(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "scripts" / SCRIPT.name), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


HASHED_TOOLCHAIN_INSTALL = (
    "          python -m pip install\n"
    "          --require-hashes\n"
    "          --only-binary=:all:\n"
    "          -r requirements-build.lock"
)


def _mutate(tmp_path: Path, relative: str, old: str, new: str) -> Path:
    """Clone the sandbox, assert the replacement text is present, apply it."""
    root = _clone(tmp_path / "repo")
    target = root / relative
    text = _text(target)
    assert old in text, f"mutation target not found in {relative}: {old!r}"
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return root


# --------------------------------------------------------------------------
# the lock itself
# --------------------------------------------------------------------------


def test_build_lock_pins_build_and_twine_exactly_with_hashes() -> None:
    """A range or a bare name lets the release toolchain drift between runs."""
    entries = guard.parse_lock(BUILD_LOCK)
    for package in guard.TOOLCHAIN_PACKAGES:
        assert package in entries, f"{package} is not pinned in {BUILD_LOCK.name}"
        assert re.fullmatch(r"\d+(\.\d+)*", str(entries[package]["version"])), (
            f"{package} must be pinned to an exact version, got {entries[package]['version']!r}"
        )
        assert entries[package]["hashes"], f"{package} has no --hash entry"


def test_build_lock_gives_every_requirement_exactly_one_hash() -> None:
    """`--require-hashes` needs a digest per requirement, or the install fails."""
    text = _text(BUILD_LOCK)
    pins = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.lstrip().startswith("--hash")
    ]
    assert pins
    assert text.count("--hash=sha256:") == len(pins)
    assert guard.check_lock(guard.parse_lock(BUILD_LOCK)) == []


def test_twine_is_not_installed_on_the_command_line() -> None:
    """`--require-hashes` rejects unhashed command-line requirements.

    Verified: `pip install --require-hashes -r requirements-build.lock twine==7.0.0`
    fails with "Hashes are required in --require-hashes mode". So the pin has to
    live in the lock file to be hash-verified at all.
    """
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        for job, body in guard.iter_jobs(_text(path)):
            for block in guard.run_blocks(body):
                for line in block.splitlines():
                    stripped = line.strip()
                    if "pip install" not in stripped:
                        continue
                    for package in guard.TOOLCHAIN_PACKAGES:
                        if re.search(rf"(?<![\w-]){package}(?![\w-])", stripped):
                            assert "requirements-build.lock" in stripped, (
                                f"{path.name}:{job} pins {package} on the command line: "
                                f"{stripped!r}. It must come from the hash-locked file."
                            )


def test_release_workflow_validates_metadata_with_strict_twine_check() -> None:
    """The artifacts must be rendered before upload, not only after."""
    release = _text(RELEASE_WORKFLOW)
    assert "twine check --strict" in release
    build = release.index("Build sdist + wheel without dependency isolation")
    twine = release.index("twine check --strict")
    assert build < twine, "twine must validate the artifacts the build just produced"
    assert twine < release.index("Publish to PyPI (Trusted Publishing)")


# --------------------------------------------------------------------------
# pin / Metadata-Version compatibility
# --------------------------------------------------------------------------


def test_pinned_twine_can_validate_the_emitted_metadata_version() -> None:
    """The pin and the metadata version are one fact, not two.

    The artifacts are built here rather than read from disk so the assertion is
    about what the build backend actually emits, which is the thing that moved
    in the sibling failure.
    """
    twine_version = _lock_pins()["twine"]
    metadata_version = _emitted_metadata_version()
    assert guard.twine_can_validate(twine_version, metadata_version), (
        f"twine {twine_version} cannot validate Metadata-Version {metadata_version}"
    )


def _emitted_metadata_version() -> str:
    """Build the artifacts and read ``Metadata-Version`` back out of them."""
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory() as tmp:
        build = subprocess.run(
            [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", tmp],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if build.returncode != 0:
            pytest.skip(f"cannot build in this environment: {build.stderr[-400:]}")
        wheels = list(Path(tmp).glob("*.whl"))
        assert len(wheels) == 1, f"expected one wheel, found {len(wheels)}"
        with zipfile.ZipFile(wheels[0]) as archive:
            names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            metadata = archive.read(names[0]).decode("utf-8", errors="replace")
    for line in metadata.splitlines():
        if not line.strip():
            break
        if line.startswith("Metadata-Version:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError("built metadata has no Metadata-Version")


@pytest.mark.parametrize(
    ("twine_version", "metadata_version", "expected"),
    [
        ("6.2.0", "2.4", True),  # the override's own ceiling
        ("6.2.0", "2.5", False),  # the sibling failure
        ("6.1.0", "2.5", False),
        ("5.1.1", "2.2", True),
        ("7.0.0", "2.5", True),  # override removed
        ("7.0.0", "2.6", True),
        ("8.1.0", "2.6", True),
    ],
)
def test_twine_metadata_compatibility_boundary(
    twine_version: str, metadata_version: str, expected: bool
) -> None:
    assert guard.twine_can_validate(twine_version, metadata_version) is expected


# --------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------


def test_guard_passes_on_the_repository_as_committed(tmp_path: Path) -> None:
    """The green baseline every mutation below is measured against."""
    result = _run_guard(_clone(tmp_path / "repo"), "--skip-network")
    assert result.returncode == 0, (
        f"guard failed on a clean tree:\n{result.stdout}\n{result.stderr}"
    )
    assert "Release toolchain verified." in result.stdout


def test_unpinned_twine_is_a_failure_not_a_warning(tmp_path: Path) -> None:
    """Dropping the pin must fail the guard, whatever the network says."""
    root = _mutate(
        tmp_path,
        "requirements-build.lock",
        "twine==7.0.0 \\\n    --hash=sha256:b854164df26db268af05f49aa5c0344b10e27a494343ff05b1e0bad3b135f5a7\n",
        "",
    )
    result = _run_guard(root, "--skip-network")
    assert result.returncode == 1, f"an unpinned twine must fail:\n{result.stdout}"
    assert "twine NOT PINNED" in result.stdout
    assert "does not pin twine" in result.stderr


def test_a_range_instead_of_an_exact_pin_is_a_failure(tmp_path: Path) -> None:
    """`twine>=7` is not a pin; it re-opens the exact hole being closed."""
    root = _mutate(
        tmp_path,
        "requirements-build.lock",
        "twine==7.0.0 \\\n    --hash=sha256:b854164df26db268af05f49aa5c0344b10e27a494343ff05b1e0bad3b135f5a7",
        "twine>=7.0.0",
    )
    result = _run_guard(root, "--skip-network")
    assert result.returncode == 1, f"a version range must fail:\n{result.stdout}"
    assert "not an exact '==' pin" in result.stderr


def test_a_pin_too_old_for_the_metadata_version_is_a_failure(tmp_path: Path) -> None:
    """The sibling failure, reproduced as a guard failure.

    twine 6.2.0 accepts at most Metadata-Version 2.4. Declaring 2.5 must be fatal
    rather than a note, because that combination fails the release.
    """
    root = _clone(tmp_path / "repo")
    lock = root / "requirements-build.lock"
    lock.write_text(
        _text(lock).replace("twine==7.0.0", "twine==6.2.0"),
        encoding="utf-8",
    )
    result = _run_guard(root, "--skip-network", "--metadata-version", "2.5")
    assert result.returncode == 1, f"a too-old pin must fail:\n{result.stdout}"
    assert "cannot validate" in result.stderr
    assert "2.5" in result.stderr


def test_missing_strict_is_a_failure(tmp_path: Path) -> None:
    """Without --strict, rendering problems pass silently."""
    root = _mutate(
        tmp_path,
        ".github/workflows/release.yml",
        "python -m twine check --strict dist/*",
        "python -m twine check dist/*",
    )
    result = _run_guard(root, "--skip-network")
    assert result.returncode == 1, f"a missing --strict must fail:\n{result.stdout}"
    assert "without --strict" in result.stderr


def test_adhoc_twine_install_is_a_failure(tmp_path: Path) -> None:
    """Installing twine on the command line bypasses the pin entirely."""
    root = _mutate(
        tmp_path,
        ".github/workflows/release.yml",
        HASHED_TOOLCHAIN_INSTALL,
        "          python -m pip install --upgrade pip\n          pip install twine",
    )
    result = _run_guard(root, "--skip-network")
    assert result.returncode == 1, f"an ad hoc twine install must fail:\n{result.stdout}"
    assert "installs twine ad hoc" in result.stderr


def test_twine_check_without_the_hash_locked_toolchain_is_a_failure(tmp_path: Path) -> None:
    """A `twine check` in a job that never installs the lock uses an unknown twine."""
    root = _mutate(
        tmp_path,
        ".github/workflows/release.yml",
        HASHED_TOOLCHAIN_INSTALL,
        "          python -m pip install --upgrade pip",
    )
    result = _run_guard(root, "--skip-network")
    assert result.returncode == 1, f"an unlocked twine check must fail:\n{result.stdout}"
    assert "never installs requirements-build.lock" in result.stderr


def test_no_twine_check_at_all_is_a_failure(tmp_path: Path) -> None:
    """If nothing renders the metadata, an unrenderable artifact reaches the index."""
    root = _mutate(
        tmp_path,
        ".github/workflows/release.yml",
        "        run: python -m twine check --strict dist/*",
        "        run: echo 'skipped the metadata render check'",
    )
    result = _run_guard(root, "--skip-network")
    assert result.returncode == 1, f"a missing twine check must fail:\n{result.stdout}"
    assert "No workflow runs `twine check`" in result.stderr


def test_network_failure_is_a_warning_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable index must not be reported as a repository defect.

    This is the sibling's second lesson: a guard that verifies pins over the
    network fails (or worse, silently verifies nothing) when the network is bad.
    The offline checks must still pass, with a warning that says what was skipped.
    """
    root = _clone(tmp_path / "repo")

    def explode() -> list[str]:
        raise OSError("simulated network outage")

    monkeypatch.setattr(guard, "verify_pins_upstream", explode)
    monkeypatch.setattr(guard, "verify_action_pins", explode)

    result = _run_guard(root)
    assert result.returncode == 0, (
        f"a network failure must not fail the guard:\n{result.stdout}\n{result.stderr}"
    )
    assert "Release toolchain verified." in result.stdout


def test_missing_github_token_warns_instead_of_claiming_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unauthenticated GitHub calls are capped at 60/hour.

    Verifying action pins without a token looks like it works and can silently
    confirm nothing, which is worse than not checking. The guard must say so.
    """
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    notes = guard.verify_action_pins()
    assert any("GITHUB_TOKEN is unset" in note for note in notes), notes
    assert any("60/hour" in note for note in notes), notes


def test_authenticated_action_pin_verification_confirms_the_real_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a token, every committed action pin must resolve upstream."""
    monkeypatch.setenv("GITHUB_TOKEN", "not-a-real-token-but-present")
    calls: list[str] = []

    def fake_get_json(
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> dict[str, str]:
        calls.append(url)
        return {"sha": "x"}

    monkeypatch.setattr(guard, "_get_json", fake_get_json)
    notes = guard.verify_action_pins()
    assert notes, "expected one note per distinct action pin"
    assert all("resolves upstream" in note for note in notes), notes
    assert all("api.github.com/repos/" in url for url in calls), calls


def test_unresolvable_action_pin_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead SHA must be named, not swallowed."""
    from email.message import Message

    monkeypatch.setenv("GITHUB_TOKEN", "present")

    def fake_get_json(
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> dict[str, str]:
        raise urllib.error.HTTPError(url, 422, "No commit found for SHA", Message(), None)

    monkeypatch.setattr(guard, "_get_json", fake_get_json)
    notes = guard.verify_action_pins()
    assert notes
    assert all("did not resolve upstream" in note for note in notes), notes


# --------------------------------------------------------------------------
# packaging metadata sanity (requirement 5)
# --------------------------------------------------------------------------


class _WheelMetadata(NamedTuple):
    """The wheel's core metadata, split into headers and shipped dist-info files."""

    headers: dict[str, list[str]]
    files: dict[str, str]

    def one(self, key: str) -> str:
        """Return the single value of a header, failing loudly if not exactly one."""
        values = self.headers.get(key, [])
        assert len(values) == 1, f"expected exactly one {key}, found {values!r}"
        return values[0]


def _build_wheel(directory: str) -> Path:
    """Build a wheel with the ambient toolchain and return its path."""
    build = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", directory],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        pytest.skip(f"cannot build in this environment: {build.stderr[-400:]}")
    wheels = list(Path(directory).glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {len(wheels)}"
    return wheels[0]


def _wheel_metadata() -> _WheelMetadata:
    """Build the wheel and return its core metadata headers plus dist-info files."""
    with tempfile.TemporaryDirectory() as tmp:
        wheel = _build_wheel(tmp)
        with zipfile.ZipFile(wheel) as archive:
            names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            assert len(names) == 1, f"expected one METADATA, found {len(names)}"
            raw = archive.read(names[0]).decode("utf-8", errors="replace")
            dist_info = names[0].split("/")[0]
            files = {
                name: archive.read(name).decode("utf-8", errors="replace")
                for name in archive.namelist()
                if name.startswith(dist_info)
            }
    headers: dict[str, list[str]] = {}
    for line in raw.splitlines():
        if not line.strip():
            break  # headers end at the first blank line; the rest is the description
        key, separator, value = line.partition(":")
        if separator:
            headers.setdefault(key.strip(), []).append(value.strip())
    return _WheelMetadata(headers=headers, files=files)


def test_license_uses_pep639_expression_and_not_legacy_classifiers() -> None:
    """PEP 639 supersedes `License ::` classifiers; both together is a contradiction."""
    metadata = _wheel_metadata()
    assert metadata.headers.get("License-Expression"), "expected a PEP 639 License-Expression"
    classifiers = metadata.headers.get("Classifier", [])
    legacy = [c for c in classifiers if c.startswith("License ::")]
    assert not legacy, f"PEP 639 License-Expression must not be mixed with {legacy}"


def test_license_file_is_declared_and_present() -> None:
    metadata = _wheel_metadata()
    declared = metadata.headers.get("License-File", [])
    assert declared, "expected at least one License-File"
    for name in declared:
        shipped = any(key.endswith(f"/{name}") for key in metadata.files)
        assert shipped, f"License-File {name!r} is declared but not shipped in the wheel"


def test_requires_python_is_present_and_sane() -> None:
    metadata = _wheel_metadata()
    requires_python = metadata.one("Requires-Python")
    assert re.match(r"^>=\d+\.\d+", requires_python), requires_python


def test_metadata_declares_no_hardcoded_dependencies() -> None:
    """This package is zero-runtime-dependency by design."""
    metadata = _wheel_metadata()
    runtime = [v for v in metadata.headers.get("Requires-Dist", []) if "extra ==" not in v]
    assert runtime == [], f"unexpected runtime dependencies: {runtime}"


def test_packaging_metadata_is_json_serialisable_and_stable() -> None:
    """Guard against the lockstep-style drift this repo keeps getting bitten by."""
    metadata = _wheel_metadata()
    payload = {
        "Metadata-Version": metadata.one("Metadata-Version"),
        "Name": metadata.one("Name"),
        "Version": metadata.one("Version"),
        "Requires-Python": metadata.one("Requires-Python"),
        "License-Expression": metadata.one("License-Expression"),
    }
    assert json.loads(json.dumps(payload)) == payload
