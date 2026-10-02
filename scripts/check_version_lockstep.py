#!/usr/bin/env python3
"""Fail CI when the published version metadata sources drift apart.

This is the *lockstep* gate. ``scripts/check_version_discipline.py`` answers a
different question -- "did a package-affecting change ship with a monotonic
version bump?" -- while this script answers "do all five release surfaces agree
on the version right now?".

Release surfaces checked
------------------------

=========================  =========================================================
``pyproject.toml``         ``[project] version`` -- the authoritative source.
``package.json``           ``version`` for the Node launcher package.
``src/cds/_version.py``    ``__version__`` -- re-exported as ``cds.__version__``.
``CITATION.cff``           top-level and ``preferred-citation`` version fields.
``CHANGELOG.md``           must carry a ``## [vX.Y.Z]`` heading for that version.
=========================  =========================================================

``src/cds/__init__.py`` is verified *behaviourally* rather than textually: the
check imports ``cds`` and asserts ``cds.__version__`` equals the pyproject
version. That catches the class of bug where the re-export is deleted, renamed
or pointed at a stale constant -- a purely textual grep would happily pass while
``cds.__version__`` raised ``AttributeError`` for every downstream user.

Why CHANGELOG is enforced
-------------------------

A release whose version is absent from ``CHANGELOG.md`` is undocumented. That
was the state of ``2.1.0`` on ``main``: PyPI, the wheel METADATA and
``CITATION.cff`` all declared ``2.1.0`` while the changelog stopped at
``2.0.1``. Nothing failed, because nothing compared them. The drift is silent by
construction unless a machine reads all of them.

Usage
-----

    python scripts/check_version_lockstep.py            # metadata agreement only
    python scripts/check_version_lockstep.py --dist-dir dist   # + built artifacts

With ``--dist-dir`` the check additionally asserts the built wheel and sdist
carry exactly the same version and distribution name, so a stale or
hand-renamed artifact cannot be published even if every metadata file agrees.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]

DIST_NAME = "scientific-computing-system"
WHEEL_DIST = "scientific_computing_system"

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
PY_VERSION = re.compile(r'^__version__\s*=\s*version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
CFF_VERSION = re.compile(r"^\s*version:\s*[\"']?([^\s\"']+)[\"']?\s*$", re.MULTILINE)
CHANGELOG_HEADING = re.compile(r"^## \[v?(\d+\.\d+\.\d+)\]", re.MULTILINE)


class VersionDriftError(Exception):
    """Raised when two or more release surfaces disagree."""


def _pyproject_version() -> str:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("version"), str):
        raise VersionDriftError("pyproject.toml must contain a string [project] version")
    return project["version"]


def _package_json_version() -> str:
    path = ROOT / "package.json"
    if not path.exists():
        raise VersionDriftError("package.json is missing; the npm launcher cannot be versioned")
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if not isinstance(version, str):
        raise VersionDriftError("package.json must contain a string version")
    return version


def _python_version() -> str:
    match = PY_VERSION.search(
        (ROOT / "src" / "cds" / "_version.py").read_text(encoding="utf-8-sig")
    )
    if match is None:
        raise VersionDriftError("src/cds/_version.py must define __version__ = version = X.Y.Z")
    return match.group(1)


def _citation_versions() -> tuple[str, str]:
    matches = CFF_VERSION.findall((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    if len(matches) != 2:
        raise VersionDriftError(
            "CITATION.cff must contain exactly two `version:` fields "
            "(top-level and preferred-citation)"
        )
    return matches[0], matches[1]


def changelog_versions() -> list[str]:
    """Every released version heading in ``CHANGELOG.md``, newest first."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    seen: list[str] = []
    for version in CHANGELOG_HEADING.findall(text):
        if version not in seen:
            seen.append(version)
    return seen


def collected_versions() -> dict[str, str]:
    """Every metadata surface mapped to the version it declares."""
    citation, preferred = _citation_versions()
    return {
        "pyproject.toml [project].version": _pyproject_version(),
        "package.json version": _package_json_version(),
        "src/cds/_version.py __version__": _python_version(),
        "CITATION.cff top-level version": citation,
        "CITATION.cff preferred-citation version": preferred,
    }


def check_importable_version(expected: str) -> None:
    """Assert ``cds.__version__`` really equals the declared version."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import cds  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import failure is the bug
        raise VersionDriftError(f"cannot import the cds package: {exc}") from exc
    actual = getattr(cds, "__version__", None)
    if actual != expected:
        raise VersionDriftError(
            f"cds.__version__ is {actual!r} but pyproject declares {expected!r}; "
            "the re-export in src/cds/__init__.py is stale or missing"
        )


def check_changelog(expected: str) -> None:
    """Assert the changelog documents the version being released."""
    versions = changelog_versions()
    if expected not in versions:
        raise VersionDriftError(
            f"CHANGELOG.md has no `## [v{expected}]` heading; released versions "
            f"present: {versions[:5]}. Every release must be documented."
        )


def check_built_artifacts(dist_dir: Path, expected: str) -> list[str]:
    """Assert the built wheel and sdist match ``expected`` exactly."""
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1:
        raise VersionDriftError(f"expected exactly one wheel in {dist_dir}, found {len(wheels)}")
    if len(sdists) != 1:
        raise VersionDriftError(f"expected exactly one sdist in {dist_dir}, found {len(sdists)}")

    wheel, sdist = wheels[0], sdists[0]
    verified: list[str] = []

    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise VersionDriftError(
                f"{wheel.name}: expected one METADATA, found {len(metadata_names)}"
            )
        metadata = archive.read(metadata_names[0]).decode("utf-8")
    fields = _metadata_fields(metadata)
    _require_field(wheel.name, fields, "Name", DIST_NAME)
    _require_field(wheel.name, fields, "Version", expected)
    verified.append(f"{wheel.name}: {fields['Name']}=={fields['Version']}")

    prefix = f"{WHEEL_DIST}-{expected}/"
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        pkg_info = [n for n in names if n == f"{prefix}PKG-INFO"]
        if not pkg_info:
            raise VersionDriftError(f"{sdist.name}: PKG-INFO not found under {prefix}")
        sdist_metadata = archive.extractfile(pkg_info[0])
        assert sdist_metadata is not None  # noqa: S101 - tarfile guarantees a readable member
        sdist_fields = _metadata_fields(sdist_metadata.read().decode("utf-8"))
    _require_field(sdist.name, sdist_fields, "Name", DIST_NAME)
    _require_field(sdist.name, sdist_fields, "Version", expected)
    verified.append(f"{sdist.name}: {sdist_fields['Name']}=={sdist_fields['Version']}")

    if wheel.name != f"{WHEEL_DIST}-{expected}-py3-none-any.whl":
        raise VersionDriftError(
            f"unexpected wheel filename {wheel.name!r} for version {expected}; "
            f"expected {WHEEL_DIST}-{expected}-py3-none-any.whl"
        )
    if sdist.name != f"{WHEEL_DIST}-{expected}.tar.gz":
        raise VersionDriftError(
            f"unexpected sdist filename {sdist.name!r} for version {expected}; "
            f"expected {WHEEL_DIST}-{expected}.tar.gz"
        )
    return verified


def _metadata_fields(metadata: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in metadata.splitlines():
        if not line.strip():
            break  # headers end at the first blank line (body is the long description)
        key, separator, value = line.partition(": ")
        if separator:
            fields.setdefault(key, value.strip())
    return fields


def _require_field(artifact: str, fields: dict[str, str], key: str, expected: str) -> None:
    actual = fields.get(key)
    if actual != expected:
        raise VersionDriftError(f"{artifact}: {key} is {actual!r}, expected {expected!r}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=None,
        help="also verify the built wheel/sdist in this directory against the declared version",
    )
    args = parser.parse_args(argv)

    try:
        versions = collected_versions()
        declared = versions["pyproject.toml [project].version"]

        if SEMVER.fullmatch(declared) is None:
            raise VersionDriftError(f"version must be X.Y.Z, got {declared!r}")

        mismatched = {source: value for source, value in versions.items() if value != declared}
        if mismatched:
            detail = ", ".join(
                f"{source}={value!r}" for source, value in sorted(mismatched.items())
            )
            raise VersionDriftError(f"version drift against {declared}: {detail}")

        check_importable_version(declared)
        check_changelog(declared)

        print(f"Version lockstep verified at {declared} across {len(versions)} metadata surfaces:")
        for source in sorted(versions):
            print(f"  {source} = {versions[source]}")
        print("  cds.__version__ (runtime import) = " + declared)
        print("  CHANGELOG.md documents v" + declared)

        if args.dist_dir is not None:
            for line in check_built_artifacts(args.dist_dir, declared):
                print(f"  built artifact verified: {line}")
    except VersionDriftError as exc:
        print(f"VERSION LOCKSTEP FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
