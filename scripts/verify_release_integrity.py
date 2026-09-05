"""Verify the public PyPI release and metadata-only GitHub Release policy.

This is a release safety gate. PyPI must expose exactly one wheel and one sdist
with valid SHA-256 metadata. The matching GitHub Release must exist but must not
store wheel or sdist distribution assets; GitHub is release metadata only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import tomllib

DIST_NAME = "scientific-computing-system"
DEFAULT_REPOSITORY = "Furox-Art/scientific-computing-system"
DEFAULT_ATTEMPTS = 12
DEFAULT_DELAY_SECONDS = 5.0


@dataclass(frozen=True)
class Artifact:
    """One published PyPI distribution artifact."""

    filename: str
    size: int
    sha256: str


def project_version() -> str:
    """Read the checked-out package version."""
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(project_file.read_text(encoding="utf-8"))["project"]
    version = project["version"]
    if not isinstance(version, str):
        raise TypeError("project.version must be a string")
    return version


def _read_json(url: str, *, token: str | None = None) -> dict[str, object]:
    headers = {
        "Accept": "application/vnd.github+json" if "api.github.com" in url else "application/json",
        "User-Agent": "scientific-computing-system-release-integrity",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object from {url}")
    return payload


def _is_distribution(filename: str) -> bool:
    return filename.endswith(".whl") or filename.endswith(".tar.gz")


def pypi_manifest(version: str) -> dict[str, Artifact]:
    """Return the exact wheel + sdist manifest from public PyPI."""
    payload = _read_json(f"https://pypi.org/pypi/{DIST_NAME}/{version}/json")
    urls = payload.get("urls")
    if not isinstance(urls, list):
        raise TypeError("PyPI response does not contain a urls list")

    artifacts: dict[str, Artifact] = {}
    package_types: set[str] = set()
    for raw in urls:
        if not isinstance(raw, dict):
            continue
        package_type = raw.get("packagetype")
        filename = raw.get("filename")
        if package_type not in {"bdist_wheel", "sdist"} or not isinstance(filename, str):
            continue
        digests = raw.get("digests")
        if not isinstance(digests, dict) or not isinstance(digests.get("sha256"), str):
            raise ValueError(f"PyPI artifact {filename} has no SHA-256 digest")
        size = raw.get("size")
        if not isinstance(size, int):
            raise ValueError(f"PyPI artifact {filename} has no integer size")
        if filename in artifacts:
            raise ValueError(f"duplicate PyPI artifact filename: {filename}")
        artifacts[filename] = Artifact(filename, size, digests["sha256"])
        package_types.add(package_type)

    if package_types != {"bdist_wheel", "sdist"} or len(artifacts) != 2:
        raise ValueError(
            f"PyPI release must contain exactly one wheel and one sdist; found {sorted(artifacts)}"
        )
    return artifacts


def github_distribution_assets(
    repository: str,
    tag: str,
    *,
    token: str | None = None,
) -> tuple[str, ...]:
    """Return forbidden wheel/sdist asset names from the matching GitHub Release."""
    payload = _read_json(
        f"https://api.github.com/repos/{repository}/releases/tags/{tag}",
        token=token,
    )
    release_tag = payload.get("tag_name")
    if release_tag != tag:
        raise ValueError(f"GitHub Release tag mismatch: expected {tag!r}, found {release_tag!r}")

    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise TypeError("GitHub release response does not contain an assets list")

    forbidden: list[str] = []
    for raw in assets:
        if not isinstance(raw, dict):
            continue
        filename = raw.get("name")
        if isinstance(filename, str) and _is_distribution(filename):
            forbidden.append(filename)
    return tuple(sorted(forbidden))


def verify_release_integrity(
    version: str,
    repository: str,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    token: str | None = None,
) -> None:
    """Verify PyPI completeness and the asset-free GitHub Release policy."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")

    tag = f"v{version}"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"Release integrity attempt {attempt}/{attempts}: {repository} {tag}")
            artifacts = pypi_manifest(version)
            forbidden = github_distribution_assets(repository, tag, token=token)
            if forbidden:
                raise ValueError(
                    "GitHub Release must not contain wheel/sdist distribution assets; "
                    f"found {list(forbidden)}"
                )
            for artifact in artifacts.values():
                print(f"PYPI {artifact.filename} size={artifact.size} sha256={artifact.sha256}")
            print(
                f"Release integrity verified: {DIST_NAME} {version} is on PyPI and {tag} "
                "contains no GitHub distribution assets"
            )
            return
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay_seconds)

    if last_error is not None:
        raise RuntimeError(f"release integrity could not be verified: {last_error}") from last_error
    raise RuntimeError("release integrity could not be verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--repository", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY)
    )
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS)
    args = parser.parse_args()

    verify_release_integrity(
        args.version or project_version(),
        args.repository,
        attempts=args.attempts,
        delay_seconds=args.delay_seconds,
        token=os.environ.get("GITHUB_TOKEN"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
