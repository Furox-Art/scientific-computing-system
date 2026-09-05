"""Install the exact released package from public PyPI and verify its CLI.

This is intentionally registry-only: it never builds or installs a local wheel.
It is used after a successful release so the package users actually receive from
PyPI is exercised independently from the pre-publish wheel smoke test.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import tomllib

DIST_NAME = "scientific-computing-system"
PYPI_INDEX = "https://pypi.org/simple"
DEFAULT_ATTEMPTS = 12
DEFAULT_DELAY_SECONDS = 10.0


def project_version() -> str:
    """Read the version declared by the checked-out release source."""
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(project_file.read_text(encoding="utf-8"))["project"]
    version = project["version"]
    if not isinstance(version, str):
        raise TypeError("project.version must be a string")
    return version


def install_from_public_pypi(
    version: str,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
) -> None:
    """Install one exact version from public PyPI, retrying brief propagation lag."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")

    # GitHub-hosted runners are fresh, but remove any accidental preinstalled copy
    # so the verification is unambiguously against the registry download below.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", DIST_NAME],
        check=False,
    )

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--index-url",
        PYPI_INDEX,
        f"{DIST_NAME}=={version}",
    ]

    for attempt in range(1, attempts + 1):
        print(f"Public PyPI install attempt {attempt}/{attempts}: {DIST_NAME}=={version}")
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            return
        if attempt < attempts:
            time.sleep(delay_seconds)

    raise RuntimeError(f"could not install {DIST_NAME}=={version} from {PYPI_INDEX}")


def verify_registry_release(version: str) -> None:
    """Install from PyPI, then run the existing installed-distribution CLI gate."""
    install_from_public_pypi(version)
    verifier = Path(__file__).with_name("verify_cli_install.py")
    subprocess.run(
        [sys.executable, str(verifier), "--expected-version", version],
        check=True,
    )
    print(f"Public PyPI registry verification passed for {DIST_NAME} {version}.")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default=None)
    args = parser.parse_args()
    version = args.expected_version or project_version()
    verify_registry_release(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
