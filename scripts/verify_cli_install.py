"""Smoke-test the installed ``cds`` console script and module entry point.

This script deliberately runs commands in a temporary directory so a source
checkout cannot shadow the installed wheel.  It is used by CI and the release
workflow as a packaging/entry-point gate before anything is published to PyPI.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path


DIST_NAME = "scientific-computing-system"


def _run(
    command: list[str],
    *,
    cwd: Path,
    stdin: str | None = None,
    expected_text: str | None = None,
) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{output}"
        )
    if expected_text is not None and expected_text not in output:
        raise RuntimeError(
            f"command output did not contain {expected_text!r}: {' '.join(command)}\n{output}"
        )
    print(f"PASS: {' '.join(command)}")


def verify_cli(expected_version: str | None = None) -> None:
    """Run representative commands against the installed distribution."""
    installed_version = version(DIST_NAME)
    if expected_version is not None and installed_version != expected_version:
        raise RuntimeError(
            f"installed version is {installed_version}, expected {expected_version}"
        )

    with tempfile.TemporaryDirectory(prefix="cds-cli-smoke-") as temp_dir:
        cwd = Path(temp_dir)
        commands: list[tuple[list[str], str | None, str | None]] = [
            (["cds", "--version"], None, installed_version),
            ([sys.executable, "-m", "cds", "--version"], None, installed_version),
            (["cds", "--help"], None, "Scientific Computing System"),
            (["cds", "version"], None, installed_version),
            (["cds", "modules"], None, "cds."),
            (["cds", "info"], None, "Version:"),
            (["cds", "constants"], None, "Physical Constants"),
            (["cds", "stats", "1,2,3,4"], None, "Descriptive stats"),
            (["cds", "sample", "gaussian", "-n", "3", "--seed", "123"], None, None),
            (
                ["cds", "integrate", "x2", "--a", "0", "--b", "1", "-n", "100"],
                None,
                "x2(x)",
            ),
            (["cds", "plot", "1,3,2"], None, "CLI Plot"),
            (["cds", "prompt", "Does X affect Y?", "--num", "1"], None, "Does X affect Y?"),
            (
                ["cds", "hypothesis", "Does X affect Y?", "--num", "1", "--dry-run"],
                None,
                "Dry run mode",
            ),
            (["cds", "benchmark"], None, "Benchmarking"),
            (["cds", "calc", "ke"], "2\n3\n", "Kinetic Energy"),
            (["cds", "dashboard", "--help"], None, "dashboard"),
        ]
        for command, stdin, expected_text in commands:
            _run(command, cwd=cwd, stdin=stdin, expected_text=expected_text)

    print(f"Installed CLI verification passed for {DIST_NAME} {installed_version}.")


def main() -> int:
    """CLI entry point for the verification script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default=None)
    args = parser.parse_args()
    verify_cli(args.expected_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
