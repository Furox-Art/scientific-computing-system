#!/usr/bin/env python3
"""Fail CI when the release toolchain can drift from the metadata it must validate.

The gap this closes
-------------------

A sibling repository failed a real PyPI release because ``twine`` was installed
unpinned. twine<= 6.2.0 replaces ``packaging.metadata._VALID_METADATA_VERSIONS``
at *import* time with a hardcoded list ending at ``2.4``, discarding packaging's
own knowledge of ``2.5``. ``twine check --strict`` then rejects a perfectly valid
artifact:

    InvalidDistribution: Invalid distribution metadata: '2.5' is not a valid
    metadata version

Nothing is wrong with the package; the validator is wrong. Measured here:

    twine 6.2.0, Metadata-Version 2.4 -> PASSED
    twine 6.2.0, Metadata-Version 2.5 -> ERROR InvalidDistribution
    twine 7.0.0, Metadata-Version 2.5 -> PASSED

and the mutation is observable from Python:

    before ``import twine.package``: 2.2 2.3 2.4 2.5 2.6
    after  ``import twine.package``: 2.0 2.1 2.2 2.3 2.4     <- twine 6.2.0
    after  ``import twine.package``: 2.2 2.3 2.4 2.5 2.6     <- twine 7.0.0

Floating a validator is only safe if something asserts the relationship between
the pinned validator and the metadata version actually emitted. That is this
script.

What is fatal, and what is not
-------------------------------

Fatal (exit 1) -- these are decidable offline and are defects in the repository:

* ``requirements-build.lock`` does not pin ``build`` and ``twine`` with ``==``
  and a ``--hash`` for each.
* a workflow installs ``build`` or ``twine`` ad hoc instead of from that lock
* a workflow runs ``twine check`` without ``--strict``
* a workflow runs ``twine check`` in a job that never installs the hash-locked
  toolchain, so the pin it appears to use is not the pin in effect
* the pinned twine cannot validate the emitted ``Metadata-Version``

Warning only (does not fail) -- these need the network and say nothing about
whether the repository is internally consistent:

* a pinned version could not be confirmed to exist on PyPI
* action commit SHAs could not be confirmed upstream

The distinction is deliberate. A sibling guard "verified" pins by calling GitHub
unauthenticated, got rate-limited to 60 requests/hour, silently verified nothing,
and reported success. Here the offline checks are the ones with teeth, and every
network lookup degrades to a warning that names what it could not confirm.

Usage
-----

    python scripts/check_release_toolchain.py
    python scripts/check_release_toolchain.py --dist-dir dist

With ``--dist-dir`` the ``Metadata-Version`` is read from the built artifacts and
checked against the pinned twine. Without it, that one check reports that it was
skipped rather than silently passing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
BUILD_LOCK = ROOT / "requirements-build.lock"

#: The last twine release that overrode packaging's metadata-version list.
TWINE_OVERRIDE_LAST = (6, 2, 0)
#: ...and the highest Metadata-Version that override still knew about.
TWINE_OVERRIDE_MAX_METADATA = (2, 4)

#: Packages whose version determines what the artifacts look like or how they
#: are validated. Both must be exactly pinned.
TOOLCHAIN_PACKAGES = ("build", "twine")

LOCK_PIN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s\\]+)\s*\\?\s*$")
# A digest line may end with a line-continuation backslash when the requirement
# has several artifacts recorded (charset-normalizer publishes 171 wheels), so
# the trailing `\ ` is part of the format rather than something to reject.
HASH_LINE = re.compile(r"^--hash=sha256:(?P<digest>[0-9a-f]{64})\s*\\?\s*$")
JOB_HEADER = re.compile(r"^  (?P<name>[A-Za-z0-9_-]+):\s*$")
RUN_START = re.compile(r"^(?P<indent>\s*)run:\s*(?P<inline>.*)$")
ACTION_REF = re.compile(r"uses:\s*(?P<ref>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@(?P<sha>[0-9a-f]{40})")
ADHOC_INSTALL = re.compile(r"pip\s+install\s+(?P<args>[^\n|;&]*)")


class ToolchainError(Exception):
    """A defect that makes the release toolchain untrustworthy."""


# --------------------------------------------------------------------------
# lock file
# --------------------------------------------------------------------------


def parse_lock(path: Path) -> dict[str, dict[str, object]]:
    """Parse a ``--require-hashes`` lock into ``{name: {version, hashes}}``."""
    entries: dict[str, dict[str, object]] = {}
    pending: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--hash"):
            if pending is None:
                raise ToolchainError(f"{path.name}: --hash with no preceding requirement")
            digest = HASH_LINE.match(line)
            if digest is None:
                raise ToolchainError(f"{path.name}: malformed hash line: {line!r}")
            entries[pending]["hashes"].append(digest.group("digest"))  # type: ignore[union-attr]
            continue
        pin = LOCK_PIN.match(line)
        if pin is None:
            raise ToolchainError(
                f"{path.name}: requirement is not an exact '==' pin: {line!r}. A range or "
                "bare name lets the release toolchain drift between runs."
            )
        key = pin.group("name").lower().replace("_", "-").replace(".", "-")
        if key in entries:
            raise ToolchainError(f"{path.name}: duplicate requirement {key}")
        entries[key] = {"version": pin.group("version"), "hashes": []}
        pending = key
    return entries


def check_lock(lock: dict[str, dict[str, object]]) -> list[str]:
    """Require build and twine to be exactly pinned and hash-verified."""
    problems: list[str] = []
    for package in TOOLCHAIN_PACKAGES:
        entry = lock.get(package)
        if entry is None:
            problems.append(
                f"{BUILD_LOCK.name} does not pin {package}. It must be listed there so the "
                f"pin is hash-verified; '--require-hashes' refuses unhashed command-line "
                f"requirements, so '{package}' cannot be pinned on the command line."
            )
            continue
        if not re.fullmatch(r"\d+(\.\d+)*", str(entry["version"])):
            problems.append(f"{package} is pinned to {entry['version']!r}, which is not exact")
        if not entry["hashes"]:
            problems.append(f"{package}=={entry['version']} has no --hash entry")
    return problems


# --------------------------------------------------------------------------
# workflow parsing (stdlib only -- these run in the minimal build venv)
# --------------------------------------------------------------------------


def iter_jobs(text: str):
    """Yield ``(job_name, job_body)`` for each job under ``jobs:``."""
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
    inside = False
    for index, line in enumerate(lines):
        if line.rstrip() == "jobs:":
            inside = True
            continue
        if not inside:
            continue
        match = JOB_HEADER.match(line)
        if match:
            starts.append((index, match.group("name")))
    for position, (index, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        yield name, "\n".join(lines[index:end])


def run_blocks(body: str) -> list[str]:
    """Return the text of every ``run:`` block in a job body.

    Handles inline, literal (``|``) and folded (``>-``) styles. The YAML is not
    parsed with a library on purpose: the release job installs a hash-locked
    toolchain that does not include PyYAML, and a guard that cannot run in the
    job it guards is not a guard.
    """
    lines = body.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        match = RUN_START.match(lines[index])
        if match is None:
            index += 1
            continue
        indent = len(match.group("indent"))
        collected = [match.group("inline")]
        index += 1
        while index < len(lines):
            line = lines[index]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            collected.append(line)
            index += 1
        blocks.append("\n".join(collected))
    return blocks


def _strip_comment(line: str) -> str:
    return line.split(" #", 1)[0] if " #" in line else line


def check_workflows() -> list[str]:
    """Every ``twine check`` must be strict and run in a hash-locked job."""
    problems: list[str] = []
    if not WORKFLOWS.is_dir():
        raise ToolchainError(f"{WORKFLOWS} does not exist")
    saw_twine_check = False

    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        for job, body in iter_jobs(text):
            blocks = run_blocks(body)
            joined = "\n".join(blocks)
            installs_lock = "requirements-build.lock" in joined

            # Ad-hoc installs of the two toolchain packages bypass the lock.
            for block in blocks:
                for raw in block.splitlines():
                    line = _strip_comment(raw)
                    if "pip install" not in line and "pip3 install" not in line:
                        continue
                    match = ADHOC_INSTALL.search(line)
                    if match is None:
                        continue
                    args = match.group("args")
                    for package in TOOLCHAIN_PACKAGES:
                        pattern = rf"(?<![\w-]){package}(?![\w-])"
                        if re.search(pattern, args) and "requirements-build.lock" not in line:
                            problems.append(
                                f"{path.name}:{job} installs {package} ad hoc "
                                f"(`{line.strip()}`) instead of from "
                                f"{BUILD_LOCK.name}; the version in effect would be "
                                "whatever the index serves that day."
                            )

            for block in blocks:
                if "twine check" not in block:
                    continue
                saw_twine_check = True
                if "--strict" not in block:
                    problems.append(
                        f"{path.name}:{job} runs `twine check` without --strict. Without it, "
                        "rendering problems that --strict would reject pass silently."
                    )
                if not installs_lock:
                    problems.append(
                        f"{path.name}:{job} runs `twine check` but never installs "
                        f"{BUILD_LOCK.name} in that job, so the twine actually executing is "
                        "not the pinned one and the pin cannot be relied on."
                    )

    if not saw_twine_check:
        problems.append(
            "No workflow runs `twine check`. The artifacts' metadata is then never "
            "rendered before upload, so an unrenderable artifact can only be discovered "
            "by the indexer after release."
        )
    return problems


# --------------------------------------------------------------------------
# twine / Metadata-Version compatibility
# --------------------------------------------------------------------------


def parse_version(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


def _fmt(value: tuple[int, ...]) -> str:
    """Render a version tuple, so message formatting cannot index out of range."""
    return ".".join(str(part) for part in value)


def twine_can_validate(twine_version: str, metadata_version: str) -> bool:
    """Can this twine validate an artifact with this ``Metadata-Version``?

    twine<= 6.2.0 caps what it accepts at 2.4 regardless of the installed
    packaging, because of the import-time override. 7.0.0 defers to packaging and
    imposes no cap of its own.
    """
    if parse_version(twine_version) > TWINE_OVERRIDE_LAST:
        return True
    return parse_version(metadata_version) <= TWINE_OVERRIDE_MAX_METADATA


def read_metadata_versions(dist_dir: Path) -> dict[str, str]:
    """Return ``{filename: Metadata-Version}`` for the wheel and sdist."""
    found: dict[str, str] = {}
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ToolchainError(f"{wheel.name}: expected exactly one METADATA")
            text = archive.read(names[0]).decode("utf-8", errors="replace")
        found[wheel.name] = _metadata_field(text, "Metadata-Version")
    for sdist in sdists:
        with tarfile.open(sdist, "r:gz") as archive:
            pkg_info = [n for n in archive.getnames() if n.endswith("PKG-INFO")]
            if not pkg_info:
                raise ToolchainError(f"{sdist.name}: PKG-INFO not found")
            handle = archive.extractfile(pkg_info[0])
            if handle is None:
                raise ToolchainError(f"{sdist.name}: PKG-INFO unreadable")
            text = handle.read().decode("utf-8", errors="replace")
        found[sdist.name] = _metadata_field(text, "Metadata-Version")
    if not found:
        raise ToolchainError(f"{dist_dir} contains no wheel or sdist to inspect")
    return found


def _metadata_field(metadata: str, key: str) -> str:
    for line in metadata.splitlines():
        if not line.strip():
            break  # headers end at the first blank line
        name, separator, value = line.partition(": ")
        if separator and name.strip() == key:
            return value.strip()
    raise ToolchainError(f"core metadata has no {key}")


def check_compatibility(
    twine_version: str, dist_dir: Path | None, declared: str | None
) -> tuple[list[str], list[str]]:
    """Fatal problems and non-fatal notes for the pin/metadata relationship."""
    problems: list[str] = []
    notes: list[str] = []

    if dist_dir is None:
        if declared is None:
            notes.append(
                "skipped the twine/Metadata-Version compatibility check: pass --dist-dir or "
                "--metadata-version. Without it a validator that cannot read the emitted "
                "metadata would go unchallenged."
            )
            return problems, notes
        versions = {"<declared>": declared}
    else:
        try:
            versions = read_metadata_versions(dist_dir)
        except ToolchainError as error:
            problems.append(str(error))
            return problems, notes

    for name, metadata_version in sorted(versions.items()):
        verdict = (
            "can validate"
            if twine_can_validate(twine_version, metadata_version)
            else "CANNOT validate"
        )
        notes.append(
            f"twine {twine_version} {verdict} {name} (Metadata-Version {metadata_version})"
        )
        if not twine_can_validate(twine_version, metadata_version):
            problems.append(
                f"twine {twine_version} cannot validate {name}: it emits Metadata-Version "
                f"{metadata_version}, and twine<={_fmt(TWINE_OVERRIDE_LAST)} "
                f"accepts at most {_fmt(TWINE_OVERRIDE_MAX_METADATA)} "
                "because it overrides packaging's metadata-version list at import time. "
                "Pin twine>=7.0.0 or stop emitting a newer Metadata-Version."
            )
    return problems, notes


# --------------------------------------------------------------------------
# network verification (warning only)
# --------------------------------------------------------------------------


def _is_github_api(url: str) -> bool:
    """Is this URL served by the GitHub REST API?

    Compares the parsed hostname, never a substring. ``"api.github.com" in url``
    also matches ``https://evil.example/?api.github.com`` and
    ``https://api.github.com.evil.test/x``, and here that comparison decides
    whether a ``GITHUB_TOKEN`` is attached to the request -- so a loose match
    would send the credential somewhere it does not belong.
    """
    return urlsplit(url).hostname == "api.github.com"


def _get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 30):
    request = urllib.request.Request(url, headers=headers or {"accept": "application/json"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and _is_github_api(url):
        request.add_header("authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def verify_pins_upstream(lock: dict[str, dict[str, object]]) -> list[str]:
    """Confirm each toolchain pin exists on PyPI. Never fatal."""
    notes: list[str] = []
    for package in TOOLCHAIN_PACKAGES:
        entry = lock.get(package)
        if entry is None:
            continue
        version = str(entry["version"])
        url = f"https://pypi.org/pypi/{package}/{version}/json"
        try:
            payload = _get_json(url)
        except urllib.error.HTTPError as error:
            notes.append(
                f"::warning::could not confirm {package}=={version} on PyPI: HTTP {error.code}. "
                "Not failing: the offline pin checks already decided this repository is sound."
            )
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            notes.append(
                f"::warning::could not reach PyPI to confirm {package}=={version}: {error}. "
                "Not failing: this is a network problem, not a repository defect."
            )
            continue
        if payload.get("info", {}).get("version") != version:
            notes.append(
                f"::warning::PyPI reports {payload.get('info', {}).get('version')!r} for "
                f"{package} instead of the pinned {version!r}"
            )
            continue
        requires = payload.get("info", {}).get("requires_dist") or []
        notes.append(
            f"confirmed {package}=={version} exists on PyPI ({len(requires)} requirements)"
        )
    return notes


def verify_action_pins() -> list[str]:
    """Confirm every action SHA resolves upstream, authenticated when possible.

    Uses ``GITHUB_TOKEN`` because unauthenticated GitHub API calls are limited to
    60/hour, which is easy to exceed and fails in a way that looks like success.
    """
    if not os.environ.get("GITHUB_TOKEN"):
        return [
            "::warning::GITHUB_TOKEN is unset, so action commit pins were not verified "
            "upstream. Unauthenticated GitHub API calls are limited to 60/hour; without a "
            "token this check would silently verify nothing."
        ]
    notes: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for match in ACTION_REF.finditer(path.read_text(encoding="utf-8")):
            ref, sha = match.group("ref"), match.group("sha")
            if (ref, sha) in seen:
                continue
            seen.add((ref, sha))
            try:
                _get_json(f"https://api.github.com/repos/{ref}/commits/{sha}")
            except urllib.error.HTTPError as error:
                notes.append(
                    f"::warning::{path.name}: {ref}@{sha[:12]} did not resolve upstream "
                    f"(HTTP {error.code}). Fix or replace the pin."
                )
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                notes.append(
                    f"::warning::could not reach the GitHub API to confirm {ref}@{sha[:12]}: "
                    f"{error}. Not failing: network, not repository."
                )
            else:
                notes.append(f"confirmed {ref}@{sha[:12]} resolves upstream")
    return notes


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=None)
    parser.add_argument(
        "--metadata-version",
        default=None,
        help="assert the pinned twine can validate this Metadata-Version instead of "
        "reading it from artifacts",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="skip the advisory upstream checks entirely",
    )
    args = parser.parse_args(argv)

    try:
        if not BUILD_LOCK.exists():
            raise ToolchainError(f"{BUILD_LOCK} does not exist")
        lock = parse_lock(BUILD_LOCK)
        problems: list[str] = []
        notes: list[str] = []

        twine_entry = lock.get("twine")
        twine_version = str(twine_entry["version"]) if twine_entry else "<unpinned>"

        problems.extend(check_lock(lock))
        problems.extend(check_workflows())
        if twine_entry is not None:
            compat_problems, compat_notes = check_compatibility(
                twine_version, args.dist_dir, args.metadata_version
            )
            problems.extend(compat_problems)
            notes.extend(compat_notes)

        print(f"Release toolchain: {len(lock)} exact pins in {BUILD_LOCK.name}")
        for package in TOOLCHAIN_PACKAGES:
            entry = lock.get(package)
            state = f"{package}=={entry['version']}" if entry else f"{package} NOT PINNED"
            hashes = len(entry["hashes"]) if entry else 0  # type: ignore[arg-type]
            print(f"  {state} ({hashes} hash(es))")
        for note in notes:
            print(f"  note: {note}")

        if not args.skip_network:
            for note in verify_pins_upstream(lock):
                print(f"  {note}")
            for note in verify_action_pins():
                print(f"  {note}")

        if problems:
            print(file=sys.stderr)
            for problem in problems:
                print(f"RELEASE TOOLCHAIN FAILED: {problem}", file=sys.stderr)
            return 1
        print("Release toolchain verified.")
        return 0
    except ToolchainError as error:
        print(f"RELEASE TOOLCHAIN FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
