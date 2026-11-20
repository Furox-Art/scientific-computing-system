"""Guard: every provenance claim in the prose must be backed by a live registry.

Run from the repository root:

    python docs/checks/verify_provenance_claims.py                  # audit live
    python docs/checks/verify_provenance_claims.py --offline        # structure only
    python docs/checks/verify_provenance_claims.py --version 2.2.1

Why this exists
---------------
The provenance documentation once asserted, from memory, which registry carried a
build attestation. Nothing checked it, so the claim would have kept being
repeated after the next release changed the answer -- and a reviewer reading the
prose had no way to tell which statements were measured.

Three distinct things get conflated in supply-chain prose, and only one of them
is provenance:

* a **PEP 740 attestation** -- Sigstore-signed, binds a file digest to the
  publishing identity. PyPI serves these per *file*; a bare
  ``/integrity/<project>/<version>/`` directory is **not** an endpoint and 404s.
  Requesting the directory is the single easiest way to wrongly conclude that
  PyPI is unattested.
* **npm ``dist.signatures``** -- npm signing its own registry metadata for
  transport integrity. Says nothing about who built the tarball.
* a **content digest** (``sha256`` on PyPI, ``dist.integrity`` on npm) -- pins
  which bytes you get; proves nothing about origin.

Determinism
-----------
The decision logic lives in :func:`evaluate_claims`, which takes every input as
an argument and performs no I/O. ``tests/test_provenance_claims.py`` drives it
with injected registry states, so the rejection paths are unit-tested without a
network. This module's ``main`` is the thin shell that gathers live facts and
hands them over.

Network failures are reported as ``UNKNOWN``, never as ``NO``. A required CI
gate that fails on a DNS blip trains people to re-run it instead of reading it.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = sorted(
    p
    for p in (
        ROOT / "README.md",
        ROOT / "SECURITY.md",
        *sorted((ROOT / "docs").rglob("*.md")),
    )
    if p.is_file()
)

INTEGRITY_MEDIA_TYPE = "application/vnd.pypi.integrity.v1+json"
DEFAULT_TIMEOUT = 30

# Tri-state registry answers. UNKNOWN is deliberately distinct from NO.
YES = "yes"
NO = "no"
UNKNOWN = "unknown"

_STATE_TO_BOOL = {YES: True, NO: False, UNKNOWN: None}


# --------------------------------------------------------------------------
# Network layer (used only by main(); the unit test never calls these)
# --------------------------------------------------------------------------


def _get(url: str, accept: str | None = None) -> tuple[int | None, bytes]:
    """Return (status, body). ``status`` is None when the host was unreachable."""
    headers = {"User-Agent": "provenance-claim-check"}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, str(exc).encode()


def decode_attestation_sha256(body: bytes) -> str:
    """Pull the attested sha256 subject digest out of a PEP 740 bundle."""
    bundle = json.loads(body)["attestation_bundles"][-1]
    statement = bundle["attestations"][-1]["envelope"]["statement"]
    return json.loads(base64.b64decode(statement))["subject"][0]["digest"]["sha256"]


def pypi_attested(project: str, version: str, filename: str) -> tuple[str, str]:
    """Is a PEP 740 attestation served for this exact file?"""
    url = f"https://pypi.org/integrity/{project}/{version}/{filename}/provenance"
    status, body = _get(url, INTEGRITY_MEDIA_TYPE)
    if status is None:
        return UNKNOWN, f"{url} -> unreachable: {body.decode(errors='replace')}"
    if status == 404:
        return NO, f"{url} -> 404"
    if status != 200:
        return UNKNOWN, f"{url} -> HTTP {status}"
    try:
        digest = decode_attestation_sha256(body)
    except Exception as exc:  # noqa: BLE001
        return UNKNOWN, f"{url} -> unparsable bundle: {exc}"
    return YES, f"attested sha256 {digest}"


def attested_sha256(detail: str) -> str | None:
    """Extract the digest from a :func:`pypi_attested` detail string."""
    return detail.split("sha256 ", 1)[1] if "sha256 " in detail else None


def pypi_version_published(project: str, version: str) -> bool | None:
    """Does this version exist on PyPI yet?

    ``None`` when PyPI could not be read. A bump PR legitimately asks about a
    version that has not shipped, so "not published yet" must be distinguishable
    from "published and unattested".
    """
    status, _ = _get(f"https://pypi.org/pypi/{project}/{version}/json")
    if status is None:
        return None
    return status == 200


def npm_version_published(name: str, version: str) -> bool | None:
    status, _ = _get(f"https://registry.npmjs.org/{name}/{version}")
    if status is None:
        return None
    return status == 200


def pypi_published_digest(project: str, version: str, filename: str) -> str | None:
    status, body = _get(f"https://pypi.org/pypi/{project}/{version}/json")
    if status != 200:
        return None
    for entry in json.loads(body)["urls"]:
        if entry["filename"] == filename:
            return entry["digests"]["sha256"]
    return None


def npm_attested(name: str, version: str) -> tuple[str, str]:
    url = f"https://registry.npmjs.org/-/npm/v1/attestations/{name}@{version}"
    status, _ = _get(url)
    if status is None:
        return UNKNOWN, f"{url} -> unreachable"
    if status == 200:
        return YES, f"{url} -> 200"
    if status == 404:
        return NO, f"{url} -> HTTP 404"
    return UNKNOWN, f"{url} -> HTTP {status}"


def npm_dist(name: str, version: str) -> tuple[bool, str]:
    status, body = _get(f"https://registry.npmjs.org/{name}/{version}")
    if status != 200:
        return False, f"HTTP {status}"
    dist = json.loads(body).get("dist", {})
    has_sig = bool(dist.get("signatures"))
    has_int = bool(dist.get("integrity"))
    return (
        has_int,
        f"dist.integrity={has_int} dist.shasum={bool(dist.get('shasum'))} "
        f"dist.signatures={has_sig}",
    )


# --------------------------------------------------------------------------
# Prose parsing (pure)
# --------------------------------------------------------------------------


def cell_verdict(cell: str, label: str = "") -> bool | None:
    """Interpret a table cell as a claim about attestation presence.

    The row *label* supplies the subject ("PEP 740 provenance attestation"), so a
    cell that only says "yes on 2.2.1" is still a checkable claim. Matching the
    whole line instead would be useless: the label always contains the words
    "provenance attestation", so such a regex fires no matter what the cell says.
    """
    low = cell.lower()
    if not re.search(r"(attestation|sigstore|pep 740)", (label + " " + cell).lower()):
        return None
    if re.search(r"\b(no|zero|none|absent|never)\b|\bnot\b", low) and "not **yet**" not in low:
        return False
    if re.search(r"yes|\bhas\b|\bcarries\b|is served|present", low):
        return True
    return None


def read_attestation_row() -> tuple[bool | None, bool | None, str]:
    """Return (pypi_claimed, npm_claimed, raw_row) from SECURITY.md's table."""
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    for line in security.splitlines():
        if "PEP 740 provenance attestation" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            label = cells[0]
            return cell_verdict(cells[1], label), cell_verdict(cells[2], label), line.strip()
    return None, None, ""


def read_readme_npm_claim() -> tuple[bool | None, str]:
    """Read the README's npm bullet and return its attestation claim."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for line in readme.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- **npm**"):
            continue
        low = stripped.lower()
        if "attestation" not in low:
            return None, stripped
        if re.search(r"carries? \*\*no\*\*|carries? no|not attested|zero attestation", low):
            return False, stripped
        if re.search(r"carries? a|has a|with a", low):
            return True, stripped
        return None, stripped
    return None, ""


def prose_claims(text: str) -> dict[str, bool]:
    """Which provenance assertions does this file make? Conservative matching."""
    lowered = text.lower()
    return {
        "mentions_pypi_attestation": bool(
            re.search(r"pypi[^\n]{0,240}?(pep 740|provenance attestation)", lowered)
        ),
        "mentions_npm_attestation_status": bool(
            re.search(r"npm[^\n]{0,240}?(no|not|zero|cannot|404)[^\n]{0,80}?attestation", lowered)
            or "attestation endpoint returns 404" in lowered
        ),
        "distinguishes_npm_transport_sig": "dist.signatures" in lowered,
        "distinguishes_digest_from_provenance": bool(
            re.search(r"content digest", lowered) or "integrity pin" in lowered
        ),
        "states_registration_pending": bool(
            re.search(
                r"pending registration|not yet in place|is not registered|not \*\*yet\*\*",
                lowered,
            )
        ),
    }


# --------------------------------------------------------------------------
# Decision (pure) -- the unit test drives exactly this
# --------------------------------------------------------------------------


def evaluate_claims(
    *,
    pypi_claim: bool | None,
    npm_claim: bool | None,
    readme_npm_claim: bool | None,
    pypi_state: str,
    npm_state: str,
    attested_digest: str | None = None,
    published_digest: str | None = None,
    all_text: str = "",
) -> list[str]:
    """Compare the prose against the registry facts. Pure: no I/O, no globals.

    Every input is injected, which is what lets the unit test exercise the
    rejection paths deterministically without touching the network. Returns a
    list of human-readable failures; empty means the prose is consistent.
    """
    failures: list[str] = []
    pypi_fact = _STATE_TO_BOOL[pypi_state]
    npm_fact = _STATE_TO_BOOL[npm_state]

    if pypi_claim is None and npm_claim is None:
        failures.append(
            "could not parse the 'PEP 740 provenance attestation' row in SECURITY.md; "
            "the guard cannot verify a table it cannot read"
        )
    if pypi_claim is not None and pypi_fact is not None and pypi_claim != pypi_fact:
        failures.append(
            f"SECURITY.md claims PyPI attestation={pypi_claim}, live registries say {pypi_fact}"
        )
    if npm_claim is not None and npm_fact is not None and npm_claim != npm_fact:
        failures.append(
            f"SECURITY.md claims npm attestation={npm_claim}, live registries say {npm_fact}"
        )
    if readme_npm_claim is not None and npm_fact is not None and readme_npm_claim != npm_fact:
        failures.append(
            f"README claims npm attestation={readme_npm_claim}, live registries say {npm_fact}"
        )

    if (
        attested_digest is not None
        and published_digest is not None
        and attested_digest != published_digest
    ):
        failures.append(
            f"attested subject {attested_digest} != PyPI published sha256 {published_digest}"
        )

    # The prose must not describe a digest mismatch as acceptable.
    if re.search(
        r"(unrelated to|need not match|does not need to match) the published digest",
        all_text,
        re.I,
    ):
        failures.append("prose permits the attested subject to differ from the published digest")

    # Npm integrity pin must be described as a pin, not as provenance. Only the
    # positive claim is rejected: "not proof of origin" is correct prose.
    for match in re.finditer(r"dist\.integrity[^\n]{0,140}", all_text, re.I):
        window = match.group(0)
        if re.search(r"(attests|proves who|proof of origin)", window, re.I) and not re.search(
            r"(not|never|no|rather than|only)\s+(a\s+)?(proof|attest|provenance)",
            window,
            re.I,
        ):
            failures.append(f"prose overstates dist.integrity as provenance: {window[:80]!r}")

    return failures


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def scan_notes() -> tuple[list[str], bool]:
    notes = ["prose claim scan:"]
    any_claim = False
    for path in DOCS:
        made = [k for k, v in prose_claims(path.read_text(encoding="utf-8")).items() if v]
        if made:
            any_claim = True
            notes.append(f"  {path.relative_to(ROOT)}: {', '.join(made)}")
    return notes, any_claim


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="structure checks only")
    parser.add_argument(
        "--version",
        default="2.2.1",
        help="published version whose live state the prose is checked against",
    )
    parser.add_argument(
        "--strict-network",
        action="store_true",
        help="fail if a registry could not be reached (default: warn and continue)",
    )
    args = parser.parse_args(argv)

    notes, any_claim = scan_notes()
    failures: list[str] = []
    if not any_claim:
        failures.append("no provenance claim found in the prose; the guard has nothing to check")

    if args.offline:
        notes.append("offline mode: skipping live registry comparison")
        print("\n".join(notes))
        if failures:
            print("\nFAIL:")
            for f in failures:
                print("  -", f)
            return 1
        print("\nOK (offline)")
        return 0

    project = "scientific-computing-system"
    filename = f"{project.replace('-', '_')}-{args.version}-py3-none-any.whl"

    notes.append(f"\nlive registry checks (version {args.version}):")

    # A version that has not shipped yet is not "unattested". A bump PR asks
    # about a version no registry has, and that must not be reported as a
    # missing attestation -- nor must it fail the build.
    pypi_exists = pypi_version_published(project, args.version)
    npm_exists = npm_version_published(project, args.version)
    notes.append(f"  PyPI has this version: {pypi_exists}")
    notes.append(f"  npm has this version:  {npm_exists}")

    if pypi_exists is False or npm_exists is False:
        notes.append(
            f"  NOTICE: {args.version} is not published on "
            f"{'PyPI' if pypi_exists is False else 'npm'}; attestation state for it "
            "is unverifiable, so no claim is checked against it"
        )
        if args.strict_network:
            failures.append(f"version {args.version} is not published on both registries")
        print("\n".join(notes))
        print()
        if failures:
            print("FAIL:")
            for f in failures:
                print("  -", f)
            return 1
        print(f"OK: {args.version} is not published yet; nothing to verify")
        return 0

    pypi_state, pypi_detail = pypi_attested(project, args.version, filename)
    notes.append(f"  PyPI attestation: {pypi_state}  [{pypi_detail}]")

    npm_state, npm_detail = npm_attested(project, args.version)
    notes.append(f"  npm attestation:   {npm_state}  [{npm_detail}]")

    _, npm_dist_detail = npm_dist(project, args.version)
    notes.append(f"  npm dist fields:   {npm_dist_detail}")

    published = pypi_published_digest(project, args.version, filename)
    notes.append(f"  PyPI published sha256: {published}")

    attested = attested_sha256(pypi_detail) if pypi_state == YES else None
    if attested and published and attested == published:
        notes.append("  attested subject matches PyPI published sha256: yes")

    for label, state in (("PyPI", pypi_state), ("npm", npm_state)):
        if state == UNKNOWN:
            message = f"{label} registry could not be read; claims about it were not verified"
            if args.strict_network:
                failures.append(message)
            else:
                notes.append(f"  NOTICE: {message}")

    pypi_claim, npm_claim, row = read_attestation_row()
    notes.append(f"\nSECURITY.md attestation row: {row[:110]}")
    notes.append(f"  parsed claim -> pypi_attested={pypi_claim} npm_attested={npm_claim}")

    rd_npm_claim, rd_line = read_readme_npm_claim()
    notes.append(f"README npm bullet: {rd_line[:110]}")
    notes.append(f"  parsed claim -> npm_attested={rd_npm_claim}")

    all_text = "\n".join(p.read_text(encoding="utf-8") for p in DOCS)
    failures.extend(
        evaluate_claims(
            pypi_claim=pypi_claim,
            npm_claim=npm_claim,
            readme_npm_claim=rd_npm_claim,
            pypi_state=pypi_state,
            npm_state=npm_state,
            attested_digest=attested,
            published_digest=published,
            all_text=all_text,
        )
    )

    print("\n".join(notes))
    print()
    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("OK: every provenance claim in the prose matches the live registries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
