"""Guard: every provenance claim in the prose must be backed by a live registry.

Run from the repository root:

    python docs/checks/verify_provenance_claims.py            # audit live, exit 1 on drift
    python docs/checks/verify_provenance_claims.py --offline  # structure only, no network

Why this exists
---------------
An earlier revision of this documentation asserted that one channel carried a
provenance attestation and the other did not. That assertion was correct at the
time, but nothing checked it, so it would have kept being asserted after the next
release changed the answer -- and a reviewer reading the prose had no way to tell
which claims were measured and which were remembered.

Three distinct things get conflated in supply-chain prose, and only one of them
is provenance:

* a **PEP 740 attestation** -- Sigstore-signed, binds a digest to the publishing
  identity. PyPI serves these per *file*; a bare ``/integrity/<p>/<v>/``
  directory is not an endpoint and returns 404.
* **npm ``dist.signatures``** -- npm signing its own registry metadata for
  transport integrity. Not build provenance.
* a **content digest** (``sha256`` on PyPI, ``dist.integrity`` on npm) -- pins
  which bytes you get; proves nothing about origin.

The check below fails if the prose asserts something the registries contradict.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = sorted(
    p
    for p in (ROOT / "README.md", ROOT / "SECURITY.md", *sorted((ROOT / "docs").rglob("*.md")))
    if p.is_file()
)

INTEGRITY_MEDIA_TYPE = "application/vnd.pypi.integrity.v1+json"
DEFAULT_TIMEOUT = 30


def _get(url: str, accept: str | None = None) -> tuple[int, bytes]:
    headers = {"User-Agent": "provenance-claim-check"}
    if accept:
        headers["Accept"] = accept
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=DEFAULT_TIMEOUT
        ) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def pypi_attested(project: str, version: str, filename: str) -> tuple[bool, str]:
    """Is a PEP 740 attestation served for this exact file?"""
    url = f"https://pypi.org/integrity/{project}/{version}/{filename}/provenance"
    status, body = _get(url, INTEGRITY_MEDIA_TYPE)
    if status == 404:
        return False, f"{url} -> 404"
    if status != 200:
        return False, f"{url} -> HTTP {status}"
    try:
        bundle = json.loads(body)["attestation_bundles"][-1]
        subject = bundle["attestations"][-1]["envelope"]["statement"]
        import base64

        statement = json.loads(base64.b64decode(subject))
        digest = statement["subject"][0]["digest"]["sha256"]
    except Exception as exc:  # noqa: BLE001
        return False, f"{url} -> unparsable bundle: {exc}"
    return True, f"attested sha256 {digest}"


def pypi_published_digest(project: str, version: str, filename: str) -> str | None:
    status, body = _get(f"https://pypi.org/pypi/{project}/{version}/json")
    if status != 200:
        return None
    for entry in json.loads(body)["urls"]:
        if entry["filename"] == filename:
            return entry["digests"]["sha256"]
    return None


def npm_attested(name: str, version: str) -> tuple[bool, str]:
    url = f"https://registry.npmjs.org/-/npm/v1/attestations/{name}@{version}"
    status, _ = _get(url)
    if status == 200:
        return True, f"{url} -> 200"
    return False, f"{url} -> HTTP {status}"


def npm_dist(name: str, version: str) -> tuple[bool, str]:
    status, body = _get(f"https://registry.npmjs.org/{name}/{version}")
    if status != 200:
        return False, f"HTTP {status}"
    dist = json.loads(body).get("dist", {})
    has_sig = bool(dist.get("signatures"))
    has_int = bool(dist.get("integrity"))
    return (
        has_int,
        f"dist.integrity={has_int} dist.shasum={bool(dist.get('shasum'))} dist.signatures={has_sig}",
    )


def _cell_verdict(cell: str, label: str = "") -> bool | None:
    """Interpret a table cell as a claim about attestation presence.

    The row *label* supplies the subject ("PEP 740 provenance attestation"), so a
    cell that only says "yes on 2.2.1" is still a checkable claim. Bare substring
    matching over the whole line is not enough in either direction: the label
    contains the words "provenance attestation", so such a regex matches whatever
    the cell happens to say.
    """
    low = cell.lower()
    context = (label + " " + cell).lower()
    if not re.search(r"(attestation|sigstore|pep 740)", context):
        return None
    negated = re.search(r"\b(no|zero|none|absent|never)\b|\bnot\b", low)
    if negated and not re.search(r"not \*\*yet\*\*", low):
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
            return _cell_verdict(cells[1], label), _cell_verdict(cells[2], label), line.strip()
    return None, None, ""


def read_readme_npm_claim() -> tuple[bool | None, str]:
    """Read the README's npm bullet and return its attestation claim."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for line in readme.splitlines():
        stripped = line.strip()
        if stripped.startswith("- **npm**"):
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
                r"pending registration|not yet in place|is not registered|not \*\*yet\*\*", lowered
            )
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="structure checks only")
    ap.add_argument("--version", default="2.2.1")
    args = ap.parse_args()

    failures: list[str] = []
    notes: list[str] = []

    notes.append("prose claim scan:")
    any_claim = False
    for path in DOCS:
        claims = prose_claims(path.read_text(encoding="utf-8"))
        made = [k for k, v in claims.items() if v]
        if made:
            any_claim = True
            notes.append(f"  {path.relative_to(ROOT)}: {', '.join(made)}")
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
    pypi_ok, pypi_detail = pypi_attested(project, args.version, filename)
    notes.append(f"  PyPI attestation: {pypi_ok}  [{pypi_detail}]")

    npm_ok, npm_detail = npm_attested(project, args.version)
    notes.append(f"  npm attestation:   {npm_ok}  [{npm_detail}]")

    npm_int, npm_dist_detail = npm_dist(project, args.version)
    notes.append(f"  npm dist fields:   {npm_dist_detail}")

    pypi_digest = pypi_published_digest(project, args.version, filename)
    notes.append(f"  PyPI published sha256: {pypi_digest}")

    if pypi_ok and "sha256 " in pypi_detail:
        attested = pypi_detail.split("sha256 ", 1)[1]
        if pypi_digest and attested != pypi_digest:
            failures.append(f"attested subject {attested} != PyPI published sha256 {pypi_digest}")
        else:
            notes.append("  attested subject matches PyPI published sha256: yes")

    all_text = "\n".join(p.read_text(encoding="utf-8") for p in DOCS)
    pypi_claim, npm_claim, row = read_attestation_row()
    notes.append(f"\nSECURITY.md attestation row: {row[:110]}...")
    notes.append(f"  parsed claim -> pypi_attested={pypi_claim} npm_attested={npm_claim}")

    if pypi_claim is None and npm_claim is None:
        failures.append(
            "could not parse the 'PEP 740 provenance attestation' row in SECURITY.md; "
            "the guard cannot verify a table it cannot read"
        )
    if pypi_claim is not None and pypi_claim != pypi_ok:
        failures.append(
            f"SECURITY.md claims PyPI attestation={pypi_claim}, live registries say {pypi_ok}"
        )
    if npm_claim is not None and npm_claim != npm_ok:
        failures.append(
            f"SECURITY.md claims npm attestation={npm_claim}, live registries say {npm_ok}"
        )

    rd_npm_claim, rd_line = read_readme_npm_claim()
    notes.append(f"README npm bullet: {rd_line[:110]}")
    notes.append(f"  parsed claim -> npm_attested={rd_npm_claim}")
    if rd_npm_claim is not None and rd_npm_claim != npm_ok:
        failures.append(
            f"README claims npm attestation={rd_npm_claim}, live registries say {npm_ok}"
        )

    # The prose must not describe a digest mismatch as acceptable.
    if re.search(
        r"(unrelated to|need not match|does not need to match) the published digest", all_text, re.I
    ):
        failures.append("prose permits the attested subject to differ from the published digest")

    # Npm integrity pin must be described as a pin, not as provenance.
    # Guard against the positive claim only: "not proof of origin" is correct
    # prose and must not trip this.
    for match in re.finditer(r"dist\.integrity[^\n]{0,140}", all_text, re.I):
        window = match.group(0)
        if re.search(r"(attests|proves who|proof of origin)", window, re.I) and not re.search(
            r"(not|never|no|rather than|only)\s+(a\s+)?(proof|attest|provenance)", window, re.I
        ):
            failures.append(f"prose overstates dist.integrity as provenance: {window[:80]!r}")

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
