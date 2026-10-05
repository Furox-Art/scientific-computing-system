"""The provenance-claim guard must reject prose the registries contradict.

`docs/checks/verify_provenance_claims.py` exists because the documentation once
asserted, from memory, which registry carried a build attestation, and nothing
checked it. A runnable script is not a gate: it only helps whoever remembers to
run it. This module makes the script's decision logic a committed, deterministic
test, and wires the live run into CI (the ``version_discipline`` job).

Every registry answer is **injected**. Nothing here opens a socket, so the suite
cannot flake, cannot be rate-limited by PyPI or npm, and cannot pass or fail
because a registry was briefly unreachable. The three rejection behaviours are
the ones that were negative-controlled by hand before the script existed, plus
the tri-state rule that keeps a network outage from masquerading as "unattested".

Coverage note: ``[tool.coverage.run] source = ["src/cds"]``, so exercising this
module cannot reduce the 100% blended-coverage gate on the library itself.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "docs" / "checks" / "verify_provenance_claims.py"

PYPI_WHEEL = "scientific_computing_system-2.2.1-py3-none-any.whl"
PYPI_SDIST = "scientific_computing_system-2.2.1.tar.gz"
WHEEL_SHA256 = "b14b89478fdc8b11482ffd556bb6eb59c6b21887f2a09418edec7029d4569bb9"
SDIST_SHA256 = "6ba4ad4c2ebb6bef86356bcbcd251e12978737945e58d9e75b9c68860efa77f6"

# A digest that is deliberately not any published digest, for the mismatch case.
BOGUS_SHA256 = "0" * 64


def _load_guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_provenance_claims", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _evaluate(
    *,
    pypi_claim: bool | None = True,
    npm_claim: bool | None = False,
    readme_npm_claim: bool | None = False,
    pypi_state: str = guard.YES,
    npm_state: str = guard.NO,
    attested_digest: str | None = WHEEL_SHA256,
    published_digest: str | None = WHEEL_SHA256,
    all_text: str = "",
) -> list[str]:
    result: list[str] = guard.evaluate_claims(
        pypi_claim=pypi_claim,
        npm_claim=npm_claim,
        readme_npm_claim=readme_npm_claim,
        pypi_state=pypi_state,
        npm_state=npm_state,
        attested_digest=attested_digest,
        published_digest=published_digest,
        all_text=all_text,
    )
    return list(result)


# ---------------------------------------------------------------------------
# The three negative controls
# ---------------------------------------------------------------------------


def test_false_pypi_not_attested_claim_is_rejected() -> None:
    """ "PyPI is not attested" must fail while PyPI is in fact attested.

    This is the exact false claim that was reported against the real repository,
    where it originated from requesting the `/integrity/<project>/<version>/`
    *directory* instead of the per-file endpoint.
    """
    failures = _evaluate(pypi_claim=False, pypi_state=guard.YES)

    assert failures, "a false 'PyPI has no attestation' claim was accepted"
    assert any("claims PyPI attestation=False" in f for f in failures), failures


def test_false_npm_attested_claim_is_rejected() -> None:
    """ "npm is attested" must fail while npm serves no attestation."""
    failures = _evaluate(npm_claim=True, npm_state=guard.NO)

    assert failures, "a false 'npm is attested' claim was accepted"
    assert any("SECURITY.md claims npm attestation=True" in f for f in failures), failures


def test_attested_subject_digest_mismatch_is_rejected() -> None:
    """An attested subject that differs from the published digest must fail.

    A digest is only meaningful if it is the digest of the bytes actually
    published; accepting a mismatch would let an attestation "cover" a different
    file than the one a consumer downloads.
    """
    failures = _evaluate(attested_digest=BOGUS_SHA256, published_digest=WHEEL_SHA256)

    assert failures, "an attested subject that disagrees with the published digest was accepted"
    assert any("attested subject" in f and "published sha256" in f for f in failures), failures


def test_readme_false_npm_attested_claim_is_rejected() -> None:
    """The README carries its own npm claim; it is checked separately."""
    failures = _evaluate(readme_npm_claim=True, npm_state=guard.NO)

    assert failures, "a false README npm claim was accepted"
    assert any("README claims npm attestation=True" in f for f in failures), failures


# ---------------------------------------------------------------------------
# The documented truth must pass
# ---------------------------------------------------------------------------


def test_measured_truth_passes() -> None:
    """PyPI attested (2.2.1) and npm not attested is the state as measured."""
    assert _evaluate() == []


def test_unparsable_table_row_is_a_failure() -> None:
    """A guard that cannot read the table must fail, not silently pass."""
    failures = _evaluate(pypi_claim=None, npm_claim=None)

    assert any("could not parse" in f for f in failures), failures


def test_unreachable_registry_does_not_manufacture_a_failure() -> None:
    """An outage is UNKNOWN, not NO: it must neither fail nor be read as "unattested".

    A required gate that fails on a DNS blip trains people to re-run it rather
    than read it.
    """
    failures = _evaluate(pypi_state=guard.UNKNOWN, npm_state=guard.UNKNOWN)

    assert failures == [], failures


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("The subject need not match the published digest", True),
        ("dist.integrity attests who built the tarball", True),
        ("dist.integrity is a content pin, not proof of origin", False),
        ("a content digest pins which bytes you get", False),
    ],
)
def test_prose_safety_rules(phrase: str, expected: bool) -> None:
    """Overstating a digest, or excusing a mismatch, must be rejected."""
    failures = _evaluate(all_text=phrase)

    assert bool(failures) is expected, (phrase, failures)


# ---------------------------------------------------------------------------
# Registry helpers: exercised with injected HTTP, never a real socket
# ---------------------------------------------------------------------------


def test_pypi_attested_reads_a_real_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 200 body is decoded down to the attested subject digest."""
    import base64
    import json

    statement = json.dumps(
        {"subject": [{"name": PYPI_WHEEL, "digest": {"sha256": SDIST_SHA256}}]}
    ).encode()
    bundle = {
        "attestation_bundles": [
            {"attestations": [{"envelope": {"statement": base64.b64encode(statement).decode()}}]}
        ]
    }
    monkeypatch.setattr(guard, "_get", lambda url, accept=None: (200, json.dumps(bundle).encode()))

    state, detail = guard.pypi_attested("scientific-computing-system", "2.2.1", PYPI_WHEEL)

    assert state == guard.YES
    assert guard.attested_sha256(detail) == SDIST_SHA256


def test_pypi_attested_maps_404_to_no(monkeypatch: pytest.MonkeyPatch) -> None:
    """The directory-vs-file trap: a 404 on the per-file URL means no attestation."""
    monkeypatch.setattr(guard, "_get", lambda url, accept=None: (404, b""))

    state, detail = guard.pypi_attested("scientific-computing-system", "2.2.1", PYPI_WHEEL)

    assert state == guard.NO
    assert guard.attested_sha256(detail) is None


def test_pypi_attested_maps_outage_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport failure is UNKNOWN, never NO."""
    monkeypatch.setattr(guard, "_get", lambda url, accept=None: (None, b"dns failure"))

    state, _ = guard.pypi_attested("scientific-computing-system", "2.2.1", PYPI_WHEEL)

    assert state == guard.UNKNOWN


def test_pypi_attested_maps_bad_bundle_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 200 that is not a usable bundle is UNKNOWN, not a silent pass."""
    monkeypatch.setattr(guard, "_get", lambda url, accept=None: (200, b"not json"))

    state, detail = guard.pypi_attested("scientific-computing-system", "2.2.1", PYPI_WHEEL)

    assert state == guard.UNKNOWN
    assert "unparsable bundle" in detail


def test_published_digest_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The published sha256 is read from the per-file list."""
    import json

    body = json.dumps(
        {"urls": [{"filename": PYPI_WHEEL, "digests": {"sha256": WHEEL_SHA256}}]}
    ).encode()
    monkeypatch.setattr(guard, "_get", lambda url, accept=None: (200, body))

    assert guard.pypi_published_digest("scientific-computing-system", "2.2.1", PYPI_WHEEL) == (
        WHEEL_SHA256
    )
    assert guard.pypi_published_digest("scientific-computing-system", "2.2.1", "absent.whl") is None


def test_published_digest_lookup_handles_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "_get", lambda url, accept=None: (500, b""))
    assert guard.pypi_published_digest("scientific-computing-system", "2.2.1", PYPI_WHEEL) is None


@pytest.mark.parametrize(
    ("status", "expected"),
    [(200, guard.YES), (404, guard.NO), (500, guard.UNKNOWN), (None, guard.UNKNOWN)],
)
def test_npm_attested_status_mapping(
    monkeypatch: pytest.MonkeyPatch, status: int | None, expected: str
) -> None:
    """npm's three distinguishable answers, plus an outage."""
    monkeypatch.setattr(guard, "_get", lambda url, accept=None: (status, b""))

    state, _ = guard.npm_attested("scientific-computing-system", "2.2.1")

    assert state == expected


def test_npm_dist_reports_integrity_and_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    """dist.integrity and dist.signatures are reported, and distinguished."""
    import json

    body = json.dumps(
        {
            "dist": {
                "integrity": "sha512-abc",
                "shasum": "deadbeef",
                "signatures": [{"sig": "x"}],
            }
        }
    ).encode()
    monkeypatch.setattr(guard, "_get", lambda url, accept=None: (200, body))

    has_integrity, detail = guard.npm_dist("scientific-computing-system", "2.2.1")

    assert has_integrity is True
    assert "dist.integrity=True" in detail
    assert "dist.signatures=True" in detail


def test_npm_dist_handles_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "_get", lambda url, accept=None: (None, b""))
    has_integrity, detail = guard.npm_dist("scientific-computing-system", "2.2.1")
    assert has_integrity is False
    assert detail.startswith("HTTP ")


# ---------------------------------------------------------------------------
# Prose parsing against the files actually in the tree
# ---------------------------------------------------------------------------


def test_security_table_row_parses_to_the_measured_verdict() -> None:
    """SECURITY.md's real row must parse to (True, True) today."""
    pypi_claim, npm_claim, row = guard.read_attestation_row()

    assert row, "the attestation row disappeared from SECURITY.md"
    assert pypi_claim is True
    assert npm_claim is True


def test_readme_npm_bullet_parses_to_an_attestation() -> None:
    readme_claim, line = guard.read_readme_npm_claim()

    assert line, "the README npm bullet disappeared"
    assert readme_claim is True


@pytest.mark.parametrize(
    ("cell", "label", "expected"),
    [
        ("**yes on 2.2.1** (verified)", "**PEP 740 provenance attestation**", True),
        (
            "**none** — attestations endpoint returns 404",
            "**PEP 740 provenance attestation**",
            False,
        ),
        ("**not on 2.2.0**", "**PEP 740 provenance attestation**", False),
        ("nothing checkable here", "**Some other row**", None),
        ("**not *yet*** registered", "**attestation**", False),
    ],
)
def test_cell_verdict(cell: str, label: str, expected: bool | None) -> None:
    """Cell parsing reads the cell, with the row label supplying the subject."""
    assert guard.cell_verdict(cell, label) is expected


def test_prose_claims_scan_finds_the_expected_signals() -> None:
    """The committed SECURITY.md must show every signal the guard relies on."""
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    claims = guard.prose_claims(security)

    assert claims["mentions_pypi_attestation"]
    assert claims["mentions_npm_attestation_status"]
    assert claims["distinguishes_npm_transport_sig"]
    assert claims["distinguishes_digest_from_provenance"]
    # Both trusted publishers are registered and 2.2.3 is attested on npm, so the
    # prose must no longer claim a pending registration.
    assert not claims["states_registration_pending"]


def test_scan_notes_find_claims_in_the_tree() -> None:
    """The scanner must actually find claims, or the guard checks nothing."""
    notes, any_claim = guard.scan_notes()

    assert any_claim
    assert any("SECURITY.md" in n for n in notes)


def test_offline_main_passes_on_the_committed_tree() -> None:
    """`--offline` is the deterministic mode CI-style callers can rely on."""
    assert guard.main(["--offline"]) == 0


def test_main_fails_when_no_claim_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With every doc emptied of claims, the guard must fail rather than pass."""
    monkeypatch.setattr(guard, "DOCS", [tmp_path / "empty.md"])
    (tmp_path / "empty.md").write_text("# nothing to verify here\n", encoding="utf-8")

    assert guard.main(["--offline"]) == 1


def test_strict_network_escalates_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--strict-network` turns an unreadable registry into a failure."""
    monkeypatch.setattr(guard, "pypi_attested", lambda *a: (guard.UNKNOWN, "unreachable"))
    # npm answers normally: the outage under test is PyPI's, and a contradictory
    # npm answer would fail for the wrong reason (the docs claim npm is attested).
    monkeypatch.setattr(guard, "npm_attested", lambda *a: (guard.YES, "200"))
    monkeypatch.setattr(guard, "npm_dist", lambda *a: (True, "dist.integrity=True"))
    monkeypatch.setattr(guard, "pypi_published_digest", lambda *a: None)

    assert guard.main(["--version", "2.2.3", "--strict-network"]) == 1
    assert guard.main(["--version", "2.2.3"]) == 0


def test_decoding_rejects_a_bundle_without_a_subject() -> None:
    """A bundle whose statement has no subject digest raises, not guesses."""
    import base64
    import json

    statement = json.dumps({"predicateType": "x"}).encode()
    bundle = {
        "attestation_bundles": [
            {"attestations": [{"envelope": {"statement": base64.b64encode(statement).decode()}}]}
        ]
    }
    with pytest.raises(Exception):
        guard.decode_attestation_sha256(json.dumps(bundle).encode())
