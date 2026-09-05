from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
REGISTRY_WORKFLOW = ROOT / ".github" / "workflows" / "pypi-registry-smoke.yml"
ATTEST_WORKFLOW = ROOT / ".github" / "workflows" / "attest.yml"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_release_workflow_keeps_distribution_assets_off_github() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "Create metadata-only GitHub Release and remove distribution assets" in release
    assert "gh release create" in release
    assert "gh release upload" not in release
    assert "gh release delete-asset" in release
    assert "Distribution files are never" in release


def test_release_workflow_attests_ephemeral_build_before_publish() -> None:
    release = _text(RELEASE_WORKFLOW)
    attest = release.index("Attest verified ephemeral wheel and sdist")
    publish = release.index("Publish to PyPI (Trusted Publishing)")
    assert "actions/attest-build-provenance@v2" in release
    assert 'subject-path: "dist/*"' in release
    assert attest < publish


def test_release_workflow_verifies_pypi_and_asset_free_policy() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "scripts/verify_release_integrity.py" in release
    assert "Verify PyPI publication and metadata-only GitHub Release policy" in release


def test_release_order_is_publish_then_tag_then_metadata_release_then_integrity() -> None:
    release = _text(RELEASE_WORKFLOW)
    publish = release.index("Publish to PyPI (Trusted Publishing)")
    tag = release.index("Create or verify release tag after successful PyPI state")
    github_release = release.index(
        "Create metadata-only GitHub Release and remove distribution assets"
    )
    integrity = release.index("Verify PyPI publication and metadata-only GitHub Release policy")
    assert publish < tag < github_release < integrity


def test_release_recovery_is_idempotent_and_refuses_tag_rewrite() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "Inspect existing public PyPI release for safe recovery" in release
    assert "already_published=true" in release
    assert "steps.pypi_state.outputs.already_published != 'true'" in release
    assert 'git rev-list -n 1 "$RELEASE_TAG"' in release
    assert "Refusing to rewrite a release tag" in release


def test_registry_smoke_is_read_only_and_rechecks_asset_policy() -> None:
    registry = _text(REGISTRY_WORKFLOW)
    assert "gh release upload" not in registry
    assert "contents: write" not in registry
    assert "scripts/verify_release_integrity.py" in registry
    assert "Verify PyPI publication and metadata-only GitHub Release policy" in registry


def test_legacy_release_asset_attestation_workflow_is_removed() -> None:
    assert not ATTEST_WORKFLOW.exists()


def test_release_workflow_changes_trigger_registry_contract_check() -> None:
    registry = _text(REGISTRY_WORKFLOW)
    assert '".github/workflows/release.yml"' in registry
    assert '"scripts/verify_release_integrity.py"' in registry
