from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
REGISTRY_WORKFLOW = ROOT / ".github" / "workflows" / "pypi-registry-smoke.yml"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_release_workflow_keeps_distribution_assets_off_github() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "metadata-only GitHub Release" in release
    assert "Create or verify metadata-only GitHub Release" in release
    assert "gh release create" in release
    assert "gh release upload" not in release
    assert "dist/*" not in release
    assert "never uploaded as Actions artifacts or GitHub" in release


def test_release_workflow_verifies_public_pypi_and_asset_free_github() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "scripts/verify_release_integrity.py" in release
    assert "Verify PyPI publication and GitHub asset-free release policy" in release


def test_release_order_is_publish_then_tag_then_metadata_release_then_policy() -> None:
    release = _text(RELEASE_WORKFLOW)
    publish = release.index("Publish to PyPI (Trusted Publishing)")
    tag = release.index("Create or verify release tag after successful PyPI state")
    github_release = release.index("Create or verify metadata-only GitHub Release")
    cleanup = release.index("Remove legacy wheel and sdist assets from GitHub Releases")
    integrity = release.index("Verify PyPI publication and GitHub asset-free release policy")
    assert publish < tag < github_release < cleanup < integrity


def test_release_recovery_is_idempotent_and_refuses_tag_rewrite() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "Inspect existing public PyPI release for safe recovery" in release
    assert "already_published=true" in release
    assert "steps.pypi_state.outputs.already_published != 'true'" in release
    assert 'git rev-list -n 1 "$RELEASE_TAG"' in release
    assert "Refusing to rewrite a release tag" in release


def test_release_workflow_cleans_legacy_distribution_assets() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert 'endswith(".whl") or endswith(".tar.gz")' in release
    assert 'gh api --method DELETE "repos/$GITHUB_REPOSITORY/releases/assets/$asset_id"' in release
    assert "GitHub-generated" in release


def test_registry_smoke_waits_for_successful_release_and_never_uploads_assets() -> None:
    registry = _text(REGISTRY_WORKFLOW)
    assert "workflow_run:" in registry
    assert "- Release" in registry
    assert "github.event.workflow_run.conclusion == 'success'" in registry
    assert "scripts/verify_pypi_release.py" in registry
    assert "scripts/verify_release_integrity.py" in registry
    assert "gh release upload" not in registry
    assert "--clobber" not in registry
    assert "push:" not in registry
