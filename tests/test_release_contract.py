from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
REGISTRY_WORKFLOW = ROOT / ".github" / "workflows" / "pypi-registry-smoke.yml"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_release_workflow_keeps_distribution_assets_off_github() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "Create metadata-only GitHub Release" in release
    assert "gh release create" in release
    assert "gh release upload" not in release
    assert "published distribution assets" not in release
    assert '--title "$RELEASE_TAG" dist/*' not in release


def test_release_workflow_verifies_asset_free_policy() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "scripts/verify_release_integrity.py" in release
    assert "Verify PyPI release and asset-free GitHub Release policy" in release


def test_release_order_is_publish_then_tag_then_metadata_release_then_integrity() -> None:
    release = _text(RELEASE_WORKFLOW)
    publish = release.index("Publish to PyPI (Trusted Publishing)")
    tag = release.index("Create or verify release tag after successful PyPI state")
    github_release = release.index("Create metadata-only GitHub Release")
    integrity = release.index("Verify PyPI release and asset-free GitHub Release policy")
    assert publish < tag < github_release < integrity


def test_release_recovery_is_idempotent_and_refuses_tag_rewrite() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "Inspect existing public PyPI release for safe recovery" in release
    assert "already_published=true" in release
    assert "steps.pypi_state.outputs.already_published != 'true'" in release
    assert 'git rev-list -n 1 "$RELEASE_TAG"' in release
    assert "Refusing to rewrite a release tag" in release


def test_registry_smoke_removes_forbidden_distribution_assets() -> None:
    registry = _text(REGISTRY_WORKFLOW)
    assert "Remove forbidden wheel and sdist assets from matching GitHub Release" in registry
    assert "gh release upload" not in registry
    assert "gh api --method DELETE" in registry
    assert "releases/assets/$asset_id" in registry
    assert "scripts/verify_release_integrity.py" in registry
    assert "Verify public PyPI and asset-free GitHub Release policy" in registry


def test_registry_smoke_does_not_copy_pypi_distributions_to_github() -> None:
    registry = _text(REGISTRY_WORKFLOW)
    assert "Download exact wheel and sdist from public PyPI" not in registry
    assert "Synchronize exact PyPI files to matching GitHub Release" not in registry
    assert "dist/* --clobber" not in registry


def test_release_workflow_changes_trigger_registry_contract_check() -> None:
    registry = _text(REGISTRY_WORKFLOW)
    assert '".github/workflows/release.yml"' in registry
    assert '"scripts/verify_release_integrity.py"' in registry