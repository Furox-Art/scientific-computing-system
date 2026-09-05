from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
REGISTRY_WORKFLOW = ROOT / ".github" / "workflows" / "pypi-registry-smoke.yml"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_release_workflow_attaches_distribution_assets() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "Create GitHub Release with published distribution assets" in release
    assert "gh release create" in release
    assert "gh release upload" in release
    assert "dist/*" in release
    assert "asset-free" not in release.lower()


def test_release_workflow_verifies_cross_registry_integrity() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "scripts/verify_release_integrity.py" in release
    assert "Verify PyPI and GitHub Release artifact parity" in release


def test_release_order_is_publish_then_tag_then_release_then_integrity() -> None:
    release = _text(RELEASE_WORKFLOW)
    publish = release.index("Publish to PyPI (Trusted Publishing)")
    tag = release.index("Create release tag after successful PyPI publish")
    github_release = release.index("Create GitHub Release with published distribution assets")
    integrity = release.index("Verify PyPI and GitHub Release artifact parity")
    assert publish < tag < github_release < integrity


def test_registry_smoke_repairs_and_rechecks_release_assets() -> None:
    registry = _text(REGISTRY_WORKFLOW)
    assert "gh release upload" in registry
    assert "--clobber" in registry
    assert "scripts/verify_release_integrity.py" in registry
    assert "Verify synchronized PyPI and GitHub Release parity" in registry


def test_release_workflow_changes_trigger_registry_contract_check() -> None:
    registry = _text(REGISTRY_WORKFLOW)
    assert '".github/workflows/release.yml"' in registry
    assert '"scripts/verify_release_integrity.py"' in registry
