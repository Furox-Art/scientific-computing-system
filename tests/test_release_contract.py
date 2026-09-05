from __future__ import annotations

import re
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
REGISTRY_WORKFLOW = ROOT / ".github" / "workflows" / "pypi-registry-smoke.yml"
LEGACY_ATTEST_WORKFLOW = ROOT / ".github" / "workflows" / "attest.yml"
BUILD_LOCK = ROOT / "requirements-build.lock"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_release_workflow_keeps_distribution_assets_off_github() -> None:
    release = _text(RELEASE_WORKFLOW)
    assert "Create metadata-only GitHub Release" in release
    assert "gh release create" in release
    assert "gh release upload" not in release
    assert "published distribution assets" not in release
    assert '--title "$RELEASE_TAG" dist/*' not in release


def test_release_workflow_attests_verified_build_before_publish() -> None:
    release = _text(RELEASE_WORKFLOW)
    smoke = release.index("Smoke-test installed CLI before publish")
    attest = release.index("Attest verified runner-local build provenance")
    publish = release.index("Publish to PyPI (Trusted Publishing)")
    assert smoke < attest < publish
    assert "actions/attest-build-provenance@" in release
    assert 'subject-path: "dist/*"' in release
    assert "attestations: write" in release
    assert not LEGACY_ATTEST_WORKFLOW.exists()


def test_every_release_action_is_pinned_to_an_immutable_commit_sha() -> None:
    release = _text(RELEASE_WORKFLOW)
    actions = re.findall(r"^\s*-?\s*uses:\s*([^\s#]+)", release, re.MULTILINE)
    assert actions
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", action) for action in actions)
    assert "@v5" not in release
    assert "@v7" not in release
    assert "@v8" not in release
    assert "@release/v1" not in release


def test_release_build_toolchain_is_hash_locked_and_non_isolated() -> None:
    release = _text(RELEASE_WORKFLOW)
    lock = _text(BUILD_LOCK)
    assert "--require-hashes" in release
    assert "--only-binary=:all:" in release
    assert "-r requirements-build.lock" in release
    assert "python -m build --no-isolation" in release
    assert "pip install build" not in release
    assert "pip install --upgrade pip" not in release
    package_lines = [
        line.strip()
        for line in lock.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not line.lstrip().startswith("--hash")
    ]
    assert package_lines
    assert all("==" in line for line in package_lines)
    assert lock.count("--hash=sha256:") == len(package_lines)


def test_release_workflow_rechecks_version_discipline_before_build() -> None:
    release = _text(RELEASE_WORKFLOW)
    discipline = release.index("Verify synchronized monotonic release metadata")
    build = release.index("Build sdist + wheel without dependency isolation")
    assert discipline < build
    assert "scripts/check_version_discipline.py" in release


def test_current_public_version_metadata_is_synchronized() -> None:
    pyproject = tomllib.loads(_text(ROOT / "pyproject.toml"))["project"]["version"]
    version_source = _text(ROOT / "src" / "cds" / "_version.py")
    match = re.search(r'__version__\s*=\s*version\s*=\s*"([^"]+)"', version_source)
    assert match is not None
    citation_versions = re.findall(
        r"^\s*version:\s*[\"']?([^\s\"']+)[\"']?\s*$",
        _text(ROOT / "CITATION.cff"),
        re.MULTILINE,
    )
    assert citation_versions == [pyproject, pyproject]
    assert match.group(1) == pyproject


def test_ci_gates_optional_dependency_audit_and_version_discipline() -> None:
    ci = _text(CI_WORKFLOW)
    assert "optional_audit:" in ci
    assert "version_discipline:" in ci
    assert "pip freeze --exclude scientific-computing-system" in ci
    assert "pip-audit -r /tmp/scientific-io-resolved.txt --strict" in ci
    assert 'pip install pip-audit ".[test,scientific,io]"' in ci
    assert "scripts/check_version_discipline.py --base-ref HEAD^1" in ci
    assert "- optional_audit" in ci
    assert "- version_discipline" in ci
    assert "needs.optional_audit.result" in ci
    assert "needs.version_discipline.result" in ci


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
