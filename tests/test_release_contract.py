from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
REGISTRY_WORKFLOW = ROOT / ".github" / "workflows" / "pypi-release-policy.yml"
CHANGELOG_WORKFLOW = ROOT / ".github" / "workflows" / "changelog.yml"
PR_AUTOMATION_WORKFLOW = ROOT / ".github" / "workflows" / "pr-automation.yml"
LEGACY_ATTEST_WORKFLOW = ROOT / ".github" / "workflows" / "attest.yml"
BUILD_LOCK = ROOT / "requirements-build.lock"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _action_refs(path: Path) -> tuple[str, ...]:
    refs: list[str] = []
    for line in _text(path).splitlines():
        stripped = line.strip()
        if not stripped.startswith(("uses:", "- uses:")):
            continue
        value = stripped.split("uses:", 1)[1].strip().split(" #", 1)[0]
        if "@" not in value:
            raise AssertionError(f"action reference has no @ revision: {path}: {value}")
        refs.append(value.rsplit("@", 1)[1])
    return tuple(refs)


def test_dashboard_ships_inside_the_wheel() -> None:
    # Parsed with a regex, not tomllib: the tests target is mypy-checked at
    # python_version = "3.10", where tomllib does not exist.
    pyproject = _text(ROOT / "pyproject.toml")
    match = re.search(r"^force-include\s*=\s*\{(.*)\}\s*$", pyproject, re.MULTILINE)
    assert match is not None, "the wheel force-include table is missing"
    force = dict(re.findall(r'"([^"]+)"\s*=\s*"([^"]+)"', match.group(1)))
    assert force["dashboard/app.py"] == "cds/dashboard/app.py"


def test_release_gate_self_heals_a_stranded_version() -> None:
    release = _text(RELEASE_WORKFLOW)
    bind = release.index("Bind release to the exact CI-tested commit")
    gate = release[bind : release.index("publish:")]
    # A version bump whose first main CI run failed must not stay stranded
    # behind a later fix commit. Whenever the tested commit did not itself
    # change the version source, the gate releases the tree version if PyPI
    # does not serve it yet. The publish job re-checks PyPI before uploading.
    assert "pypi.org/pypi/scientific-computing-system/${version}/json" in gate
    assert "Self-healing" in gate
    assert "should_release=true" in gate
    assert "if: steps.pypi_state.outputs.already_published != 'true'" in release


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
    assert "actions/attest-build-provenance@4d101475d8b20a2381f78447822ac1eab6504dd8" in release
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
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.lstrip().startswith("--hash")
    ]
    assert package_lines
    assert all("==" in line for line in package_lines)
    # At least one digest per requirement, not exactly one. `--require-hashes`
    # compares whichever artifact pip actually downloads, and a requirement with
    # per-platform wheels (charset-normalizer publishes 171 of them) needs every
    # platform's digest. This assertion used to demand equality, which is what
    # produced a lock that installed on Windows and failed on Linux and macOS
    # with "THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE".
    assert lock.count("--hash=sha256:") >= len(package_lines)
    # A digest must not be orphaned: pip ignores a `--hash` line that follows no
    # requirement, so it would verify nothing while appearing to be locked.
    first_pin = next(
        index
        for index, line in enumerate(lock.splitlines())
        if line.strip() and not line.lstrip().startswith(("#", "--hash"))
    )
    assert not any(line.strip().startswith("--hash") for line in lock.splitlines()[:first_pin]), (
        "a --hash line precedes every requirement"
    )


def test_release_workflow_rechecks_version_discipline_before_build() -> None:
    release = _text(RELEASE_WORKFLOW)
    discipline = release.index("Verify synchronized monotonic release metadata")
    build = release.index("Build sdist + wheel without dependency isolation")
    assert discipline < build
    assert "scripts/check_version_discipline.py" in release


def test_current_public_version_metadata_is_synchronized() -> None:
    match = re.search(
        r'^version\s*=\s*["\']([^"\']+)["\']',
        _text(ROOT / "pyproject.toml"),
        re.MULTILINE,
    )
    assert match is not None
    pyproject = match.group(1)
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
    assert (
        "run: python scripts/check_version_discipline.py --base-ref "
        '"${{ steps.base.outputs.ref }}"' in ci
    )
    assert "optional_audit" in ci
    assert "version_discipline" in ci
    assert "needs.optional_audit.result" in ci
    assert "needs.version_discipline.result" in ci


def test_security_sensitive_workflows_pin_actions_to_full_commit_shas() -> None:
    for workflow in (RELEASE_WORKFLOW, REGISTRY_WORKFLOW, CHANGELOG_WORKFLOW):
        refs = _action_refs(workflow)
        assert refs, f"no action refs found in {workflow}"
        for ref in refs:
            assert len(ref) == 40, f"mutable/non-full action ref in {workflow}: {ref}"
            assert all(char in "0123456789abcdef" for char in ref.lower()), (
                f"non-hex action ref in {workflow}: {ref}"
            )


def test_changelog_dispatch_validates_the_tag_before_using_it() -> None:
    changelog = _text(CHANGELOG_WORKFLOW)
    validate = changelog.index("Validate the tag input")
    use = changelog.index("args: --verbose --tag ${{ steps.tag.outputs.tag }}")
    assert validate < use, "the tag input must be validated before git-cliff runs"
    # The raw input is read only through env, then emitted as a validated output.
    assert "REQUESTED_TAG: ${{ inputs.tag }}" in changelog
    assert "tag_re='^v(0|[1-9][0-9]*)" in changelog
    assert '"refs/tags/${tag}"' in changelog
    assert "args: --verbose --tag ${{ inputs.tag }}" not in changelog


def test_pr_automation_validates_dispatch_input_through_env() -> None:
    workflow = _text(PR_AUTOMATION_WORKFLOW)
    assert "parseInt('${{ inputs.pr_number }}')" not in workflow
    assert "REQUESTED_PR_NUMBER: ${{ inputs.pr_number }}" in workflow
    assert "/^[1-9][0-9]{0,9}$/" in workflow
    assert "core.setFailed" in workflow


def test_runtime_lock_gate_does_not_claim_to_audit_an_empty_lock() -> None:
    ci = _text(CI_WORKFLOW)
    # `pip-audit -r requirements.lock` audits nothing while the runtime
    # dependency set is empty; the real audit resolves pyproject.toml.
    assert "pip-audit -r requirements.lock" not in ci
    assert "Assert runtime lock is empty" in ci
    assert "resolved-audit:" in ci


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


def test_release_policy_never_downloads_released_distributions() -> None:
    policy = _text(REGISTRY_WORKFLOW)
    assert "name: PyPI Release Policy" in policy
    assert "pip install" not in policy
    assert "pypi.org/simple" not in policy
    assert "verify_pypi_release.py" not in policy
    assert "registry-cli:" not in policy
    assert "workflow_dispatch:" in policy
    assert "verify_release_integrity.py" in policy


def test_release_policy_removes_forbidden_distribution_assets() -> None:
    registry = _text(REGISTRY_WORKFLOW)
    assert "Remove forbidden wheel and sdist assets from matching GitHub Release" in registry
    assert "gh release upload" not in registry
    assert "gh api --method DELETE" in registry
    assert "releases/assets/$asset_id" in registry
    assert "scripts/verify_release_integrity.py" in registry
    assert "Verify public PyPI and asset-free GitHub Release policy" in registry


def test_release_policy_does_not_copy_pypi_distributions_to_github() -> None:
    registry = _text(REGISTRY_WORKFLOW)
    assert "Download exact wheel and sdist from public PyPI" not in registry
    assert "Synchronize exact PyPI files to matching GitHub Release" not in registry
    assert "dist/* --clobber" not in registry


def test_release_workflow_changes_trigger_release_policy_check() -> None:
    registry = _text(REGISTRY_WORKFLOW)
    assert '".github/workflows/release.yml"' in registry
    assert '"scripts/verify_release_integrity.py"' in registry
