# Maintenance & Distribution Guide

How the project is maintained, tested, and released. Every publish step runs in
GitHub Actions; nothing is uploaded from a maintainer machine.

## Local Development Setup

To install all dependencies required for development, testing, and documentation:

```bash
pip install -e ".[dev,test,docs]"
```

## Running Quality Checks

Before submitting a PR, ensure all checks pass:

```bash
# Linting and formatting
ruff check .
ruff format --check .

# Type checking
mypy src/
mypy tests/

# Testing
pytest tests/
```

These are the same gates CI runs. CI additionally enforces branch coverage and
the packaging/release-contract suites.

## Version Surfaces

The version is written in six places, all of which must agree. Drift fails
`scripts/check_version_lockstep.py` in CI and again in the release workflow.

- `pyproject.toml` — `[project] version` (the authoritative value)
- `src/cds/_version.py` — `__version__` (re-exported as `cds.__version__`)
- `package.json` — `version` of the npm launcher
- `codemeta.json` — `version` of the CodeMeta record
- `CITATION.cff` — top-level and `preferred-citation` version fields
- `CHANGELOG.md` — a `## [vX.Y.Z]` heading

`scripts/check_version_discipline.py` additionally requires the new version to be
greater than the previous release tag.

## Release (automatic)

1. Open a PR that bumps every version surface above, and merge it to `main`.
2. The `CI` workflow (`.github/workflows/tests.yml`) runs the full matrix. Main
   runs are never cancelled, because the release workflow consumes their result.
3. When `CI` completes successfully for a push to `main` whose commit changed
   `src/cds/_version.py`, the `Release` workflow
   (`.github/workflows/release.yml`) starts via `workflow_run`, binds itself to
   the exact CI-tested commit, and:
   - resolves the previous release tag and re-runs the version-discipline and
     lockstep gates against it;
   - builds the sdist and wheel with the hash-locked toolchain
     (`requirements-build.lock`, `python -m build --no-isolation`);
   - validates metadata (`twine check --strict`), artifact contents
     (`scripts/check_distribution_contents.py`), and the installed CLI
     (`scripts/verify_cli_install.py`);
   - attests build provenance with `actions/attest-build-provenance`;
   - publishes to PyPI through **Trusted Publishing** (OIDC, `pypi`
     environment) — no API token secret is used or needed;
   - creates the annotated `vX.Y.Z` tag only after PyPI accepted the upload;
   - creates a **metadata-only** GitHub Release (no distribution assets; the
     artifacts live on PyPI);
   - verifies the result with `scripts/verify_release_integrity.py`.

A push to `main` that does not touch `src/cds/_version.py` does not release —
unless the version in `pyproject.toml` is not yet on PyPI. That self-healing
path releases the tree version from the CI-tested commit, so a version bump
whose first main CI run failed cannot stay stranded behind a later fix. The
publish job re-checks PyPI before uploading, so it cannot duplicate a release.

### One-time PyPI setup

The `pypi` environment is protected and registered as a Trusted Publisher on
PyPI: owner `Furox-Art`, repository `scientific-computing-system`, workflow
`release.yml`, environment `pypi`. There is no `PYPI_API_TOKEN` secret.

### Recovery

- If PyPI accepted the upload but the tag or GitHub Release step failed, dispatch
  `Release` manually with `version` (optionally `release_sha`). That recovery
  path never builds or uploads, and refuses versions that are not already on
  PyPI.
- Re-running a failed `Release` run is safe: when the exact wheel and sdist
  already exist on PyPI, the workflow switches to recovery mode and skips the
  upload.
- Do not push release tags by hand. Release tags are immutable and created by the
  workflow after a successful publish.

## npm Launcher

`.github/workflows/npm-publish.yml` is the sole npm authority. It runs when the
`Release` workflow completes and proceeds only when the PyPI publish job
succeeded. A manual dispatch is accepted only from `main`, a `v*` tag push only
when the tag matches the `package.json` version, and every path requires that
exact version to already exist on PyPI. Publishing uses OIDC trusted publishing
with a provenance attestation; dispatch with `use_token_fallback=true` only while
the npm trusted publisher is unavailable, which publishes without provenance.
One-time setup: npmjs.com → `scientific-computing-system` → Settings → Trusted
Publisher (owner `Furox-Art`, workflow `npm-publish.yml`, environment `npm`, with
`npm publish` allowed).

## Documentation Deployment

Documentation deploys automatically. The `deploy-docs` job in
`.github/workflows/tests.yml` runs after the aggregate CI gate succeeds on a
push to `main`, builds with `mkdocs build --strict`, and then runs
`mkdocs gh-deploy`. No maintainer machine deploys the docs.

## Local Release Preflight

`scripts/publish.py` is a local preflight only: it verifies a clean tree on
`main`, runs the tests, builds, and confirms the built wheel matches the version
declared in `pyproject.toml`. It does not push tags, deploy docs, or upload
anything. Merge to `main` and let CI perform the release.
