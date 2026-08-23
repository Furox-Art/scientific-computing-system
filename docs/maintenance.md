# Maintenance & Distribution Guide

This guide explains how to maintain the Scientific Computing System and publish new versions to PyPI.

## Local Development Setup

To install all dependencies required for development, testing, and documentation:

```bash
pip install -e ".[dev,test,docs]"
```

## Running Quality Checks

Before submitting a PR, ensure all checks pass:

```bash
# Linting
ruff check .

# Type checking
mypy src/

# Testing
pytest tests/
```

## Publishing to PyPI

CDS publishes to PyPI automatically via **GitHub Actions** (`.github/workflows/release.yml`) using a scoped **PyPI API token** stored as the `PYPI_API_TOKEN` repository secret. The release pipeline is the **sole** publish authority — `scripts/publish.py` only builds, verifies, runs tests, pushes the tag (which triggers CI), and deploys docs; it never uploads to PyPI directly.

### 0. One-time setup
Create a PyPI API token scoped to `scientific-computing-system` (Account → API tokens) and add it as a repo secret:
```bash
gh secret set PYPI_API_TOKEN < ~/.pypi-token
```

### 1. Update Version
Update the `version` field in `pyproject.toml` and mirror it in `src/cds/_version.py` (kept in lockstep).

### 2. Commit and Tag
Commit your changes and push a new version tag:
```bash
git add .
git commit -m "chore: bump version to vX.Y.Z"
git push origin main

git tag vX.Y.Z
git push origin vX.Y.Z
```

### 3. Automated Release
The `release.yml` workflow triggers automatically on the tag push. It will:
- Build the source and wheel distributions.
- Upload them to PyPI using the `PYPI_API_TOKEN` secret.
- Create a GitHub Release with the artifacts attached.
The `attest.yml` workflow then fires on `release: published` to sign the artifacts (sigstore build provenance).

## Documentation Deployment

The MkDocs site is deployed to GitHub Pages via `scripts/publish.py`, which runs `mkdocs gh-deploy --force --no-history` locally after the release tag is pushed (use `--skip-docs` to opt out). The `CI` workflow (`tests.yml`) only runs `mkdocs build --strict` as a link/reference lint — it does **not** deploy. This keeps the canonical docs build on the local toolchain, matching `publish.py`'s role as the local half of the two-stage release.
