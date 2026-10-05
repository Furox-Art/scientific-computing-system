# Security Policy

## Supported Versions

| Version | Supported | Notes |
|---|---|---|
| 2.2.x | Yes | Current stable release line |
| < 2.2 | No | Superseded by the 2.2 release line unless a specific security advisory states otherwise |

Security fixes target the current stable release line. Older release lines should not be assumed to receive backports unless an advisory explicitly says so.

## Reporting a Vulnerability

If you find a security vulnerability, please **do not open a public issue**.

Use GitHub's private vulnerability reporting feature:

- GitHub: https://github.com/Furox-Art/scientific-computing-system/security/advisories/new

You may also contact the maintainer directly. Reports should include the affected version, a minimal reproduction where practical, impact, and any suggested mitigation.

## Threat Model

`scientific-computing-system` is a local-first, pure-Python scientific-computing library distributed through PyPI. The core package has no required runtime dependencies. Optional scientific backends are loaded only when explicitly requested.

It is also published to npm as `scientific-computing-system`, but that package is a **Node launcher shim only** — it contains no Python, resolves an interpreter (`python`/`python3`/`py`), and runs `python -m cds`. npm users must install the Python distribution so that `python -m cds` resolves; the launcher never looks for a `cds` console script on `PATH`. The two registries carry **different provenance guarantees**; see [Distribution channels and provenance](#distribution-channels-and-provenance).

### In Scope

| Threat | Mitigation |
|---|---|
| **Supply chain: malicious or substituted PyPI artifact** | The release workflow is the sole PyPI publish authority. It builds wheel + sdist on a GitHub-hosted runner from a hash-pinned toolchain, verifies package metadata/version, installs the built wheel, smoke-tests the installed CLI, and publishes through PyPI Trusted Publishing (OIDC). PyPI serves a PEP 740 provenance bundle for 2.2.1 whose subject digest matches the published `sha256`. |
| **Supply chain: malicious or substituted npm artifact** | The npm workflow is the sole npm publish authority and gates on the registry version-existence check, a `npm pack --dry-run` tarball allowlist, and npm/Python version lockstep. **No attestation is possible yet:** the npmjs.com trusted publisher is not registered, so the token fallback is the working mode and npm's `--provenance` cannot be honoured. Consumers can only pin `dist.integrity` until that is registered. See below. |
| **Registry drift** | Public PyPI is treated as the distribution registry for the library. The release integrity check requires exactly one wheel and one sdist on PyPI and requires the matching GitHub Release to contain no wheel/sdist assets. A registry-policy workflow rechecks the public PyPI package and removes accidental distribution assets from GitHub Releases. |
| **Dependency vulnerabilities** | The core has no required runtime dependencies. Development/test/docs lock files are audited in CI with `pip-audit`; optional backends are isolated behind extras and lazy loading. |
| **Code execution from package install** | The build backend is `hatchling`; there is no `setup.py` execution and package versioning is static in `pyproject.toml` plus `src/cds/_version.py`. |
| **Unexpected scientific-tool loading** | Optional tools are selected through an explicit registry/capability layer and are not imported into the zero-dependency core unless requested. |
| **Untrusted CLI input** | The CLI uses `argparse`-based typed/explicit parsing and does not evaluate arbitrary Python expressions. |
| **Scientific workflow overclaiming** | The research orchestrator is fail-closed: blocked methods, missing tools, denied approvals, incomplete execution, validation failures, and unresolved method suitability prevent an unqualified final conclusion. |

### Distribution channels and provenance

The two channels carry **different** artifacts and **different** verification
properties. This section states exactly what is verifiable today. It is
version-specific; re-check it rather than trusting it indefinitely.

| | PyPI `scientific-computing-system` | npm `scientific-computing-system` |
|---|---|---|
| Contents | the library (wheel + sdist) | `index.js`, `bin/scs.js`, `LICENSE`, `README.md`, `CHANGELOG.md`, `SECURITY.md` — **no Python** |
| Requires the other channel | no | yes; `scs` runs `python -m cds`, so the Python distribution must be installed. It resolves `python`/`python3`/`py` and does **not** look for a `cds` console script on `PATH` |
| Publish authentication | Trusted Publishing (OIDC) | OIDC trusted publishing implemented and default; token fallback exists |
| **PEP 740 provenance attestation** | **yes on 2.2.1** (verified; see below) | **none** — attestations endpoint returns 404 |
| **npm `dist.signatures`** | n/a | present — registry transport signature, **not** build provenance |
| **Content digest a consumer can pin** | `sha256` per file on the PyPI file page | `dist.integrity` (sha512) and `dist.shasum` (sha1) |

#### Three things that are easy to confuse

1. **A PEP 740 provenance attestation** is a Sigstore-signed statement binding a
   file digest to the identity and workflow that published it. PyPI serves it
   from `/integrity/<project>/<version>/<filename>/provenance`. It is the only
   one of the three that attests *who built this*.
2. **npm's `dist.signatures`** is npm signing its own registry metadata for
   transport integrity. It proves the metadata came from npm's key. It says
   nothing about who built the tarball, and it is not a provenance attestation.
3. **A content digest** (`sha256` on PyPI, `dist.integrity`/`dist.shasum` on npm)
   pins *which bytes* you get. Useful for integrity and for detecting a
   substituted file; on its own it proves nothing about origin, because anyone
   who substitutes a file also substitutes its digest.

#### What is published today, and how to check it

Verified against the live registries for **PyPI 2.2.1**:

```bash
# PEP 740 attestation exists (HTTP 200). Note the per-FILE path; a bare
# /integrity/<project>/<version>/ directory is not an endpoint and 404s.
curl -sS -H "Accept: application/vnd.pypi.integrity.v1+json" \
  https://pypi.org/integrity/scientific-computing-system/2.2.1/scientific_computing_system-2.2.1-py3-none-any.whl/provenance

# The digest PyPI publishes for that file, to compare against the attestation
# subject (predicateType https://docs.pypi.org/attestations/publish/v1):
curl -sS https://pypi.org/pypi/scientific-computing-system/2.2.1/json | python -c \
  "import json,sys; [print(f['filename'], f['digests']['sha256']) for f in json.load(sys.stdin)['urls']]"

# npm: no attestation; there IS a transport signature and an integrity digest.
curl -sS -o /dev/null -w '%{http_code}\n' \
  https://registry.npmjs.org/-/npm/v1/attestations/scientific-computing-system@2.2.1   # -> 404
npm view scientific-computing-system@2.2.1 dist.integrity dist.shasum
```

The published PyPI 2.2.1 digests, useful for pinning:

| File | sha256 |
|---|---|
| `scientific_computing_system-2.2.1-py3-none-any.whl` | `b14b89478fdc8b11482ffd556bb6eb59c6b21887f2a09418edec7029d4569bb9` |
| `scientific_computing_system-2.2.1.tar.gz` | `6ba4ad4c2ebb6bef86356bcbcd251e12978737945e58d9e75b9c68860efa77f6` |

Both were confirmed to equal the `subject` digest inside their attestation
bundles, so the attested bytes and the published bytes are the same bytes.

#### Reproducible builds: what is and is not claimed

The build toolchain is hash-pinned (`requirements-build.lock`, installed with
`--require-hashes --only-binary=:all:`) and the release workflow runs
`python -m build --no-isolation` on that lock.

Two consecutive builds of the same commit with the same pinned backend produced
**byte-identical** wheel and sdist (sha256 `c7734fb5…` and `daa7dca2…`).

Those digests **do not** match the published ones, because that check ran on a
different platform with a locally installed backend rather than the release's
hash-locked Linux toolchain. So the honest statement is: *the build is
deterministic for a fixed toolchain, and this repository does not currently
claim that the published artifacts are bit-for-bit reproducible from source.*
Treat the published digests above as the pin to use, not a rebuild target.

#### When provenance will appear

The OIDC plumbing is implemented on both sides; the registries-side trusted
publisher records are **not yet in place**, so an npm release cannot carry an
attestation yet:

- **PyPI** — Trusted Publisher: owner `Furox-Art`, repository
  `scientific-computing-system`, workflow `release.yml`, environment `pypi`.
  Already exercised: 2.2.1 was published through it and is attested.
- **npm** — Trusted Publisher: owner `Furox-Art`, repository
  `scientific-computing-system`, workflow `npm-publish.yml`, environment `npm`.
  **Pending registration.** Until it exists, the default OIDC path cannot
  complete a publish, and the token fallback — which cannot mint a Sigstore
  identity — is the only working mode. That is why the published npm artifacts
  have no attestation.

  Register it under **Settings → Trusted Publisher → GitHub Actions** and enable
  the **`npm publish`** allowed action. That last field is not optional here: npm's
  current UI allows `npm stage publish` by default on new configurations and makes
  direct `npm publish` opt-in, while this repository's workflow publishes
  directly. A publisher registered without it still fails, with a permission error
  rather than a 404.

  **A missing registration looks like a missing version.** npm will not disclose
  whether a package exists to an unauthenticated caller, so an unrecognised
  trusted publisher produces the same `E404` as an unpublished version:

  ```
  npm error 404 The requested resource 'scientific-computing-system@2.2.2' could
  not be found or you do not have permission to access it.
  ```

  Do not bump the version in response to that line. A bad or missing *token* fails
  differently — `401`, or `ENEEDAUTH` — so `E404` on a PUT for a package that is
  known to exist means the registration, not the version.

**Practical consequence:** for anything security-sensitive, install from PyPI and
check the PEP 740 bundle against the digests PyPI publishes. Do not rely on the
npm channel for supply-chain assurance today; treat its `dist.integrity` as an
integrity pin only.

### Optional Backend Boundary

Optional backends such as SciPy, statsmodels, scikit-learn, SymPy, Z3, h5py, and netCDF4 have their own parser, file-format, numerical, and security behavior. CDS does not make those libraries safe for adversarial input.

In particular, `sympy_verify_identity()` passes caller-provided symbolic strings to SymPy's parser. Do not treat that adapter as a sandbox for hostile expressions. Likewise, HDF5/NetCDF files should be treated according to the security guidance of their respective backend libraries.

### Out of Scope

| Threat | Reasoning |
|---|---|
| **Denial of service from intentionally pathological numerical input** | The library is designed for local scientific/research workloads rather than hostile multi-tenant execution. Resource limits should be applied by the host application when processing untrusted inputs. |
| **Side-channel resistance** | Numerical kernels are not designed or audited as constant-time cryptographic primitives. |
| **Remote confidentiality guarantees** | The core package does not provide a hosted data service. Applications embedding CDS are responsible for their own storage, access control, and network policy. |
| **Cryptographic implementation correctness** | CDS is not a cryptographic library and does not provide custom cryptographic primitives. |

## Known Limitations

- **Pure-Python performance:** many core algorithms prioritize transparency and zero required dependencies rather than accelerated throughput. Large numerical workloads should use appropriate optional accelerated backends.
- **Numerical kernels are not formally verified:** Z3 is available as an optional constraint/formal-verification backend, but that does not imply the numerical library itself has machine-checked proofs.
- **Optional-backend compatibility is a moving boundary:** backend APIs can change independently of CDS. Pin and test the optional scientific stack used by high-assurance deployments.
- **Single-maintainer project:** security response and backport capacity are limited compared with a staffed security team.

## Security Best Practices for Users

1. **Pin the package version** in reproducible environments, for example `pip install "scientific-computing-system==X.Y.Z"` with the exact release you have evaluated, rather than an unconstrained range. Deliberately no concrete version is written here: a pinned example goes stale at the next release and readers copy it regardless, so take the number from the [release notes](https://github.com/Furox-Art/scientific-computing-system/releases) for the release you are deploying, and record it in your own lockfile or constraints file.
2. **Install only the optional extras you need.** Fewer third-party packages reduce supply-chain and compatibility surface.
3. **Verify provenance for high-assurance use.** For PyPI, fetch the PEP 740 bundle from `/integrity/<project>/<version>/<filename>/provenance` and confirm its `subject` digest equals the `sha256` PyPI publishes for that file. There is currently **no** attestation for the npm package; its `dist.integrity` is an integrity pin, not provenance. See [Distribution channels and provenance](#distribution-channels-and-provenance).
4. **Treat optional backend inputs as backend inputs.** Do not pass hostile symbolic expressions or untrusted scientific files without the validation/sandboxing appropriate to SymPy, HDF5, NetCDF, or the relevant backend.
5. **Keep the environment current.** Review dependency updates and run vulnerability auditing against the exact environment deployed.
6. **Do not use the library as the sole validation layer for safety-critical conclusions.** Independent domain validation remains necessary.

## Repository Security Controls

CI includes strict type checking, Ruff lint/format checks, full test coverage gates, property-based tests, dependency auditing, CodeQL, and installed-wheel CLI smoke tests across supported operating systems.

These checks are only effective as merge controls when the default branch is protected by a branch rule/ruleset that requires the relevant status checks. Repository administrators should require the aggregate `CI` check, CodeQL, and Installed CLI Smoke before merges and disallow force-push/deletion of `main`.

## Acknowledgments

Responsible disclosures may be credited in the relevant GitHub Security Advisory with the reporter's consent.

## Contact

- Maintainer: Furox-Art (@Furox-Art)
- Private reporting: https://github.com/Furox-Art/scientific-computing-system/security/advisories/new
