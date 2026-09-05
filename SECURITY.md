# Security Policy

## Supported Versions

| Version | Supported | Notes |
|---|---|---|
| 2.0.x | Yes | Current stable release line |
| < 2.0 | No | Superseded by the 2.0 release line unless a specific security advisory states otherwise |

Security fixes target the current stable release line. Older release lines should not be assumed to receive backports unless an advisory explicitly says so.

## Reporting a Vulnerability

If you find a security vulnerability, please **do not open a public issue**.

Use GitHub's private vulnerability reporting feature:

- GitHub: https://github.com/Furox-Art/scientific-computing-system/security/advisories/new

You may also contact the maintainer directly. Reports should include the affected version, a minimal reproduction where practical, impact, and any suggested mitigation.

## Threat Model

`scientific-computing-system` is a local-first, pure-Python scientific-computing library distributed through PyPI. The core package has no required runtime dependencies. Optional scientific backends are loaded only when explicitly requested.

### In Scope

| Threat | Mitigation |
|---|---|
| **Supply chain: malicious or substituted release artifact** | The release workflow is the sole PyPI publish authority. It builds wheel + sdist on a GitHub-hosted runner, verifies package metadata/version, installs the built wheel, smoke-tests the installed CLI, generates GitHub artifact attestation for those exact runner-local files, and publishes through PyPI Trusted Publishing (OIDC). |
| **Registry drift** | Public PyPI is treated as the distribution registry. The release integrity check requires exactly one wheel and one sdist on PyPI and requires the matching GitHub Release to contain no wheel/sdist assets. A registry-policy workflow rechecks the public PyPI package and removes accidental distribution assets from GitHub Releases. |
| **Dependency vulnerabilities** | The core has no required runtime dependencies. Development/test/docs lock files are audited in CI with `pip-audit`; optional backends are isolated behind extras and lazy loading. |
| **Code execution from package install** | The build backend is `hatchling`; there is no `setup.py` execution and package versioning is static in `pyproject.toml` plus `src/cds/_version.py`. |
| **Unexpected scientific-tool loading** | Optional tools are selected through an explicit registry/capability layer and are not imported into the zero-dependency core unless requested. |
| **Untrusted CLI input** | The CLI uses `argparse`-based typed/explicit parsing and does not evaluate arbitrary Python expressions. |
| **Scientific workflow overclaiming** | The research orchestrator is fail-closed: blocked methods, missing tools, denied approvals, incomplete execution, validation failures, and unresolved method suitability prevent an unqualified final conclusion. |

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

1. **Pin the package version** in reproducible environments, for example `scientific-computing-system==2.0.0` rather than an unconstrained range.
2. **Install only the optional extras you need.** Fewer third-party packages reduce supply-chain and compatibility surface.
3. **Verify provenance for high-assurance use.** Compare the PyPI wheel/sdist SHA-256 digests with the subjects recorded by the GitHub release workflow's artifact attestation.
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
