## Summary

<!-- What does this PR do, and why? Link any related issue (e.g. Closes #123). -->

## Types of changes

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Refactor / chore (no behavior change)
- [ ] Docs / CI / build
- [ ] Breaking change (fix or feature that would cause existing functionality to change)

## How was it tested?

<!-- Commands run, test count, manual checks. CI is not a substitute for local verification. -->

```bash
# e.g.
pytest
ruff check src/ tests/ && ruff format --check src/ tests/
mypy src/ && mypy tests/
mkdocs build --strict
```

If you changed documentation, say whether you executed the code snippets you
added or edited. Documentation claims in this project are expected to be
runnable — a snippet that raises `TypeError` is a defect.

## Checklist

- [ ] Tests pass locally (`pytest`): see the CI badge for the current count
- [ ] Lint + format pass (`ruff check src/ tests/` && `ruff format --check src/ tests/`)
- [ ] Type checks pass (`mypy src/`, `mypy tests/`)
- [ ] New code has tests covering it
- [ ] Public API additions are documented (docstrings + `docs/`)
- [ ] `CHANGELOG.md` updated (user-facing changes only)
- [ ] No secrets, credentials, or large binary artifacts committed

## Compatibility

- [ ] The change stays inside the zero-dependency core, or is isolated behind an optional extra
- [ ] Any public API change is reflected in `docs/api.md` and the module tables in `docs/index.md`

## Notes for reviewers

<!-- Anything reviewers should focus on, alternative approaches considered, follow-ups. -->

If this PR touches benchmarks, include the provenance of the numbers: which
script, which commit, which machine. Numbers without provenance cannot be
verified and will be asked about.