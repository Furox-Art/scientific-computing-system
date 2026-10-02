---
name: Bug report
about: Report something that isn't working as expected
title: "[BUG] "
labels: bug
assignees: ''
---

> **Security vulnerability?** Do **not** open a public issue.
> Report it through GitHub's private advisory form:
> https://github.com/Furox-Art/scientific-computing-system/security/advisories/new
> See [SECURITY.md](../../SECURITY.md) for the policy, response expectations, and threat model.
> Use this template only for ordinary bugs.
>
> **Usage question rather than a bug?** Use the Question template instead.

---

## Describe the bug
A clear and concise description of what the bug is.

## To reproduce
Steps to reproduce the behavior:
1. Run `...`
2. Call `...`
3. See error

```python
# Minimal reproducible example
import cds

...
```

## Expected behavior
What you expected to happen.

## Actual behavior / traceback
```
Paste the full error or unexpected output here.
```

## Environment
- CDS version: `cds --version` →
- Python version:
- OS:
- How installed: `pip install scientific-computing-system` / from source / other
- Optional extras installed (`[plot]`, `[io]`, `[pandas]`, `[scientific]`):

## Module / function affected
e.g. `cds.quantum`, `cds.numerical_integration.simpson`, `cds.cli`

Reminder: the distribution is `scientific-computing-system`, the import name is
`cds`, and the CLI command is `cds` — not `scs`, which is a different project.

## Additional context
Anything else relevant.
