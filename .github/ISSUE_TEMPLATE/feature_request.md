---
name: Feature request
about: Suggest a new module, function, or improvement
title: "[FEATURE] "
labels: enhancement
assignees: ''
---

> Before filing: check whether an issue or Discussion already covers this, and
> whether a workaround is documented in the
> [Cookbook](https://furox-art.github.io/scientific-computing-system/cookbook/).

## Problem
What problem does this feature solve? What are you trying to do that CDS doesn't let you do today?

## Proposed solution
Describe the feature/module you'd like. A short API sketch helps a lot:

```python
import cds
# Ideal usage:
result = cds.<module>.<function>(...)
```

## Which module(s)?
e.g. `cds.quantum`, `cds.numerical_integration`, a brand-new module, CLI, docs...

## Zero-dependency constraint
CDS core has **no runtime dependencies** and must stay pure standard library.
If your proposal needs a third-party package, say which extra it would live
behind (`[scientific]`, `[io]`, `[plot]`, `[pandas]`) and how it would be loaded
lazily. A feature that would make a core module import NumPy cannot be accepted
as-is.

## Alternatives considered
Have you tried any workarounds? Are there other libraries that do this?

## Additional context
Links to papers, references, or related issues.

Documentation-only fixes are very welcome too — if the docs and the code
disagree, that is a real bug and you can open it as a bug report.
