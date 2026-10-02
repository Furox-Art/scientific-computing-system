---
title: "scientific-computing-system: readable numerical methods in pure Python"
tags:
  - Python
  - scientific computing
  - numerical methods
authors:
  - name: Furox-Art
    affiliation: 1
affiliations:
  - name: Independent
    index: 1
date: 2 October 2026
bibliography: paper.bib
---

# Summary

scientific-computing-system is a pure-Python library for numerical methods,
modeling, validation, uncertainty, and scientific workflows. The runtime
dependency list is empty. Linear algebra, ordinary differential equations,
statistics, signal processing, Monte Carlo, and related routines are written
in Python so a reader can open one module and see the algorithm. The library
is slower than NumPy and SciPy, often by one or two orders of magnitude, and
it does not claim to replace them. A separate package,
scientific-computing-system-2.0, is the NumPy build of the same project and
is not described here.

# Statement of need

Teaching and review of numerical code usually stops at a compiled library.
NumPy [@harris2020array] and SciPy [@virtanen2020scipy] are the right tools
for production array computation, and their internals are not a comfortable
place to read a Runge-Kutta step or a small SVD. Students and reviewers who
need the implementation itself end up rewriting a fragment, or trusting a
notebook that no longer runs.

This package is that readable layer. It installs with `pip` on Python 3.10
or newer and imports no third-party module at runtime. Optional extras exist
for tests, docs, and for users who want a NumPy or HDF5 bridge; the core
does not require them. Algorithms are covered by automated tests in the
repository. The documentation site walks through the solvers with examples
that are part of the repository, not a separate unverifiable tutorial.

The software is not a new numerical result and not a faster backend. Use
NumPy or SciPy when the question is throughput. Use this package when the
question is what the method does, or when a deployment cannot take a
compiled scientific stack. The NumPy-accelerated install lives in another
repository on purpose, so this paper has one subject.

# Acknowledgements

None.

# References
