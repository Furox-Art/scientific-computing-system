# Why Pure Python?

Fair question. NumPy exists. Why reinvent it?

## 1. Learning

Every algorithm here is readable top to bottom. Want to know how SVD actually
works? Open `src/cds/math_utils/svd.py`. No Fortran, no BLAS calls, no magic.

The [Tour of Numerical Methods](tour_of_numerical_methods.md) walks through the
solvers with real output so you can check each step against the code.

## 2. Trust

When your simulation gives a weird answer, you can trace every single
operation — including the error paths, because they are ordinary Python
`ValueError`s rather than a segfault. Try doing that with a compiled LAPACK
binding.

## 3. Zero dependencies

`pip install` and you are done. No MKL vs OpenBLAS drama. No broken wheels on
ARM. No version conflicts with the rest of your environment. The core declares
`dependencies = []` and ships a `py.typed` marker, so your type checker sees
real signatures.

## The honest tradeoff

**It is slow, and the gap is large.** On one reference run, 100×100 matrix
multiplication took ~0.05 s in CDS versus ~0.0001 s for NumPy — roughly
580× slower. The exact numbers vary by machine; see
[Benchmarks](benchmarks.md) for the full report, the hardware it ran on, and the
raw JSON behind it.

The rule of thumb that matters: if your workload is dominated by dense linear
algebra on large arrays, pure Python is the wrong tool and the slowdown is not
worth paying.

## When to use something else anyway

- **Production HPC, GPU compute, or very large array workloads** — use NumPy,
  SciPy, JAX, or PyTorch directly.
- **Needing speed and wanting this project's API surface** — see
  [`scientific-computing-system-2.0`](https://github.com/Furox-Art/scientific-computing-system-2.0),
  a separate project that takes the opposite trade-off: it builds on
  NumPy/SciPy/pandas/matplotlib and adds compiled and GPU acceleration.
- **Safety-critical conclusions** — CDS's optional backends can serve as an
  *independent* reference implementation to cross-check, but per
  [SECURITY.md](https://github.com/Furox-Art/scientific-computing-system/blob/main/SECURITY.md)
  it must not be the sole validation layer.

## When CDS is the right choice

- Learning and teaching numerical methods where the algorithm *is* the lesson.
- Auditing an implementation line by line.
- Prototyping across several scientific domains under one importable package
  with no build step.
- Reproducible local work where you want an inspectable path from input to
  result — validation checks, uncertainty propagation, sensitivity analysis,
  and provenance capture are all in the box.

## A note on scope

The machine-learning and NLP modules are explicitly **educational**: they are
readable from-scratch implementations, not competitive implementations. Do not
put them on a production path.