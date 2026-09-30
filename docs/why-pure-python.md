# Why Pure Python?

Fair question. NumPy exists. Why reinvent it?

## 1. Learning

Every algorithm here is readable top to bottom. Want to know how SVD actually works? Open the file. No Fortran, no BLAS calls, no magic.

## 2. Trust

When your simulation gives a weird answer, you can trace every single operation. Try doing that with a compiled LAPACK binding.

## 3. Zero dependencies

pip install and you are done. No MKL vs OpenBLAS drama. No broken wheels on ARM. No version conflicts with your other packages.

## The honest tradeoff

It is slow. For production workloads, use scientific-computing-system-2.0 instead, which wraps NumPy/SciPy with the same philosophy but actual speed.
