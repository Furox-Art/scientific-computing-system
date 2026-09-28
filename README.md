<p align="center">  
  <img src="assets/logo.svg" alt="scientific-computing-system" width="640">  
</p>  
  
<h1 align="center">scientific-computing-system</h1>  
  
<p align="center"><b>A pure-Python computational science platform for numerical methods, modeling, validation, uncertainty, scientific workflows, dimensional analysis, and reproducible research.</b></p>  
  
<p align="center">  
  <a href="https://pypi.org/project/scientific-computing-system/"><img src="https://img.shields.io/pypi/v/scientific-computing-system.svg" alt="PyPI version"></a>  
  <a href="https://www.npmjs.com/package/scientific-computing-system"><img src="https://img.shields.io/npm/v/scientific-computing-system.svg" alt="npm version"></a>  
  <a href="https://pypi.org/project/scientific-computing-system/"><img src="https://img.shields.io/pypi/dm/scientific-computing-system.svg" alt="PyPI downloads"></a>  
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python 3.10+"></a>  
  <a href="https://codecov.io/gh/Furox-Art/scientific-computing-system"><img src="https://codecov.io/gh/Furox-Art/scientific-computing-system/branch/main/graph/badge.svg" alt="codecov"></a>  
  <a href="https://github.com/Furox-Art/scientific-computing-system/actions/workflows/tests.yml"><img src="https://github.com/Furox-Art/scientific-computing-system/actions/workflows/tests.yml/badge.svg" alt="CI"></a>  
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>  
  <a href="https://furox-art.github.io/scientific-computing-system/"><img src="https://img.shields.io/badge/docs-mkdocs-teal.svg" alt="Docs"></a>  
  <a href="https://github.com/Furox-Art/scientific-computing-system/releases"><img src="https://img.shields.io/github/v/release/Furox-Art/scientific-computing-system.svg" alt="GitHub release"></a>  
</p>  
  
I wrote this because NumPy and SciPy are incredible, but they're also 20 years old and carry two decades of design decisions that don't always make sense anymore.  
  
This is a from-scratch rethinking of what scientific computing in Python could look like if we started today. No C extensions, no Fortran legacy, no dependency hell. Just Python, type hints, and algorithms that are actually readable.  
  
## What's inside  
  
- **Linear algebra**: SVD, QR, Cholesky, eigenvalues-all implemented in pure Python with proper error handling  
- **Optimization**: gradient descent, constrained optimization, metaheuristics  
- **Statistics**: hypothesis testing, Bayesian inference, time series  
- **Machine learning**: PCA, clustering, simple neural nets (educational, not production)  
- **Quantum computing**: circuit simulation, state vectors, basic gates  
- **Signal processing**: filters, wavelets, STFT  
- **ODE/PDE solvers**: stiff and non-stiff, symplectic integrators  
  
## The catch  
  
It's slower than NumPy. Sometimes 10x slower, sometimes 100x. That's the price of pure Python. But it's also completely transparent-you can read every algorithm, understand every step, and modify anything without compiling C.  
  
I use it for prototyping, for teaching, and for cases where I need to know exactly what the computer is doing. For production number crunching, I still reach for NumPy.  
  
## License  
  
MIT. 
