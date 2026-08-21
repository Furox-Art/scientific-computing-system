# CDS Quick Start Tutorial

Welcome to the **Cognitive Discovery System (CDS)**. This guide will show you how to perform scientific calculations with **Zero Dependencies**.

## 1. Setup and Version Check

First, let's make sure the platform is installed and check the version:

```python
import cds

print(f"CDS Version: {cds.__version__}")
```

## 2. Basic Math and Constants

CDS provides high-precision physical constants and standard formulas:

```python
from cds.scientific import get_constant, kinetic_energy

c = get_constant("c")
print(f"Speed of light: {c} m/s")

ke = kinetic_energy(mass=10, velocity=5)
print(f"Kinetic Energy of 10kg at 5m/s: {ke} J")
```

## 3. Statistical Analysis

Perform descriptive statistics and linear regressions effortlessly:

```python
from cds.stats import mean, stdev, linear_regression

data = [12.5, 14.3, 11.8, 15.1, 13.7]
print(f"Mean: {mean(data):.2f}")
print(f"Std Dev: {stdev(data):.2f}")

x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.2, 7.8, 10.1]
reg = linear_regression(x, y)
print(f"Regression: y = {reg.slope:.2f}x + {reg.intercept:.2f}")
```
