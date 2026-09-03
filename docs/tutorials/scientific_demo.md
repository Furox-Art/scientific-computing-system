# Scientific Constants & Formulas Tutorial

CDS bundles a curated set of physical constants and classical physics formulas, all pure Python.

## 1. Lookup a Constant

```python
from cds.scientific import get_constant, CONSTANTS

print(get_constant("c"))  # speed of light, m/s
print(sorted(CONSTANTS))  # all available names
```

## 2. Mechanics

```python
from cds.scientific import kinetic_energy, gravitational_force, escape_velocity

print(kinetic_energy(mass=10.0, velocity=5.0))  # Joules
print(gravitational_force(m1=5.0, m2=10.0, r=2.0))  # Newtons
print(escape_velocity(mass=5.972e24, radius=6.371e6))  # Earth, m/s
```

## 3. Waves, Photons & Relativity

```python
from cds.scientific import (
    photon_energy,
    wave_frequency,
    de_broglie_wavelength,
    schwarzschild_radius,
)

print(photon_energy(frequency=5.0e14))  # green photon, J
print(wave_frequency(wavelength=500e-9))  # Hz
print(de_broglie_wavelength(mass=9.109e-31, velocity=1e6))  # electron, m
print(schwarzschild_radius(mass=1.989e30))  # Sun, m
```

## 4. Thermodynamics

```python
from cds.scientific import ideal_gas_pressure

print(ideal_gas_pressure(n_moles=1.0, temperature=300.0, volume=0.024))  # Pa
```
