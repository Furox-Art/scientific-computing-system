"""
scientific-computing-system

Pure Python computational science system for research, simulation,
and scientific discovery.

Key features:
- Zero heavy dependencies (pure Python)
- Quantum simulation (single & multi-qubit with entanglement)
- Signal processing (FFT, 2D FFT, convolution, filtering)
- Optimization, statistics, probability, linear algebra
- Multiple-testing/FDR corrections and conservative causal estimators
- Hypothesis generation for structured research ideas
- Workflow orchestration with approval gates and validation
- Uncertainty propagation, local/global sensitivity, and identifiability analysis
- Distribution-drift and out-of-distribution validation
- SI units, dimensional analysis, and memory-bounded scientific data I/O
- Optional scientific-tool capability discovery and normalized adapters
- CLI for quick calculations and discovery workflows

All modules are designed to be readable, testable, and usable
for education, research, and custom scientific discovery workflows.

Usage:
    import cds
    print(cds.__version__)

    from cds.quantum import ghz_state, is_entangled
    from cds.hypothesis import generate_hypotheses
"""

from __future__ import annotations

import importlib
from types import ModuleType

# Subpackages published as ``cds.<name>`` but not part of the eager import
# block below. ``cds.pde`` and ``cds.interpolate`` ship as their own
# subpackages, so ``import cds.pde`` and ``from cds.pde import solve_heat``
# always worked — but the flat package layout meant ``import cds`` alone never
# bound the attribute, and ``cds.pde`` raised ``AttributeError`` even though
# docs/index.md and docs/api.md both advertise it as public surface.
#
# A PEP 562 module ``__getattr__`` closes the gap without moving the two
# solvers onto the import path of every ``cds`` user. The imported module is
# cached in ``globals()`` so the lookup happens exactly once.
_LAZY_SUBPACKAGES = frozenset({"interpolate", "pde"})


def __getattr__(name: str) -> ModuleType:
    """Lazily import a documented ``cds`` subpackage on first attribute access."""
    if name in _LAZY_SUBPACKAGES:
        module = importlib.import_module(f"cds.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose the lazily-imported subpackages to ``dir(cds)``."""
    return sorted(set(globals()) | _LAZY_SUBPACKAGES)


# Convenient top-level re-exports for common scientific tools
# Core modules
# Scientific computing modules
from cds import (
    bayes,
    causal,
    core,
    data_analysis,
    data_io,
    diffeq,
    fractals,
    genetics,
    graph,
    hypothesis,
    infotheory,
    knowledge,
    math_utils,
    ml,
    modeling,
    montecarlo,
    nlp,
    numerical_integration,
    optimization,
    plot,
    probability,
    provenance,
    quantum,
    scientific,
    sensitivity,
    signals,
    stats,
    tools,
    uncertainty,
    units,
    validation,
    wavelets,
    workflow,
)
from cds._version import __version__
from cds.scientific.constants import CONSTANTS, get_constant
from cds.scientific.formulas import (
    centripetal_acceleration,
    coulomb_force,
    de_broglie_wavelength,
    doppler_frequency,
    escape_velocity,
    gravitational_force,
    ideal_gas_pressure,
    kinetic_energy,
    pendulum_period,
    photon_energy,
    schwarzschild_radius,
    wave_frequency,
)

__all__ = [
    "__version__",
    "CONSTANTS",
    "get_constant",
    "kinetic_energy",
    "gravitational_force",
    "wave_frequency",
    "ideal_gas_pressure",
    "schwarzschild_radius",
    "de_broglie_wavelength",
    "escape_velocity",
    "photon_energy",
    "coulomb_force",
    "centripetal_acceleration",
    "pendulum_period",
    "doppler_frequency",
    "bayes",
    "causal",
    "core",
    "data_analysis",
    "data_io",
    "diffeq",
    "fractals",
    "genetics",
    "graph",
    "hypothesis",
    "infotheory",
    "interpolate",
    "knowledge",
    "math_utils",
    "ml",
    "modeling",
    "montecarlo",
    "nlp",
    "numerical_integration",
    "optimization",
    "pde",
    "plot",
    "probability",
    "provenance",
    "quantum",
    "scientific",
    "sensitivity",
    "signals",
    "stats",
    "tools",
    "uncertainty",
    "units",
    "validation",
    "wavelets",
    "workflow",
]
