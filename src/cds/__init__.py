"""
scientific-computing-system

Pure Python computational science system for research, simulation,
and scientific discovery.

Key features:
- Zero heavy dependencies (pure Python)
- Quantum simulation (single & multi-qubit with entanglement)
- Signal processing (FFT, 2D FFT, convolution, filtering)
- Optimization, statistics, probability, linear algebra
- Hypothesis generation for structured research ideas
- Workflow orchestration with approval gates and validation
- Uncertainty propagation and reproducibility provenance
- SI units, dimensional analysis, and scalable local data I/O
- Optional scientific-tool capability discovery
- CLI for quick calculations and discovery workflows

All modules are designed to be readable, testable, and usable
for education, research, and custom scientific discovery workflows.

Usage:
    import cds
    print(cds.__version__)

    from cds.quantum import ghz_state, is_entangled
    from cds.hypothesis import generate_hypotheses
"""

# Convenient top-level re-exports for common scientific tools
# Core modules
# Scientific computing modules
from cds import (
    bayes,
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
    "core",
    "data_analysis",
    "data_io",
    "diffeq",
    "fractals",
    "genetics",
    "graph",
    "hypothesis",
    "infotheory",
    "knowledge",
    "math_utils",
    "ml",
    "modeling",
    "montecarlo",
    "nlp",
    "numerical_integration",
    "optimization",
    "plot",
    "probability",
    "provenance",
    "quantum",
    "scientific",
    "signals",
    "stats",
    "tools",
    "uncertainty",
    "units",
    "validation",
    "wavelets",
    "workflow",
]
