"""Current architecture and module-catalog handlers for the ``cds`` CLI."""

from __future__ import annotations

import argparse

from cds.cli._style import _format_table, _print, _render

_MODULE_INFO: tuple[tuple[str, str], ...] = (
    ("cds.quantum", "Single/multi-qubit circuits, Bell/GHZ states, entanglement"),
    ("cds.signals", "DFT/FFT, convolution, filtering, spectral utilities"),
    ("cds.math_utils", "Linear algebra, decompositions, calculus utilities"),
    ("cds.optimization", "Gradient/Newton/Adam, Nelder-Mead, annealing, search"),
    ("cds.stats", "Inference, regression, tests, time-series statistics"),
    ("cds.probability", "Distributions, CDFs, quantiles, and sampling"),
    ("cds.montecarlo", "Monte Carlo integration, simulation, and random walks"),
    ("cds.diffeq", "Explicit/adaptive/stiff ODE solvers and PDE utilities"),
    ("cds.modeling", "Symbolic models, equation solving, and parameter fitting"),
    ("cds.ml", "Classical ML estimators, preprocessing, validation, PCA"),
    ("cds.data_analysis", "Tabular analysis, normalization, visualization helpers"),
    ("cds.data_io", "Memory-bounded streaming plus optional HDF5/NetCDF adapters"),
    ("cds.units", "SI units, conversions, and dimensional analysis"),
    ("cds.uncertainty", "Analytic/correlated Monte Carlo uncertainty propagation"),
    ("cds.sensitivity", "Dependency-free local parameter sensitivity analysis"),
    ("cds.validation", "Scientific checks, cross-method verification, final audit"),
    ("cds.workflow", "Approval-gated scientific workflow orchestration"),
    ("cds.provenance", "Run manifests, hashes, tool versions, decisions, checkpoints"),
    ("cds.tools", "Lazy scientific backends plus SciPy/SymPy/Z3 adapters"),
    ("cds.knowledge", "Knowledge graph, concept mapping, notes, retrieval"),
    ("cds.hypothesis", "Structured scientific hypothesis generation"),
    ("cds.scientific", "Physical constants and common scientific formulas"),
    ("cds.numerical_integration", "Adaptive and fixed numerical quadrature"),
    ("cds.graph", "Graph traversal, shortest paths, MST, topological sort"),
    ("cds.nlp", "Educational tokenizer, embeddings, attention, MiniGPT"),
    ("cds.plot", "Optional matplotlib scientific plots"),
)


def _cmd_info(args: argparse.Namespace) -> int:
    """Show the installed system architecture and health summary."""
    from cds import __version__

    _print(_render("[bold]System (CDS)[/]"))
    _print(_render("[dim]Pure Python scientific computing system[/]"))
    _print("")
    _print(_render("[bold green]Status:[/] Stable"))
    _print(_render("[bold blue]Tests:[/] full suite green in CI (see badge)"))
    _print(_render("[bold magenta]Core deps:[/] 0 External (Pure Python core)"))
    _print(_render("[bold yellow]Optional tools:[/] scientific, I/O, plotting, dashboard extras"))
    _print(_render(f"[bold cyan]Version:[/] {__version__}"))
    _print("")
    _print(_render("[bold]Architecture:[/]"))
    for line in (
        "compute       quantum / signals / math / ODE-PDE / integration",
        "analysis      stats / probability / ML / modeling / sensitivity",
        "assurance     validation / uncertainty / units / provenance",
        "orchestration workflow / optional scientific tools",
        "data          data_analysis / streaming I/O / knowledge",
    ):
        _print(f"  • {line}")
    return 0


def _cmd_modules(args: argparse.Namespace) -> int:
    """List the current scientific modules available in the System."""
    rows = [[name, description] for name, description in _MODULE_INFO]
    _print(_format_table("System Scientific Modules", ["Module", "Key Capabilities"], rows))
    _print(_render("\n[dim]Core remains pure Python with zero runtime dependencies.[/]"))
    _print("Optional backends: pip install 'scientific-computing-system[scientific,io,plot]'")
    return 0
