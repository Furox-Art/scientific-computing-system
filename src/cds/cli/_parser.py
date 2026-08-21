"""Argument-parser construction for the ``cds`` CLI."""

from __future__ import annotations

import argparse

from cds.cli._handlers import (
    _cmd_benchmark,
    _cmd_calc,
    _cmd_constants,
    _cmd_dashboard,
    _cmd_hypothesis,
    _cmd_info,
    _cmd_integrate,
    _cmd_modules,
    _cmd_plot,
    _cmd_prompt,
    _cmd_sample,
    _cmd_stats,
    _cmd_version,
)

_DOMAIN_CHOICES = ["physics", "cosmology", "mathematics", "biology", "chemistry", "general_science"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``cds`` argument parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="cds",
        description="Cognitive Discovery System — computational science platform.",
    )
    parser.add_argument("--version", "-v", action="store_true", help="Show System version and exit")

    sub = parser.add_subparsers(dest="command")

    p_version = sub.add_parser("version", help="Show System version.")
    p_version.set_defaults(func=_cmd_version)

    p_hyp = sub.add_parser(
        "hypothesis", help="Generate scientific hypotheses for a research question."
    )
    p_hyp.add_argument("question", help="The core research question or problem")
    p_hyp.add_argument(
        "--domain",
        "-d",
        default="general_science",
        choices=_DOMAIN_CHOICES,
        help="Scientific domain focus",
    )
    p_hyp.add_argument("--num", "-n", type=int, default=3, help="Number of hypotheses to propose")
    p_hyp.add_argument("--output", "-o", help="Save results as JSON")
    p_hyp.add_argument("--show-prompt", action="store_true", help="Print the exact prompt template")
    p_hyp.add_argument("--dry-run", action="store_true", help="Do not run generation logic")
    p_hyp.set_defaults(func=_cmd_hypothesis)

    p_prompt = sub.add_parser("prompt", help="Print a ready-to-use prompt for a custom generator.")
    p_prompt.add_argument("question", help="Research question")
    p_prompt.add_argument(
        "--domain",
        "-d",
        default="general_science",
        choices=_DOMAIN_CHOICES,
        help="Scientific domain focus",
    )
    p_prompt.add_argument(
        "--num", "-n", type=int, default=3, help="Number of hypotheses to propose"
    )
    p_prompt.set_defaults(func=_cmd_prompt)

    p_info = sub.add_parser("info", help="Show System info, module status, and System health.")
    p_info.set_defaults(func=_cmd_info)

    sub.add_parser("dashboard", help="Launch the interactive System dashboard.").set_defaults(
        func=_cmd_dashboard
    )
    sub.add_parser("benchmark", help="Run built-in benchmarks to verify performance.").set_defaults(
        func=_cmd_benchmark
    )
    sub.add_parser("constants", help="List available physical constants.").set_defaults(
        func=_cmd_constants
    )

    p_plot = sub.add_parser(
        "plot",
        help="Plot a series of numbers (ASCII terminal, or PNG with --file if cds[plot] installed).",
    )
    p_plot.add_argument("values", help="Comma-separated list of numbers (e.g. '1,5,3,8')")
    p_plot.add_argument("--title", "-t", default="CLI Plot", help="Title of the plot")
    p_plot.add_argument(
        "--file",
        "-f",
        default=None,
        help="Save a PNG via optional matplotlib (requires cds[plot]); omit for ASCII",
    )
    p_plot.add_argument(
        "--kind",
        "-k",
        default="series",
        choices=["series", "hist", "acf"],
        help="Chart type when using --file (default: series)",
    )
    p_plot.set_defaults(func=_cmd_plot)

    p_calc = sub.add_parser("calc", help="Quick physics calculations.")
    p_calc.add_argument("formula", help="Formula: ke, gravity, wave, gas")
    p_calc.set_defaults(func=_cmd_calc)

    p_stats = sub.add_parser(
        "stats", help="Descriptive statistics for a comma-separated number list."
    )
    p_stats.add_argument("values", help="Comma-separated numbers (e.g. '1,2,3,4')")
    p_stats.set_defaults(func=_cmd_stats)

    p_sample = sub.add_parser("sample", help="Draw samples from a probability distribution.")
    p_sample.add_argument(
        "dist",
        choices=["uniform", "gaussian", "exponential", "poisson"],
        help="Distribution name",
    )
    p_sample.add_argument("-n", type=int, default=5, help="Number of samples (default 5)")
    p_sample.add_argument("--seed", type=int, default=None, help="RNG seed")
    p_sample.add_argument("--a", type=float, default=0.0, help="uniform lower bound")
    p_sample.add_argument("--b", type=float, default=1.0, help="uniform upper bound")
    p_sample.add_argument("--mu", type=float, default=0.0, help="gaussian mean")
    p_sample.add_argument("--sigma", type=float, default=1.0, help="gaussian stdev")
    p_sample.add_argument("--lam", type=float, default=1.0, help="rate λ for exponential/poisson")
    p_sample.set_defaults(func=_cmd_sample)

    p_int = sub.add_parser(
        "integrate", help="Numerically integrate a built-in function over [a, b]."
    )
    p_int.add_argument(
        "integrand",
        choices=["sin", "cos", "exp", "x2", "unit"],
        help="Integrand name",
    )
    p_int.add_argument("--a", type=float, default=0.0, help="Lower limit (default 0)")
    p_int.add_argument("--b", type=float, default=1.0, help="Upper limit (default 1)")
    p_int.add_argument("-n", type=int, default=1000, help="Number of panels (default 1000)")
    p_int.add_argument(
        "--method",
        choices=["simpson", "trap"],
        default="simpson",
        help="Quadrature rule (default simpson)",
    )
    p_int.set_defaults(func=_cmd_integrate)

    sub.add_parser(
        "modules", help="List all scientific modules available in the System."
    ).set_defaults(func=_cmd_modules)

    return parser


# Historical private name kept as an alias for backwards compatibility.
_build_parser = build_parser
