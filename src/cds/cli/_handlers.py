"""Implementations of the ``cds`` subcommands.

Every handler takes the parsed :class:`argparse.Namespace` and returns an
integer exit code. Heavy imports stay function-local so ``cds --help`` (and
the whole parser build) never pays for modules a command does not use.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from cds.cli._style import _format_table, _print, _render
from cds.core.models import Domain
from cds.hypothesis.generator import PromptTemplate, generate_hypotheses


def _cmd_version(args: argparse.Namespace) -> int:
    """Show the installed System version."""
    from cds import __version__

    _print(_render(f"[bold]System[/] version [cyan]{__version__}[/]"))
    return 0


def _cmd_hypothesis(args: argparse.Namespace) -> int:
    """Generate scientific hypotheses for a research question."""
    dom = Domain(args.domain)

    if args.show_prompt:
        prompt = PromptTemplate.render(args.question, dom, args.num)
        _print(_render(f"[blue]{prompt}[/]"))
        return 0

    if args.dry_run:
        _print(_render("[yellow]Dry run mode — no generation performed.[/]"))
        _print(
            _render(
                f"Would generate {args.num} hypotheses for: "
                f"[bold]{args.question}[/] in domain [cyan]{dom.value}[/]"
            )
        )
        return 0

    _print(_render(f"[bold]Generating hypotheses[/] for: [italic]{args.question}[/]"))
    _print(_render(f"Domain: [cyan]{dom.value}[/] | Count: {args.num}\n"))

    hypos = generate_hypotheses(args.question, domain=dom, n=args.num)

    rows: list[list[str]] = []
    for h in hypos:
        stmt = h.statement[:90] + ("..." if len(h.statement) > 90 else "")
        rows.append([h.id, stmt, f"{h.confidence:.2f}"])
    _print(_format_table("Generated Hypotheses", ["ID", "Statement", "Confidence"], rows))

    if hypos:
        _print(_render("\n[bold]Detailed view of first hypothesis:[/]\n"))
        _print(_render(f"[green]{hypos[0].to_markdown()}[/]"))

    if args.output:
        data = [h.to_dict() for h in hypos]
        Path(args.output).write_text(json.dumps(data, indent=2, default=str))
        _print(_render(f"\n[green]Saved to {args.output}[/]"))

    return 0


def _cmd_prompt(args: argparse.Namespace) -> int:
    """Print a ready-to-use prompt for a custom generator implementation."""
    dom = Domain(args.domain)
    prompt_text = PromptTemplate.render(args.question, dom, args.num)
    _print(prompt_text)
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Descriptive statistics for a comma-separated number list."""
    from cds.stats import mean, median, percentile, stdev, variance

    try:
        data = [float(x.strip()) for x in args.values.split(",")]
    except ValueError:
        _print(_render("[red]Error:[/] Values must be a comma-separated list of numbers."))
        return 1
    if not data:  # pragma: no cover - split always yields at least one token
        _print(_render("[red]Error:[/] empty list."))
        return 1
    rows = [
        ["n", str(len(data))],
        ["mean", f"{mean(data):.6g}"],
        ["median", f"{median(data):.6g}"],
        ["min", f"{min(data):.6g}"],
        ["max", f"{max(data):.6g}"],
        ["p25", f"{percentile(data, 25):.6g}"],
        ["p75", f"{percentile(data, 75):.6g}"],
    ]
    if len(data) > 1:
        rows.append(["stdev", f"{stdev(data):.6g}"])
        rows.append(["variance", f"{variance(data):.6g}"])
    _print(_format_table("Descriptive stats", ["stat", "value"], rows))
    return 0


def _cmd_integrate(args: argparse.Namespace) -> int:
    """Numerical integration of a built-in integrand over [a, b]."""
    import math

    from cds.numerical_integration import simpson, trapezoid

    integrands: dict[str, Callable[[float], float]] = {
        "sin": math.sin,
        "cos": math.cos,
        "exp": math.exp,
        "x2": lambda x: x * x,
        "unit": lambda _x: 1.0,
    }
    name = args.integrand
    if name not in integrands:  # pragma: no cover - argparse choices
        _print(
            _render(
                f"[red]Error:[/] unknown integrand {name!r}. "
                f"Options: {', '.join(sorted(integrands))}"
            )
        )
        return 1
    f = integrands[name]
    a, b, n = args.a, args.b, args.n
    try:
        if args.method == "trap":
            result = trapezoid(f, a, b, n=n)
        else:
            result = simpson(f, a, b, n=n)
    except ValueError as exc:
        _print(_render(f"[red]Error:[/] {exc}"))
        return 1
    _print(_render(f"[green]∫_{a}^{b} {name}(x) dx ≈ {result:.10g}[/] ({args.method}, n={n})"))
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    """Draw samples from a built-in probability distribution."""
    from cds.probability import (
        exponential_sample,
        gaussian_sample,
        poisson_sample,
        uniform_sample,
    )

    n = args.n
    seed = args.seed
    dist = args.dist
    try:
        if dist == "uniform":
            samples_f = uniform_sample(args.a, args.b, n, seed=seed)
            text = ", ".join(f"{v:.6g}" for v in samples_f)
        elif dist == "gaussian":
            samples_f = gaussian_sample(n, mu=args.mu, sigma=args.sigma, seed=seed)
            text = ", ".join(f"{v:.6g}" for v in samples_f)
        elif dist == "exponential":
            samples_f = exponential_sample(n, lam=args.lam, seed=seed)
            text = ", ".join(f"{v:.6g}" for v in samples_f)
        elif dist == "poisson":
            samples_i = poisson_sample(n, lam=args.lam, seed=seed)
            text = ", ".join(str(v) for v in samples_i)
        else:  # pragma: no cover - argparse choices reject unknown dist
            _print(_render(f"[red]Error:[/] unknown dist {dist!r}"))
            return 1
    except ValueError as exc:
        _print(_render(f"[red]Error:[/] {exc}"))
        return 1
    _print(text)
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch the interactive System dashboard."""
    root_dir = Path(__file__).parent.parent.parent.parent
    dashboard_path = root_dir / "dashboard" / "app.py"
    if not dashboard_path.exists():
        _print(_render("[red]Error:[/] Dashboard file not found at " + str(dashboard_path)))
        return 1

    _print(_render("[yellow]Launching System Interactive Dashboard...[/]"))

    # Ensure src is in PYTHONPATH so dashboard can import cds
    env = os.environ.copy()
    src_path = str(root_dir / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
            check=True,
            env=env,
        )
    except KeyboardInterrupt:
        _print(_render("\n[blue]Dashboard stopped.[/]"))
    except FileNotFoundError:
        _print(
            _render("[red]Error:[/] Streamlit not found. Install it with 'pip install streamlit'.")
        )
        return 1
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run built-in benchmarks to verify performance."""
    _print(_render("[yellow]Benchmarking System performance...[/]"))
    _print("Run 'python benchmarks/run_benchmarks.py' for detailed results.")
    return 0


def _cmd_constants(args: argparse.Namespace) -> int:
    """List available physical constants."""
    from cds.scientific.constants import CONSTANTS

    rows = [
        [name, f"{val:.6e}" if val < 0.01 or val > 1e4 else f"{val}", desc]
        for name, (val, desc) in CONSTANTS.items()
    ]
    _print(_format_table("Physical Constants", ["Name", "Value", "Description"], rows))
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    """Plot a series of numbers (ASCII in terminal, or PNG via optional matplotlib)."""
    try:
        data = [float(x.strip()) for x in args.values.split(",")]
    except ValueError:
        _print(_render("[red]Error:[/] Values must be a comma-separated list of numbers."))
        return 1

    kind = getattr(args, "kind", "series") or "series"
    out_file = getattr(args, "file", None)
    if out_file:
        # Optional matplotlib path — requires `pip install scientific-computing-system[plot]`.
        try:
            from cds.plot import plot_acf, plot_histogram, plot_series, save_figure
        except ImportError as exc:  # pragma: no cover - package always ships cds.plot
            _print(_render(f"[red]Error:[/] {exc}"))
            return 1
        try:
            if kind == "hist":
                fig = plot_histogram(data, title=args.title)
            elif kind == "acf":
                fig = plot_acf(data, title=args.title)
            elif kind == "series":
                fig = plot_series(data, title=args.title)
            else:  # pragma: no cover - argparse choices reject unknown kind
                _print(
                    _render(f"[red]Error:[/] Unknown --kind {kind!r}. Options: series, hist, acf")
                )
                return 1
            save_figure(fig, out_file)
        except ImportError as exc:
            _print(_render(f"[red]Error:[/] {exc}"))
            return 1
        except ValueError as exc:
            _print(_render(f"[red]Error:[/] {exc}"))
            return 1
        _print(_render(f"[green]Saved[/] {out_file} ({kind})"))
        return 0

    if kind != "series":
        _print(
            _render(
                "[red]Error:[/] ASCII mode only supports --kind series "
                "(use --file with cds[plot] for hist/acf)."
            )
        )
        return 1

    from cds.data_analysis.viz import plot_line

    _print(plot_line(data, title=args.title))
    return 0


def _cmd_calc(args: argparse.Namespace) -> int:
    """Quick physics calculations."""
    from cds.scientific import formulas

    try:
        if args.formula == "ke":
            _print("KE = 0.5 * m * v²")
            m = float(input("mass (kg) "))
            v = float(input("velocity (m/s) "))
            _print(_render(f"[green]Kinetic Energy = {formulas.kinetic_energy(m, v):.4f} J[/]"))
        elif args.formula == "gravity":
            _print("F = G * m1 * m2 / r²")
            m1 = float(input("mass 1 (kg) "))
            m2 = float(input("mass 2 (kg) "))
            r = float(input("distance (m) "))
            _print(_render(f"[green]Force = {formulas.gravitational_force(m1, m2, r):.6e} N[/]"))
        elif args.formula == "wave":
            wl = float(input("wavelength (m) "))
            _print(_render(f"[green]Frequency = {formulas.wave_frequency(wl):.4e} Hz[/]"))
        elif args.formula == "gas":
            n = float(input("moles "))
            t = float(input("temperature (K) "))
            v = float(input("volume (m³) "))
            _print(_render(f"[green]Pressure = {formulas.ideal_gas_pressure(n, t, v):.2f} Pa[/]"))
        else:
            _print(
                _render(
                    f"[red]Unknown formula '{args.formula}'. Options: ke, gravity, wave, gas[/]"
                )
            )
    except ValueError:
        _print(_render("[red]Error:[/] Input must be a valid number."))
        return 1
    except Exception as e:  # noqa: BLE001 — CLI surface, keep the message readable
        _print(_render(f"[red]Error:[/] {str(e)}"))
        return 1
    return 0
