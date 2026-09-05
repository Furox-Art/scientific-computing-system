"""System command-line interface.

Pure-stdlib CLI built on :mod:`argparse`. It replaces the previous
``typer``/``rich`` implementation so the whole ``cds`` package stays
zero-dependency at runtime. Rich-style colour is reproduced with small ANSI
escape helpers; the textual output (help text, table contents, prompts) is
preserved verbatim where the test suite asserts on it.

The package is split for maintainability:

- :mod:`cds.cli._style` — ANSI colour + ASCII table rendering
- :mod:`cds.cli._handlers` — one function per computational subcommand
- :mod:`cds.cli._system_info` — current architecture and module catalog
- :mod:`cds.cli._parser` — argument-parser wiring

The entry point :func:`main` accepts an optional ``argv`` so tests can drive a
specific command without spawning a subprocess, and returns the integer exit
code instead of calling :func:`sys.exit` directly when ``argv`` is given.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from cds.cli._handlers import (
    _cmd_benchmark,
    _cmd_calc,
    _cmd_constants,
    _cmd_dashboard,
    _cmd_hypothesis,
    _cmd_integrate,
    _cmd_plot,
    _cmd_prompt,
    _cmd_sample,
    _cmd_stats,
    _cmd_version,
)
from cds.cli._parser import _build_parser, build_parser
from cds.cli._style import (
    _format_table,
    _print,
    _render,
    _supports_color,
    _wrap,
)
from cds.cli._system_info import _cmd_info, _cmd_modules

__all__ = [
    "main",
    "build_parser",
    "_build_parser",
    "_format_table",
    "_print",
    "_render",
    "_supports_color",
    "_wrap",
    "_cmd_benchmark",
    "_cmd_calc",
    "_cmd_constants",
    "_cmd_dashboard",
    "_cmd_hypothesis",
    "_cmd_info",
    "_cmd_integrate",
    "_cmd_modules",
    "_cmd_plot",
    "_cmd_prompt",
    "_cmd_sample",
    "_cmd_stats",
    "_cmd_version",
]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    When ``argv`` is ``None`` (the normal ``cds`` invocation) it reads
    :data:`sys.argv`; tests pass an explicit list so no subprocess is needed.

    argparse raises :class:`SystemExit` for ``--help`` and usage errors. We
    catch it here and surface its code as the return value so callers (tests
    and ``__main__``) never see an exception — only an integer exit code.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0

    if args.version:
        from cds import __version__

        _print(_render(f"[bold]System[/] version [cyan]{__version__}[/]"))
        return 0

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
