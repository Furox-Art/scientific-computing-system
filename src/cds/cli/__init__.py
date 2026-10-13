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
- :mod:`cds.cli._exit_codes` — the named exit-code contract

The entry point :func:`main` accepts an optional ``argv`` so tests can drive a
specific command without spawning a subprocess, and returns the integer exit
code instead of calling :func:`sys.exit` directly when ``argv`` is given.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from cds.cli._exit_codes import (
    EXIT_CODES,
    EXIT_ERROR,
    EXIT_MISSING_DEPENDENCY,
    EXIT_OK,
    EXIT_USAGE,
    describe,
    is_handler_code,
)
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
from cds.cli._parser import build_parser
from cds.cli._style import (
    _format_table,
    _print,
    _render,
    _supports_color,
    _wrap,
)
from cds.cli._system_info import _cmd_info, _cmd_modules

# Historical private name kept as an alias for backwards compatibility.
#
# This lives here rather than in `cds.cli._parser` because that is the module
# that exports it: `cds.cli._build_parser` is in `__all__` below and is part of
# the CLI's published surface. An alias defined in `_parser` was never referenced
# from that module, which static analysis correctly reports as an unused global.
# Defining it beside its only consumer keeps the back-compat name working while
# giving the assignment an actual purpose in its own module.
_build_parser = build_parser

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
    # The documented exit-code contract, re-exported so a caller can map a
    # status without reaching into a private module.
    "EXIT_CODES",
    "EXIT_OK",
    "EXIT_ERROR",
    "EXIT_USAGE",
    "EXIT_MISSING_DEPENDENCY",
    "describe",
    "is_handler_code",
]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    When ``argv`` is ``None`` (the normal ``cds`` invocation) it reads
    :data:`sys.argv`; tests pass an explicit list so no subprocess is needed.

    argparse raises :class:`SystemExit` for ``--help`` and usage errors. We
    catch it here and surface its code as the return value so callers (tests
    and ``__main__``) never see an exception — only an integer exit code.

    Every code this function can return is documented in
    :mod:`cds.cli._exit_codes`: ``0`` for a successful run or bare ``--help``,
    ``2`` (``EXIT_USAGE``) for a command line argparse rejected, and whatever
    the subcommand handler chose otherwise. ``tests/test_cli_exit_codes.py``
    walks the whole mapping.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # ``--help`` exits 0; a rejected command line exits EXIT_USAGE (2),
        # which is argparse's own usage-error code. A non-integer code (a
        # string message or None) is treated as success, which is what
        # ``sys.exit()`` with no argument does.
        if exc.code is None or isinstance(exc.code, int) and exc.code == EXIT_OK:
            return EXIT_OK
        return int(exc.code) if isinstance(exc.code, int) else EXIT_OK

    if args.version:
        from cds import __version__

        _print(_render(f"[bold]System[/] version [cyan]{__version__}[/]"))
        return EXIT_OK

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return EXIT_OK
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
