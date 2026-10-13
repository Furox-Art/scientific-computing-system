"""The CLI's exit-code contract.

Every handler returns one of the codes below. They were previously bare ``0``
and ``1`` literals scattered across ``cds.cli._handlers`` with nothing that
defined what the numbers meant, so a caller could not distinguish "bad numbers"
from "the dashboard could not be found" from "streamlit is not installed" —
they were all ``1``. Naming them here and routing every ``return`` through
them makes the mapping a single source of truth that a test can walk.

The values are deliberately *not* new. ``EXIT_ERROR`` is the ``1`` every
handler already returned and that the suite already asserts, and
``EXIT_USAGE`` is the code :mod:`argparse` has always raised for a malformed
command line. This module pins a contract that already existed in practice;
it does not invent one.
"""

from __future__ import annotations

__all__ = [
    "EXIT_CODES",
    "EXIT_ERROR",
    "EXIT_MISSING_DEPENDENCY",
    "EXIT_OK",
    "EXIT_USAGE",
    "describe",
    "is_handler_code",
]

#: The command completed and produced its output.
EXIT_OK = 0

#: The command was understood but could not be carried out: a malformed value,
#: an out-of-domain parameter, an unwritable output path, or a viral resource
#: (the dashboard script, an output file) that is not present.
EXIT_ERROR = 1

#: The command line itself was rejected — an unknown subcommand, a missing
#: required argument, a bad ``choices=`` value, or an unparseable number.
#: This is :mod:`argparse`'s own usage-error code.
EXIT_USAGE = 2

#: The command needs an optional extra that is not installed (``[plot]`` for
#: ``--file``, ``streamlit`` for the dashboard). Distinguished from a plain
#: input error because the remedy is an install, not a corrected invocation.
EXIT_MISSING_DEPENDENCY = 3

#: Every code the CLI can exit with, mapped to its one-line contract.
EXIT_CODES: dict[int, str] = {
    EXIT_OK: "command completed and produced its output",
    EXIT_ERROR: "command was understood but could not be carried out",
    EXIT_USAGE: "command line was rejected by the argument parser",
    EXIT_MISSING_DEPENDENCY: "command needs an optional extra that is not installed",
}

#: The set of codes a subcommand handler may return. ``EXIT_USAGE`` is produced
#: by :func:`argparse.ArgumentParser.parse_args` before any handler runs, so it
#: is not in a handler's reach and is listed for documentation only.
_HANDLER_CODES = frozenset(
    {EXIT_OK, EXIT_ERROR, EXIT_MISSING_DEPENDENCY}
)


def describe(code: int) -> str:
    """Return the one-line contract for ``code``.

    Args:
        code: an integer exit status.

    Returns:
        The documented meaning, or a message naming the documented codes when
        ``code`` is not one of them.

    Raises:
        TypeError: if ``code`` is not an integer.
    """
    if isinstance(code, bool) or not isinstance(code, int):
        raise TypeError("exit code must be an integer")
    return EXIT_CODES.get(code, f"undocumented exit code {code}")


def is_handler_code(code: int) -> bool:
    """True when a subcommand handler is allowed to return ``code``."""
    return code in _HANDLER_CODES
