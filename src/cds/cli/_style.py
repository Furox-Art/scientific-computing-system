"""ANSI colour helpers and ASCII table rendering for the CDS CLI."""

from __future__ import annotations

import io
import sys

_RESET = "\033[0m"

# Characters the CLI emits that legacy 8-bit consoles (Windows cp125x /
# cp1254, some POSIX C locales) cannot encode. Used only as a last-resort
# fallback when stdout refuses UTF-8 — see :func:`_safe_text`.
_ASCII_FALLBACKS: dict[str, str] = {
    "\u03c0": "pi",  # π
    "\u2014": "-",  # em dash
    "\u2013": "-",  # en dash
    "\u2192": "->",  # →
    "\u2248": "~=",  # ≈
    "\u00b1": "+/-",  # ±
    "\u00b3": "^3",  # ³
    "\u00b2": "^2",  # ²
    "\u2264": "<=",  # ≤
    "\u2265": ">=",  # ≥
    "\u00d7": "x",  # ×
    "\u2026": "...",  # …
    "\u2019": "'",  # ’
    "\u201c": '"',  # “
    "\u201d": '"',  # ”
    "\u00b5": "u",  # µ
    "\u0394": "delta",  # Δ
    "\u03bb": "lambda",  # λ
    "\u03c3": "sigma",  # σ
    "\u03bc": "mu",  # μ
    "\u2713": "OK",  # ✓
    "\u2717": "x",  # ✗
}


def _enable_utf8_stdout() -> None:
    """Force stdout/stderr to UTF-8 so non-ASCII output cannot crash the CLI.

    Windows consoles default to a legacy ANSI codepage (cp1254 on Turkish
    systems), which cannot encode characters the CLI prints routinely — ``π``
    in the Monte Carlo tables, em dashes in module descriptions, ``O(N³)`` in
    the info banner. Printing those raised ``UnicodeEncodeError`` and killed
    the process mid-table.

    ``TextIOWrapper.reconfigure`` is available on Python 3.7+, so this is the
    cheapest fix that keeps the real Unicode glyphs when the terminal can show
    them. ``errors="backslashreplace"`` guarantees a write can never raise even
    if the target somehow rejects a codepoint. Wrapped defensively: under
    pytest's ``capsys`` (and other captured-output shims) the streams may not
    be ``TextIOWrapper`` at all, in which case we leave them untouched and rely
    on :func:`_safe_text`.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # pragma: no cover
            continue  # pragma: no cover
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError, io.UnsupportedOperation):  # pragma: no cover
            # Detached, closed, or non-reconfigurable stream — _safe_text covers us.
            continue


_enable_utf8_stdout()


def _safe_text(text: str) -> str:
    """Downgrade ``text`` to ASCII-safe glyphs if stdout cannot encode it.

    Second line of defence behind :func:`_enable_utf8_stdout`: when stdout is
    a captured buffer or a stream we could not reconfigure, we transliterate
    the handful of non-ASCII characters the CLI actually uses (``π`` -> ``pi``,
    ``—`` -> ``-``) rather than raising. Text that already encodes cleanly is
    returned unchanged, so UTF-8 terminals keep the real glyphs.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):  # pragma: no cover
        pass  # pragma: no cover
    else:
        return text
    for char, replacement in _ASCII_FALLBACKS.items():  # pragma: no cover
        text = text.replace(char, replacement)  # pragma: no cover
    # Anything still unencodable (unexpected glyph) degrades to "?" instead of
    # taking the process down.
    return text.encode(encoding, errors="replace").decode(  # pragma: no cover
        encoding, errors="replace"
    )  # pragma: no cover


_STYLES: dict[str, str] = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "bold green": "\033[1;32m",
    "bold blue": "\033[1;34m",
    "bold cyan": "\033[1;36m",
    "bold magenta": "\033[1;35m",
    "bold red": "\033[1;31m",
    "bold yellow": "\033[1;33m",
}


def _supports_color() -> bool:
    """Return True when stdout looks like an interactive colour terminal."""
    return sys.stdout.isatty()


def _wrap(style: str, text: str) -> str:
    """Wrap ``text`` in the ANSI codes for ``style`` if stdout is a TTY."""
    if not _supports_color():
        return text
    code = _STYLES.get(style, "")
    if not code:
        return text
    return f"{code}{text}{_RESET}"


def _print(*args: object) -> None:
    """Print helper routed through stdout (so tests can capture via capsys).

    Every argument passes through :func:`_safe_text` so a legacy-codepage
    console can never turn a table cell into a ``UnicodeEncodeError``.
    """
    print(*(_safe_text(str(arg)) for arg in args))


def _render(markup: str) -> str:
    """Render rich-style ``[style]...[/]`` markup into ANSI-coloured text.

    Only the tag shapes actually used by this CLI are supported:
    ``[bold]``, ``[italic]``, ``[dim]``, ``[red]``, ``[green]``, ``[yellow]``,
    ``[blue]``, ``[cyan]``, ``[magenta]`` and a few combined ``[bold green]``
    variants. Unknown tags are stripped to their inner text. Nesting is not
    supported (the CLI never nests them).
    """
    out: list[str] = []
    i = 0
    n = len(markup)
    while i < n:
        open_idx = markup.find("[", i)
        if open_idx == -1:
            out.append(markup[i:])
            break
        out.append(markup[i:open_idx])
        close_idx = markup.find("]", open_idx)
        if close_idx == -1:
            out.append(markup[open_idx:])
            break
        tag = markup[open_idx + 1 : close_idx]
        # Closing tag [/] ends the current styled run.
        if tag.startswith("/"):
            out.append(_RESET if _supports_color() else "")
            i = close_idx + 1
            continue
        # Combined forms like "bold green" are looked up directly.
        code = _STYLES.get(tag)
        if code is None and " " in tag:
            # ``[bold green]`` style — already in the table; fall back to first word.
            code = _STYLES.get(tag.split()[0], "")
        out.append(code if (_supports_color() and code is not None) else "")
        i = close_idx + 1
    return "".join(out)


def _format_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    """Render ``headers``/``rows`` as a bordered ASCII table with a title.

    A tiny reimplementation of the ``rich.Table`` output the CLI used to
    produce: top/bottom title rule, a header row underlined with ``-``, and
    each data row on its own line. Columns are sized to the widest cell.
    """
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for c, cell in enumerate(row[:cols]):
            widths[c] = max(widths[c], len(cell))

    def _border(left: str, fill: str, right: str) -> str:
        return left + fill + right

    sep = _border("+", "+".join("-" * (w + 2) for w in widths), "+")

    lines: list[str] = []
    if title:
        lines.append(_wrap("bold", title))
    lines.append(sep)
    header_cells = " | ".join(h.ljust(widths[c]) for c, h in enumerate(headers))
    lines.append(f"| {header_cells} |")
    lines.append(sep)
    for row in rows:
        cells = " | ".join(
            str(row[c]).ljust(widths[c]) if c < len(row) else "".ljust(widths[c])
            for c in range(cols)
        )
        lines.append(f"| {cells} |")
    lines.append(sep)
    return "\n".join(lines)
