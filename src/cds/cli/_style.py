"""ANSI colour helpers and ASCII table rendering for the CDS CLI."""

from __future__ import annotations

import sys

_RESET = "\033[0m"
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
    """Print helper routed through stdout (so tests can capture via capsys)."""
    print(*args)


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
