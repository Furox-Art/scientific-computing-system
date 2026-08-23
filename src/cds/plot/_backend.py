"""Lazy matplotlib import for the optional ``cds[plot]`` extra."""

from __future__ import annotations

import importlib
import os
from typing import Any

_MATPLOTLIB_INSTALL_HINT = (
    "matplotlib is an optional dependency. Install it with "
    "`pip install scientific-computing-system[plot]` "
    "(or `pip install matplotlib`)."
)


def require_matplotlib() -> Any:
    """Import and return ``matplotlib.pyplot``, with a clear install hint.

    Selects the non-interactive Agg backend when the user has not set
    ``MPLBACKEND``, so headless CI and servers can still build figures.

    Uses :func:`importlib.import_module` so mypy stays clean both with and
    without matplotlib installed (no stubs / no unused ``type: ignore``).
    """
    try:
        matplotlib = importlib.import_module("matplotlib")
        if os.environ.get("MPLBACKEND") is None:
            use_backend = getattr(matplotlib, "use")
            use_backend("Agg")
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_MATPLOTLIB_INSTALL_HINT) from exc
