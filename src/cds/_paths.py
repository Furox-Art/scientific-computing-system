"""Path confinement for caller-supplied filesystem locations.

Provenance APIs record *where* a run read its inputs from. When that location
is derived from something the caller did not author -- a manifest field, a
checkpoint name, a path forwarded by a host application -- a value such as
``../../../../etc/shadow`` must not silently escape the directory the caller
intended to confine it to.

The confinement rule is deliberately narrow: resolve first, then compare. The
comparison runs on fully resolved :class:`~pathlib.Path` objects rather than
string prefixes, so three classic bypasses are closed by construction:

* ``..`` segments are collapsed by :meth:`Path.resolve` *before* the test, so a
  traversal that lands back inside the root is accepted while one that lands
  outside is rejected.
* A sibling directory that merely shares a textual prefix is rejected.
  ``root=/srv/data`` does not admit ``/srv/data-backup``; a ``str.startswith``
  check would have admitted it.
* Symbolic links are resolved, so a link inside the root that points outside it
  is rejected on its real target.

This is opt-in infrastructure: callers that already trust their own paths keep
their current behaviour by not supplying a root. The library never imposes a
sandbox on library-internal paths, because CDS is a local-first library rather
than a service accepting untrusted remote input.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_within_root(
    path: str | os.PathLike[str],
    root: str | os.PathLike[str],
) -> Path:
    """Resolve ``path`` and refuse any location that escapes ``root``.

    A relative ``path`` is interpreted against ``root``. An absolute ``path`` is
    resolved on its own and must already land inside ``root``.

    Args:
        path: The location to resolve. May be relative or absolute.
        root: The directory the resolved location must stay within.

    Returns:
        The fully resolved, confined absolute path.

    Raises:
        ValueError: If the resolved location is not inside ``root``.
    """
    resolved_root = Path(root).resolve()
    candidate = Path(path)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (resolved_root / candidate).resolve()
    )
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"path {resolved} escapes confined root {resolved_root}")
    return resolved


def resolve_target(
    path: str | os.PathLike[str],
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve ``path``, applying confinement only when ``root`` is supplied.

    This is the opt-in adapter used by the public provenance APIs. Callers that
    pass no root keep the library's historical behaviour exactly.

    Args:
        path: The location to resolve.
        root: Optional confinement directory. ``None`` disables confinement.

    Returns:
        A :class:`~pathlib.Path` for ``path``.

    Raises:
        ValueError: If ``root`` is given and the resolved location escapes it.
    """
    if root is None:
        return Path(path)
    return resolve_within_root(path, root)
