# Static version source. Kept in lockstep with `version` in `pyproject.toml`.
# Bump both for a release; merging the version bump to main triggers the
# verified GitHub + PyPI publish workflow. See `pyproject.toml` for the release
# checklist. This file is committed (not generated) so mypy has a concrete
# `__version__: str` to resolve in fresh checkouts without git history.
from __future__ import annotations

__all__ = ["__version__", "version", "__version_tuple__", "version_tuple"]

__version__ = version = "1.7.1"
__version_tuple__ = version_tuple = (1, 7, 1)
