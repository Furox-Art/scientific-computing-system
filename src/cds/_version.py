# Static version source. Kept in lockstep with `version` in pyproject.toml and
# version metadata in CITATION.cff. Package-affecting changes require a
# synchronized monotonic bump; scripts/check_version_discipline.py enforces it.
from __future__ import annotations

__all__ = ["__version__", "version", "__version_tuple__", "version_tuple"]

__version__ = version = "2.1.0"
__version_tuple__ = version_tuple = (2, 1, 0)
