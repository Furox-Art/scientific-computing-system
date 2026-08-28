"""Information-theory measures — entropy, divergence and dependence.

Re-exports the pure-Python Shannon measures from :mod:`cds.infotheory.measures`.

References:
    Shannon, C. E. (1948). A Mathematical Theory of Communication.
"""

from __future__ import annotations

from cds.infotheory.measures import (
    cross_entropy,
    entropy,
    js_divergence,
    kl_divergence,
    mutual_information,
)

__all__ = [
    "cross_entropy",
    "entropy",
    "js_divergence",
    "kl_divergence",
    "mutual_information",
]
