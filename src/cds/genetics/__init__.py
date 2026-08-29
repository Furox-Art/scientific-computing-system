"""Genetics module — DNA sequence analysis.

Re-exports the pure-Python helpers from
:mod:`cds.genetics.sequence`: GC content, k-mers, reverse complement and
Needleman-Wunsch global alignment.
"""

from __future__ import annotations

from cds.genetics.sequence import (
    gc_content,
    k_mers,
    needleman_wunsch,
    reverse_complement,
)

__all__ = [
    "gc_content",
    "k_mers",
    "needleman_wunsch",
    "reverse_complement",
]
