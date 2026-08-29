"""DNA sequence analysis — composition, k-mers, reverse complement and global alignment.

Pure-Python, zero-dependency implementation mirroring the ``cds2.genetics``
API from the v2 repo. All helpers validate against the ``ACGTN`` alphabet,
fold lowercase input to uppercase and treat ``N`` as an ambiguous base where
appropriate (excluded from GC denominator, preserved through complement).

References:
    Needleman, S. B. & Wunsch, C. D. (1970). A general method applicable to
    the search for similarities in the amino acid sequence of two proteins.
    Journal of Molecular Biology, 48(3), 443-453.
"""

from __future__ import annotations

__all__ = [
    "gc_content",
    "k_mers",
    "needleman_wunsch",
    "reverse_complement",
]

_VALID_NUCLEOTIDES: frozenset[str] = frozenset("ACGTN")
_COMPLEMENT: dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def _clean(seq: str) -> str:
    """Uppercase ``seq`` and reject any character outside ``ACGTN``.

    Args:
        seq: raw DNA string (case-insensitive).

    Returns:
        Uppercased sequence consisting only of ``A``, ``C``, ``G``, ``T``, ``N``.

    Raises:
        ValueError: if any character is not a valid nucleotide.
    """
    sequence = seq.upper()
    for ch in sequence:
        if ch not in _VALID_NUCLEOTIDES:
            msg = f"invalid nucleotide {ch!r}"
            raise ValueError(msg)
    return sequence


def gc_content(seq: str) -> float:
    """Fraction of ``G``/``C`` bases among non-``N`` bases.

    ``N`` (ambiguous) positions are excluded from both numerator and
    denominator, matching the v2 ``cds2.genetics.gc_content`` behaviour.
    Returns ``0.0`` when ``seq`` is empty or consists solely of ``N``.

    Args:
        seq: DNA sequence (case-insensitive, ``ACGTN`` alphabet).

    Returns:
        GC fraction in ``[0.0, 1.0]``.

    Raises:
        ValueError: if ``seq`` contains a character outside ``ACGTN``.
    """
    sequence = _clean(seq)
    counted = [ch for ch in sequence if ch != "N"]
    if not counted:
        return 0.0
    gc = sum(1 for ch in counted if ch in "GC")
    return gc / len(counted)


def reverse_complement(seq: str) -> str:
    """Reverse complement of a DNA sequence.

    Complements ``A<->T``, ``C<->G`` and preserves ``N``; the result is
    returned in uppercase regardless of input casing.

    Args:
        seq: DNA sequence (case-insensitive, ``ACGTN`` alphabet).

    Returns:
        Reverse-complemented sequence.

    Raises:
        ValueError: if ``seq`` contains a character outside ``ACGTN``.
    """
    return "".join(_COMPLEMENT[ch] for ch in reversed(_clean(seq)))


def k_mers(seq: str, k: int) -> list[str]:
    """All length-``k`` sliding-window substrings (k-mers).

    Overlapping windows are returned in order, including duplicates, so
    ``k_mers("ATATA", 2)`` yields ``["AT", "TA", "AT", "TA"]``. This matches
    the natural list view of ``kmer_counts`` from v2.

    Args:
        seq: DNA sequence (case-insensitive, ``ACGTN`` alphabet).
        k: k-mer length, must be at least 1.

    Returns:
        List of ``len(seq) - k + 1`` k-mers, or ``[]`` when ``k`` exceeds
        the sequence length (also ``[]`` for an empty ``seq``).

    Raises:
        ValueError: if ``k < 1`` or ``seq`` contains an invalid nucleotide.
    """
    if k < 1:
        msg = "k must be at least 1"
        raise ValueError(msg)
    sequence = _clean(seq)
    if k > len(sequence):
        return []
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


def needleman_wunsch(
    s1: str,
    s2: str,
    match: int = 1,
    mismatch: int = -1,
    gap: int = -1,
) -> tuple[int, str, str]:
    """Needleman-Wunsch global alignment.

    Full dynamic programming with affine-free scoring. The traceback prefers
    the diagonal (substitution/match) move when scores tie, reproducing the
    deterministic alignment from ``cds2.genetics.global_align``.

    Args:
        s1: first DNA sequence (case-insensitive, ``ACGTN`` alphabet).
        s2: second DNA sequence (case-insensitive, ``ACGTN`` alphabet).
        match: score for a matching column.
        mismatch: score for a mismatching column.
        gap: penalty for a gap (inserted as ``'-'``).

    Returns:
        Tuple ``(score, aligned_s1, aligned_s2)`` where ``score`` is the
        optimal global alignment score and the two aligned strings have equal
        length, padded with ``'-'`` for gaps. Empty inputs yield ``(0, "", "")``
        or a pure-gap alignment when only one sequence is empty.

    Raises:
        ValueError: if either sequence contains a character outside ``ACGTN``.
    """
    seq1 = _clean(s1)
    seq2 = _clean(s2)
    rows = len(seq1)
    cols = len(seq2)
    dp: list[list[int]] = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        dp[i][0] = i * gap
    for j in range(1, cols + 1):
        dp[0][j] = j * gap
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            sub = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + sub,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap,
            )
    aligned_a: list[str] = []
    aligned_b: list[str] = []
    i, j = rows, cols
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub = match if seq1[i - 1] == seq2[j - 1] else mismatch
            if dp[i][j] == dp[i - 1][j - 1] + sub:
                aligned_a.append(seq1[i - 1])
                aligned_b.append(seq2[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + gap:
            aligned_a.append(seq1[i - 1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq2[j - 1])
            j -= 1
    # Empty-input fast path: when both strings are empty the loop never runs
    # and we return the zero-score empty alignment assembled above.
    if rows == 0 and cols == 0:
        return (0, "", "")
    if rows == 0:
        # Only s2 contributed; aligned_a is all gaps already built in loop,
        # but the loop for rows==0 would have built them correctly; however
        # to avoid an extra leading gap artifact from the else-branch when
        # j decrements, reconstruct explicitly for clarity.
        # The loop above already handles this, but we keep the explicit
        # branch for mypy exhaustiveness and readability.
        pass
    aligned_s1 = "".join(reversed(aligned_a))
    aligned_s2 = "".join(reversed(aligned_b))
    # For the case where one input was empty the loop produced a single
    # string of gaps vs the other sequence; dp[rows][cols] holds the correct
    # score (gap * length). Return as-is.
    return (dp[rows][cols], aligned_s1, aligned_s2)
