"""Tests for cds.genetics — GC content, k-mers, reverse complement, alignment."""

from __future__ import annotations

import pytest

from cds.genetics import (
    gc_content,
    k_mers,
    needleman_wunsch,
    reverse_complement,
)
from cds.genetics.sequence import gc_content as gc_seq
from cds.genetics.sequence import k_mers as km_seq


def test_gc_content_basic() -> None:
    assert gc_content("GGCC") == pytest.approx(1.0)
    assert gc_content("AAAA") == pytest.approx(0.0)
    assert gc_content("GCAT") == pytest.approx(0.5)
    # via submodule import
    assert gc_seq("GCAT") == pytest.approx(0.5)


def test_gc_content_lowercase_and_n() -> None:
    assert gc_content("gcat") == pytest.approx(0.5)
    assert gc_content("NGCN") == pytest.approx(1.0)
    assert gc_content("NGNT") == pytest.approx(0.5)
    assert gc_content("NNAT") == pytest.approx(0.0)


def test_gc_content_empty_and_all_n() -> None:
    assert gc_content("") == pytest.approx(0.0)
    assert gc_content("NNNN") == pytest.approx(0.0)
    assert gc_content("n") == pytest.approx(0.0)


def test_gc_content_invalid() -> None:
    with pytest.raises(ValueError, match="invalid nucleotide"):
        gc_content("ACXT")
    with pytest.raises(ValueError, match="invalid nucleotide"):
        gc_content("B")


def test_reverse_complement_basic() -> None:
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("AAAA") == "TTTT"
    assert reverse_complement("CCCC") == "GGGG"


def test_reverse_complement_n_and_case() -> None:
    assert reverse_complement("atGCn") == "NGCAT"
    assert reverse_complement("N") == "N"
    assert reverse_complement("") == ""


def test_reverse_complement_double_involution() -> None:
    for seq in ("ACGTN", "TTTT", "GANTC", "ATGCATGC"):
        assert reverse_complement(reverse_complement(seq)) == seq.upper()


def test_reverse_complement_invalid() -> None:
    with pytest.raises(ValueError, match="invalid nucleotide"):
        reverse_complement("AXT")


def test_k_mers_basic() -> None:
    assert k_mers("ATATA", 2) == ["AT", "TA", "AT", "TA"]
    assert k_mers("AAAA", 1) == ["A", "A", "A", "A"]
    assert k_mers("ATGC", 4) == ["ATGC"]
    assert km_seq("ATGC", 4) == ["ATGC"]


def test_k_mers_case_and_n() -> None:
    assert k_mers("atata", 2) == ["AT", "TA", "AT", "TA"]
    assert k_mers("ACGN", 2) == ["AC", "CG", "GN"]


def test_k_mers_edge_cases() -> None:
    assert k_mers("ATG", 5) == []
    assert k_mers("", 1) == []
    assert k_mers("ATGC", 1) == ["A", "T", "G", "C"]


def test_k_mers_invalid_k() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        k_mers("ATG", 0)
    with pytest.raises(ValueError, match="at least 1"):
        k_mers("ATG", -1)


def test_k_mers_invalid_nucleotide() -> None:
    with pytest.raises(ValueError, match="invalid nucleotide"):
        k_mers("ATBX", 2)


def test_needleman_wunsch_identical() -> None:
    score, a, b = needleman_wunsch("GATTACA", "GATTACA")
    assert score == 7
    assert a == "GATTACA"
    assert b == "GATTACA"


def test_needleman_wunsch_gap() -> None:
    score, a, b = needleman_wunsch("CAT", "CT")
    assert score == 1
    assert a == "CAT"
    assert b == "C-T"
    score2, a2, b2 = needleman_wunsch("CT", "CAT")
    assert score2 == 1
    assert a2 == "C-T"
    assert b2 == "CAT"


def test_needleman_wunsch_tie_prefers_diagonal() -> None:
    # When diagonal and gap tie, diagonal should be preferred;
    # for "A" vs "AA" the optimal alignment is "-A"/"AA" not "A-"/"AA"
    s, a, c = needleman_wunsch("A", "AA")
    assert (a, c) == ("-A", "AA")
    s2, a2, c2 = needleman_wunsch("AA", "A")
    assert (a2, c2) == ("AA", "-A")
    assert s == 0
    assert s2 == 0


def test_needleman_wunsch_mismatch_and_custom_scores() -> None:
    score, a, b = needleman_wunsch("GT", "AC")
    assert score == -2
    assert a == "GT"
    assert b == "AC"
    score2, a2, b2 = needleman_wunsch("ACG", "AGG", match=2, mismatch=-2, gap=-2)
    assert score2 == 2
    assert a2 == "ACG"
    assert b2 == "AGG"


def test_needleman_wunsch_empty_and_case() -> None:
    # empty vs empty
    s, a, b = needleman_wunsch("", "")
    assert s == 0
    assert a == ""
    assert b == ""
    # empty vs non-empty
    s2, a2, b2 = needleman_wunsch("", "ACG")
    assert s2 == -3
    assert a2 == "---"
    assert b2 == "ACG"
    # case folding
    s3, a3, b3 = needleman_wunsch("acg", "ACG")
    assert s3 == 3
    assert a3 == "ACG"
    assert b3 == "ACG"


def test_needleman_wunsch_invalid() -> None:
    with pytest.raises(ValueError, match="invalid nucleotide"):
        needleman_wunsch("AXT", "ACG")
    with pytest.raises(ValueError, match="invalid nucleotide"):
        needleman_wunsch("ACG", "BX")
