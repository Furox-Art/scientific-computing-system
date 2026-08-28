"""Tests for cds.infotheory — entropy, divergence and mutual information.

Covers Shannon 1948 measures and validates error handling, symmetry,
non-negativity and independence properties.
"""

from __future__ import annotations

import math

import pytest

from cds.infotheory import (
    cross_entropy,
    entropy,
    js_divergence,
    kl_divergence,
    mutual_information,
)


def test_entropy_uniform() -> None:
    """Uniform distribution over 4 symbols has entropy 2 bits (base 2)."""
    assert entropy([0.25, 0.25, 0.25, 0.25]) == pytest.approx(2.0)


def test_entropy_deterministic() -> None:
    """Deterministic distribution has zero entropy."""
    assert entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert entropy([0.0, 1.0]) == pytest.approx(0.0)


def test_entropy_binary_uniform() -> None:
    """Fair coin has 1 bit of entropy."""
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)


def test_entropy_nats() -> None:
    """Base e gives entropy in nats: H([0.5,0.5]) = ln 2."""
    assert entropy([0.5, 0.5], base=math.e) == pytest.approx(math.log(2))


def test_entropy_base_10() -> None:
    """Base 10 works via change-of-base."""
    # H_10([0.5,0.5]) = log10 2
    assert entropy([0.5, 0.5], base=10.0) == pytest.approx(math.log(2) / math.log(10))


def test_entropy_ignores_zero() -> None:
    """Zero-probability entries contribute 0 (0 log 0 := 0)."""
    assert entropy([0.5, 0.5, 0.0]) == pytest.approx(1.0)
    assert entropy([0.5, 0.25, 0.25, 0.0]) == pytest.approx(1.5)


def test_entropy_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        entropy([])


def test_entropy_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        entropy([-0.1, 1.1])
    with pytest.raises(ValueError, match="non-negative"):
        entropy([0.5, -0.2, 0.7])


def test_entropy_rejects_unnormalized() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        entropy([0.5, 0.6])
    with pytest.raises(ValueError, match="sum to 1"):
        entropy([0.2, 0.2])
    with pytest.raises(ValueError, match="sum to 1"):
        entropy([0.3, 0.3, 0.3])


def test_entropy_rejects_bad_base() -> None:
    with pytest.raises(ValueError, match="base"):
        entropy([0.5, 0.5], base=1.0)
    with pytest.raises(ValueError, match="base"):
        entropy([0.5, 0.5], base=0.0)
    with pytest.raises(ValueError, match="base"):
        entropy([0.5, 0.5], base=-2.0)
    with pytest.raises(ValueError, match="base"):
        entropy([0.5, 0.5], base=1.0000000001)  # isclose to 1


def test_entropy_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        entropy([float("inf"), -float("inf")])  # not finite
    with pytest.raises(ValueError, match="non-negative"):
        entropy([float("nan"), 1.0])


def test_kl_self_is_zero() -> None:
    p = [0.3, 0.7]
    assert kl_divergence(p, p) == pytest.approx(0.0)
    assert kl_divergence([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_kl_non_negativity() -> None:
    """KL divergence is always >=0 (Gibbs inequality)."""
    assert kl_divergence([0.9, 0.1], [0.5, 0.5]) >= 0
    assert kl_divergence([0.5, 0.5], [0.9, 0.1]) >= 0
    assert kl_divergence([0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0]) == pytest.approx(
        float("inf")
    )


def test_kl_base_e() -> None:
    assert kl_divergence([0.5, 0.5], [0.9, 0.1], base=math.e) > 0


def test_kl_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        kl_divergence([1.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="same length"):
        kl_divergence([0.5, 0.5], [1.0])


def test_kl_rejects_bad_distribution() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        kl_divergence([0.6, 0.6], [0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        kl_divergence([-0.1, 1.1], [0.5, 0.5])
    with pytest.raises(ValueError, match="empty"):
        kl_divergence([], [0.5, 0.5])


def test_kl_rejects_bad_base() -> None:
    with pytest.raises(ValueError, match="base"):
        kl_divergence([0.5, 0.5], [0.5, 0.5], base=1.0)


def test_kl_infinite_when_q_zero() -> None:
    """KL is infinite when q has zero where p is positive."""
    assert math.isinf(kl_divergence([0.5, 0.5], [1.0, 0.0]))
    assert math.isinf(kl_divergence([0.3, 0.7], [0.0, 1.0]))


def test_kl_zero_p_ignores_q_zero() -> None:
    """If p_i==0, q_i may be 0 without causing infinity."""
    # p has 0 where q has 0 => term skipped, finite
    assert kl_divergence([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    # p=0, q=0, other mass uniform
    assert kl_divergence([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]) == pytest.approx(0.0)


def test_cross_entropy_identity() -> None:
    p = [0.4, 0.6]
    q = [0.5, 0.5]
    expected = entropy(p) + kl_divergence(p, q)
    assert cross_entropy(p, q) == pytest.approx(expected)
    # cross-entropy >= entropy
    assert cross_entropy(p, q) >= entropy(p)


def test_cross_entropy_self_is_entropy() -> None:
    p = [0.25, 0.25, 0.25, 0.25]
    assert cross_entropy(p, p) == pytest.approx(entropy(p))


def test_cross_entropy_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        cross_entropy([1.0], [0.5, 0.5])


def test_cross_entropy_infinite() -> None:
    assert math.isinf(cross_entropy([0.5, 0.5], [1.0, 0.0]))


def test_cross_entropy_zero_p_ignores() -> None:
    assert cross_entropy([1.0, 0.0], [0.5, 0.5]) == pytest.approx(1.0)


def test_cross_entropy_rejects_bad_base() -> None:
    with pytest.raises(ValueError, match="base"):
        cross_entropy([0.5, 0.5], [0.5, 0.5], base=1.0)


def test_js_symmetry() -> None:
    p = [0.9, 0.1]
    q = [0.1, 0.9]
    a = js_divergence(p, q)
    b = js_divergence(q, p)
    assert a == pytest.approx(b)


def test_js_bounded() -> None:
    """JS divergence is bounded in [0, 1] for base 2."""
    assert js_divergence([0.9, 0.1], [0.1, 0.9]) <= 1.0 + 1e-12
    assert js_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert js_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_js_identical_zero() -> None:
    assert js_divergence([0.3, 0.7], [0.3, 0.7]) == pytest.approx(0.0, abs=1e-12)


def test_js_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        js_divergence([1.0], [0.5, 0.5])


def test_js_rejects_bad_base() -> None:
    with pytest.raises(ValueError, match="base"):
        js_divergence([0.5, 0.5], [0.5, 0.5], base=0.0)


def test_js_non_negative() -> None:
    assert js_divergence([0.2, 0.8], [0.5, 0.5]) >= 0


def test_mi_independent_zero() -> None:
    """Independent variables have zero mutual information."""
    joint = [[0.25, 0.25], [0.25, 0.25]]
    assert mutual_information(joint) == pytest.approx(0.0, abs=1e-12)


def test_mi_identical_one_bit() -> None:
    """Perfectly correlated bits have MI = 1 bit."""
    joint = [[0.5, 0.0], [0.0, 0.5]]
    assert mutual_information(joint) == pytest.approx(1.0)
    # Anti-correlated also 1 bit
    joint2 = [[0.0, 0.5], [0.5, 0.0]]
    assert mutual_information(joint2) == pytest.approx(1.0)


def test_mi_partial_correlation() -> None:
    joint = [[0.4, 0.1], [0.1, 0.4]]
    mi = mutual_information(joint)
    assert 0 < mi < 1


def test_mi_rejects_unnormalized_joint() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        mutual_information([[0.7, 0.7], [0.7, 0.7]])
    with pytest.raises(ValueError, match="sum to 1"):
        mutual_information([[0.5, 0.5], [0.5, 0.5]])  # sums to 2


def test_mi_rejects_ragged() -> None:
    with pytest.raises(ValueError, match="2-D"):
        mutual_information([[0.5, 0.5], [0.5]])
    with pytest.raises(ValueError, match="2-D"):
        mutual_information([[0.5, 0.5, 0.0], [0.5, 0.5]])


def test_mi_rejects_empty() -> None:
    with pytest.raises(ValueError, match="2-D"):
        mutual_information([])
    with pytest.raises(ValueError, match="2-D"):
        mutual_information([[]])


def test_mi_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mutual_information([[-0.1, 1.1], [0.0, 0.0]])


def test_mi_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        mutual_information([[float("inf"), 0.5], [0.5, 0.0]])
    with pytest.raises(ValueError, match="non-negative"):
        mutual_information([[float("nan"), 0.5], [0.5, 0.0]])


def test_mi_rejects_bad_base() -> None:
    with pytest.raises(ValueError, match="base"):
        mutual_information([[0.25, 0.25], [0.25, 0.25]], base=1.0)
    with pytest.raises(ValueError, match="base"):
        mutual_information([[0.25, 0.25], [0.25, 0.25]], base=-1.0)


def test_mi_non_negative() -> None:
    joint = [[0.2, 0.3], [0.1, 0.4]]
    assert mutual_information(joint) >= 0


def test_mi_base_e() -> None:
    joint = [[0.5, 0.0], [0.0, 0.5]]
    assert mutual_information(joint, base=math.e) == pytest.approx(math.log(2))


def test_mi_zero_joint_entry_ignored() -> None:
    """Zero entries in joint contribute 0."""
    joint = [[0.5, 0.0], [0.0, 0.5]]
    # same as before, zeros ignored
    assert mutual_information(joint) == pytest.approx(1.0)


def test_cross_entropy_js_kl_consistency() -> None:
    """Sanity: JS <= (H(p,q)+H(q,p))/2 ??? Just check all functions run."""
    p = [0.2, 0.5, 0.3]
    q = [0.3, 0.3, 0.4]
    # All finite
    assert math.isfinite(entropy(p))
    assert math.isfinite(kl_divergence(p, q))
    assert math.isfinite(cross_entropy(p, q))
    assert math.isfinite(js_divergence(p, q))
    joint = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.3]]
    # normalize joint to sum 1: current sum =1.0, ok
    assert math.isfinite(mutual_information(joint))
