"""Tests for :mod:`cds.nlp.attention`.

Covers the building blocks (``matmul``, ``softmax``, ``transpose``),
the attention formulas themselves, multi-head mechanics, and the
causal mask.
"""

from __future__ import annotations

import pytest

from cds.nlp.attention import (
    causal_mask,
    matmul,
    merge_heads,
    multi_head_attention,
    scaled_dot_product_attention,
    softmax,
    split_heads,
    transpose,
)

# ---------------------------------------------------------------------- #
# matmul
# ---------------------------------------------------------------------- #


class TestMatmul:
    """Matrix multiply on nested lists."""

    def test_identity_right(self) -> None:
        a = [[1.0, 2.0], [3.0, 4.0]]
        identity = [[1.0, 0.0], [0.0, 1.0]]
        assert matmul(a, identity) == a

    def test_identity_left(self) -> None:
        a = [[1.0, 2.0], [3.0, 4.0]]
        identity = [[1.0, 0.0], [0.0, 1.0]]
        assert matmul(identity, a) == a

    def test_known_product(self) -> None:
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        assert matmul(a, b) == [[19.0, 22.0], [43.0, 50.0]]

    def test_zeros(self) -> None:
        a = [[0.0, 0.0], [0.0, 0.0]]
        b = [[1.0, 2.0], [3.0, 4.0]]
        result = matmul(a, b)
        assert all(v == 0.0 for row in result for v in row)

    def test_empty_inputs(self) -> None:
        assert matmul([], [[1.0]]) == []
        assert matmul([[1.0]], []) == []

    def test_shape_mismatch_raises(self) -> None:
        a = [[1.0, 2.0, 3.0]]  # 1x3
        b = [[1.0, 2.0]]  # 2x2
        with pytest.raises(ValueError, match="matmul shape mismatch"):
            matmul(a, b)

    def test_rectangular(self) -> None:
        a = [[1.0, 2.0, 3.0]]  # 1x3
        b = [[1.0], [2.0], [3.0]]  # 3x1
        assert matmul(a, b) == [[14.0]]

    def test_does_not_mutate_inputs(self) -> None:
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        a_snap = [list(row) for row in a]
        b_snap = [list(row) for row in b]
        matmul(a, b)
        assert a == a_snap
        assert b == b_snap


# ---------------------------------------------------------------------- #
# transpose / softmax
# ---------------------------------------------------------------------- #


class TestTranspose:
    def test_2x3(self) -> None:
        m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert transpose(m) == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_square(self) -> None:
        m = [[1.0, 2.0], [3.0, 4.0]]
        assert transpose(m) == [[1.0, 3.0], [2.0, 4.0]]

    def test_empty(self) -> None:
        assert transpose([]) == []


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        s = softmax([1.0, 2.0, 3.0])
        assert abs(sum(s) - 1.0) < 1e-12

    def test_monotonic(self) -> None:
        """Larger input → larger output."""
        s = softmax([1.0, 2.0, 3.0])
        assert s[0] < s[1] < s[2]

    def test_numerically_stable_large(self) -> None:
        """Inputs near exp overflow must not crash."""
        s = softmax([1000.0, 1001.0, 1002.0])
        assert abs(sum(s) - 1.0) < 1e-12
        assert s[2] > s[1] > s[0]

    def test_empty(self) -> None:
        assert softmax([]) == []

    def test_all_equal(self) -> None:
        """Constant input → uniform output."""
        s = softmax([0.5, 0.5, 0.5])
        for v in s:
            assert abs(v - 1.0 / 3) < 1e-12

    def test_all_neg_inf_returns_uniform(self) -> None:
        """Degenerate input (all -inf after subtracting max) → uniform."""
        s = softmax([float("-inf"), float("-inf"), float("-inf")])
        for v in s:
            assert abs(v - 1.0 / 3) < 1e-9

    def test_mixed_neg_inf(self) -> None:
        """A mix of -inf and finite → finite entries get all the mass."""
        s = softmax([float("-inf"), 0.0, 0.0])
        assert s[0] == 0.0
        assert abs(s[1] - 0.5) < 1e-12
        assert abs(s[2] - 0.5) < 1e-12


# ---------------------------------------------------------------------- #
# scaled_dot_product_attention
# ---------------------------------------------------------------------- #


class TestScaledDotProductAttention:
    def test_shape(self) -> None:
        q = [[1.0, 0.0], [0.0, 1.0]]
        k = [[1.0, 0.0], [0.0, 1.0]]
        v = [[1.0, 2.0], [3.0, 4.0]]
        out = scaled_dot_product_attention(q, k, v)
        assert len(out) == 2
        assert len(out[0]) == 2

    def test_self_attention_diagonal(self) -> None:
        """With identity-like q/k, the attention is mostly diagonal."""
        q = [[1.0, 0.0], [0.0, 1.0]]
        k = [[1.0, 0.0], [0.0, 1.0]]
        v = [[1.0, 0.0], [0.0, 1.0]]
        out = scaled_dot_product_attention(q, k, v)
        # Position 0 attends most to position 0 (and partly to 1).
        assert out[0][0] > 0.4
        assert out[0][0] > out[0][1]

    def test_mask_zeros_position(self) -> None:
        """A mask of -inf on position 1 must zero its attention weight."""
        q = [[1.0, 0.0], [0.0, 1.0]]
        k = [[1.0, 0.0], [0.0, 1.0]]
        v = [[1.0, 2.0], [3.0, 4.0]]
        # Mask out position 1 (k index 1) for query position 0.
        mask = [[0.0, float("-inf")], [0.0, 0.0]]
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        # With position 1 masked, query 0 should output exactly v[0].
        assert abs(out[0][0] - 1.0) < 1e-6
        assert abs(out[0][1] - 2.0) < 1e-6

    def test_empty_q_returns_empty(self) -> None:
        assert scaled_dot_product_attention([], [], []) == []

    def test_zero_width_q_raises(self) -> None:
        q: list[list[float]] = [[]]  # one row, zero columns
        with pytest.raises(ValueError, match="d_k"):
            scaled_dot_product_attention(q, q, q)

    def test_kv_count_mismatch_raises(self) -> None:
        q = [[1.0, 0.0]]
        k = [[1.0, 0.0]]
        v = [[1.0, 2.0], [3.0, 4.0]]  # 2 rows, k has 1
        with pytest.raises(ValueError, match="rows"):
            scaled_dot_product_attention(q, k, v)

    def test_mask_shape_mismatch_raises(self) -> None:
        q = [[1.0, 0.0], [0.0, 1.0]]
        k = [[1.0, 0.0], [0.0, 1.0]]
        v = [[1.0], [2.0]]
        mask = [[0.0]]  # 1x1, expected 2x2
        with pytest.raises(ValueError, match="mask shape"):
            scaled_dot_product_attention(q, k, v, mask=mask)

    def test_qk_width_mismatch_raises(self) -> None:
        q = [[1.0, 0.0]]
        k = [[1.0, 0.0, 0.0]]  # wider than q
        v = [[1.0]]
        with pytest.raises(ValueError, match="widths differ"):
            scaled_dot_product_attention(q, k, v)

    @pytest.mark.parametrize("d_k", [2, 4, 8, 16])
    def test_output_shape_preserved(self, d_k: int) -> None:
        n = 3
        q = [[0.1 * (i + j) for j in range(d_k)] for i in range(n)]
        k = [[0.1 * (i + j + 1) for j in range(d_k)] for i in range(n)]
        v = [[1.0] * d_k for _ in range(n)]
        out = scaled_dot_product_attention(q, k, v)
        assert len(out) == n
        assert all(len(row) == d_k for row in out)


# ---------------------------------------------------------------------- #
# split_heads / merge_heads
# ---------------------------------------------------------------------- #


class TestSplitMergeHeads:
    def test_split_shape(self) -> None:
        # n=2, d_model=4, n_heads=2 → each head is (2, 2)
        x = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        heads = split_heads(x, n_heads=2)
        assert len(heads) == 2
        assert all(len(h) == 2 and len(h[0]) == 2 for h in heads)
        # First head: cols 0-1, second head: cols 2-3.
        assert heads[0] == [[1.0, 2.0], [5.0, 6.0]]
        assert heads[1] == [[3.0, 4.0], [7.0, 8.0]]

    def test_split_merge_round_trip(self) -> None:
        x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        heads = split_heads(x, n_heads=3)
        merged = merge_heads(heads)
        assert merged == x

    def test_split_d_model_not_divisible_raises(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            split_heads([[1.0, 2.0, 3.0]], n_heads=2)

    def test_split_n_heads_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_heads"):
            split_heads([[1.0, 2.0]], n_heads=0)

    def test_split_empty(self) -> None:
        # Empty input → list of empty heads.
        heads = split_heads([], n_heads=2)
        assert len(heads) == 2
        assert all(h == [] for h in heads)

    def test_merge_empty(self) -> None:
        assert merge_heads([]) == []


# ---------------------------------------------------------------------- #
# multi_head_attention
# ---------------------------------------------------------------------- #


class TestMultiHeadAttention:
    @staticmethod
    def _random_projection(rows: int, cols: int, seed: int) -> list[list[float]]:
        import random

        rng = random.Random(seed)
        return [[rng.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]

    def test_output_shape_preserved(self) -> None:
        n, d_model, n_heads = 4, 8, 2
        x = [[0.1] * d_model for _ in range(n)]
        w_q = self._random_projection(d_model, d_model, seed=1)
        w_k = self._random_projection(d_model, d_model, seed=2)
        w_v = self._random_projection(d_model, d_model, seed=3)
        w_o = self._random_projection(d_model, d_model, seed=4)
        out = multi_head_attention(x, w_q, w_k, w_v, w_o, n_heads)
        assert len(out) == n
        assert all(len(row) == d_model for row in out)

    def test_empty_input_returns_empty(self) -> None:
        out = multi_head_attention([], [[1.0]], [[1.0]], [[1.0]], [[1.0]], n_heads=1)
        assert out == []

    def test_d_model_not_divisible_raises(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            multi_head_attention(
                [[1.0, 2.0, 3.0]],
                [[1.0, 2.0, 3.0]] * 3,
                [[1.0, 2.0, 3.0]] * 3,
                [[1.0, 2.0, 3.0]] * 3,
                [[1.0, 2.0, 3.0]] * 3,
                n_heads=2,
            )

    def test_causal_mask_blocks_future(self) -> None:
        """With a causal mask, position i cannot depend on position j > i."""
        n, d_model, n_heads = 3, 4, 2
        x = [[float(i)] * d_model for i in range(n)]
        w_q = self._random_projection(d_model, d_model, seed=7920)
        w_k = self._random_projection(d_model, d_model, seed=7921)
        w_v = self._random_projection(d_model, d_model, seed=7922)
        w_o = self._random_projection(d_model, d_model, seed=7923)
        mask = causal_mask(n)
        out = multi_head_attention(x, w_q, w_k, w_v, w_o, n_heads, mask=mask)
        # First output position must not depend on later v's — verify
        # by recomputing with a different v for position 1+.
        w_v_alt = self._random_projection(d_model, d_model, seed=99)
        out_alt = multi_head_attention(x, w_q, w_k, w_v_alt, w_o, n_heads, mask=mask)
        # Position 0's output is identical (no future v's attend into it).
        for j in range(d_model):
            assert abs(out[0][j] - out_alt[0][j]) < 1e-9


# ---------------------------------------------------------------------- #
# causal_mask
# ---------------------------------------------------------------------- #


class TestCausalMask:
    def test_shape(self) -> None:
        m = causal_mask(4)
        assert len(m) == 4
        assert all(len(row) == 4 for row in m)

    def test_upper_triangular_neg_inf(self) -> None:
        m = causal_mask(3)
        # Diagonal and below → 0.0
        for i in range(3):
            for j in range(3):
                if j <= i:
                    assert m[i][j] == 0.0
                else:
                    assert m[i][j] == float("-inf")

    def test_empty_mask(self) -> None:
        assert causal_mask(0) == []

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 0"):
            causal_mask(-1)
