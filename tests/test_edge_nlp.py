"""Coverage completion for the educational NLP module.

Targets the last uncovered lines/branches reported by
`pytest --cov=cds.nlp --cov-branch --cov-report=term-missing`:

- nlp/autograd/tensor.py  — reflected dunders (__radd__, __rsub__, ...),
                            __pos__, backward() cycle-skip, TypeError path
- nlp/autograd/ops.py     — log(0) gradient guard
- nlp/optim.py            — Adam validator branches
- nlp/bpe.py              — decode non-EOS token, train progress print
- nlp/layers.py           — _add empty-list and shape-mismatch branches
- nlp/model.py            — sample temperature, attention mask, sample() fn
- nlp/attention.py        — merge_heads empty-rows branch
"""

from __future__ import annotations

import pytest

from cds.nlp import BPEMerge, BPETokenizer
from cds.nlp.attention import merge_heads
from cds.nlp.autograd import Tensor as AGTensor
from cds.nlp.autograd import log as ag_log
from cds.nlp.bpe import train_bpe
from cds.nlp.layers import _add
from cds.nlp.model import MiniGPT, sample
from cds.nlp.optim import Adam

# ---------------------------------------------------------------------------
# 1. tensor.py — reflected operators + __pos__ + cycle skip + TypeError
# ---------------------------------------------------------------------------


class TestTensorReflectedOps:
    """Cover __radd__/__rsub__/__rmul__/__rtruediv__ (lines 56,62,74,77)."""

    def test_radd(self) -> None:
        t = AGTensor(data=3.0)
        out = 2.0 + t  # calls __radd__
        assert out.data == 5.0

    def test_rsub(self) -> None:
        t = AGTensor(data=3.0)
        out = 10.0 - t  # calls __rsub__
        assert out.data == 7.0

    def test_rmul(self) -> None:
        t = AGTensor(data=3.0)
        out = 4.0 * t  # calls __rmul__
        assert out.data == 12.0

    def test_rtruediv(self) -> None:
        t = AGTensor(data=2.0)
        out = 10.0 / t  # calls __rtruediv__
        assert out.data == 5.0


class TestTensorPosAndPowType:
    """Cover __pos__ (line 80) and __pow__ TypeError (line 255)."""

    def test_pos_returns_self(self) -> None:
        t = AGTensor(data=5.0)
        assert (+t).data == 5.0

    def test_pow_rejects_tensor_exponent(self) -> None:
        t = AGTensor(data=2.0)
        with pytest.raises(TypeError):
            # __pow__ is typed as float-only, so mypy flags this statically;
            # the test asserts it fails at runtime via NotImplemented.
            _ = t ** AGTensor(data=3.0)  # type: ignore[operator]


class TestBackwardCycleSkip:
    """Cover the `if node in visited: continue` branch (lines 153,157,158).

    Construct a graph where a child is reachable by two paths so the DFS
    encounters a node already in `visited`.
    """

    def test_diamond_graph_visits_each_node_once(self) -> None:
        a = AGTensor(data=2.0, requires_grad=True)
        b = a * AGTensor(data=3.0)
        c = a * AGTensor(data=4.0)
        d = b + c
        d.backward()
        # a appears in two paths; grad accumulates from both b and c
        # db/da = 3, dc/da = 4, so dd/da = 7
        assert a.grad == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 2. autograd/ops.py — log(0) gradient undefined (line 39)
# ---------------------------------------------------------------------------


class TestLogZeroGradient:
    """Cover the `if a.data == 0.0: raise ValueError` branch."""

    def test_log_of_zero_raises_on_forward(self) -> None:
        """log() guards against non-positive input on the forward pass."""
        zero = AGTensor(data=0.0, requires_grad=True)
        with pytest.raises(ValueError, match="log requires positive"):
            ag_log(zero)


# ---------------------------------------------------------------------------
# 3. optim.py — Adam.__post_init__ validation (lines 124, 130)
# ---------------------------------------------------------------------------


class TestAdamValidation:
    """Cover Adam.__post_init__ error branches."""

    def test_rejects_zero_lr(self) -> None:
        with pytest.raises(ValueError, match="lr must be > 0"):
            Adam(params=[], lr=0.0)

    def test_rejects_negative_weight_decay(self) -> None:
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            Adam(params=[], lr=0.01, weight_decay=-0.1)


# ---------------------------------------------------------------------------
# 4. layers.py — _add internal helper (lines 175, 179)
# ---------------------------------------------------------------------------


class TestLayersAddHelper:
    """Cover _add empty-list early return and shape-mismatch error."""

    def test_empty_returns_empty(self) -> None:
        assert _add([], []) == []

    def test_shape_mismatch_raises(self) -> None:
        a = [[1.0, 2.0]]
        b = [[1.0, 2.0], [3.0, 4.0]]  # different row count
        with pytest.raises(ValueError, match="add shape mismatch"):
            _add(a, b)


# ---------------------------------------------------------------------------
# 5. bpe.py — decode non-EOS branch (line 301) + train progress print (431)
# ---------------------------------------------------------------------------


class TestBPEDecodeProgress:
    """Cover decode path that appends a non-terminator token, and the
    progress-print branch in train_bpe (line 431)."""

    def test_train_bpe_prints_progress(self, capsys: pytest.CaptureFixture[str]) -> None:
        corpus = "low lower lowest"
        train_bpe(corpus, vocab_size=20, show_progress=True)
        out = capsys.readouterr().out
        assert "merge" in out

    def test_decode_reconstructs_multi_token_word(self) -> None:
        # Build a tokenizer whose vocab has character-level tokens (no </w>
        # suffix) so the `else: current.append(tok)` branch fires.
        vocab = {
            "l": 1,
            "o": 2,
            "w": 3,
            "lo": 4,
            "low": 5,
            "</w>": 6,
            "low</w>": 7,
        }
        merges = [
            BPEMerge(pair=("l", "o"), rank=0, new_token="lo"),
            BPEMerge(pair=("lo", "w"), rank=1, new_token="low"),
            BPEMerge(pair=("low", "</w>"), rank=2, new_token="low</w>"),
        ]
        tok = BPETokenizer(vocab=vocab, merges=merges)
        # Decode the "lo" token (id=4) — it has no </w> suffix, so the
        # `else: current.append(tok)` branch fires (line 301).
        decoded = tok.decode([4, 7])
        assert "lo" in decoded
        assert "low" in decoded


# ---------------------------------------------------------------------------
# 6. model.py — sample temperature branch (239), sample() wrapper fn (466),
#               attention mask via forward (373->382)
# ---------------------------------------------------------------------------


class TestMiniGPTSampling:
    """Cover temperature != 1.0 in sample() and the module-level sample()."""

    def _tiny_model(self) -> MiniGPT:
        return MiniGPT(vocab_size=5, d_model=4, n_heads=2, max_len=8, seed=0)

    def test_sample_with_temperature(self) -> None:
        m = self._tiny_model()
        ids = m.sample(prompt_ids=[0, 1], n_tokens=3, temperature=0.5)
        assert isinstance(ids, list)
        assert len(ids) >= 2

    def test_sample_module_function(self) -> None:
        m = self._tiny_model()
        ids = sample(m, prompt_ids=[0], n_tokens=2)
        assert isinstance(ids, list)


class TestMiniGPTAttentionMaskBranch:
    """Cover the masked-attention branch in forward (line 373->382) by
    driving a forward pass on a short prompt."""

    def test_forward_with_causal_mask(self) -> None:
        m = MiniGPT(vocab_size=5, d_model=4, n_heads=2, max_len=8, seed=7919)
        # A forward pass on a short prompt exercises the causal mask branch
        # inside _multi_head_attention.
        logits = m.forward([0, 1, 2])
        assert len(logits) > 0


# ---------------------------------------------------------------------------
# 7. attention.py — merge_heads empty-rows branch (line 214)
# ---------------------------------------------------------------------------


class TestMergeHeadsEmptyRows:
    """Cover the `if n == 0` branch inside merge_heads."""

    def test_heads_with_empty_rows(self) -> None:
        # n_heads=2, each head has zero rows -> n == 0 branch
        heads: list[list[list[float]]] = [[], []]
        out = merge_heads(heads)
        assert out == []
