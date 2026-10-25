"""A seeded ``Layer`` owns its randomness instead of the global stream.

The dense layer drew Xavier/Glorot weights from the process-global
:mod:`random` module, so two runs of the same model were reproducible only if
every unrelated caller happened to seed and consume that stream identically.
An explicit seed now gives the layer a private stream, while the default keeps
the historical global-stream behaviour.
"""

from __future__ import annotations

import random

from cds.ml import MLP, Layer


def test_same_seed_reproduces_weights_regardless_of_global_state() -> None:
    random.seed(1)
    first = Layer(3, 2, seed=20261025)
    random.seed(999)
    second = Layer(3, 2, seed=20261025)
    assert first.weights == second.weights


def test_different_seeds_produce_different_weights() -> None:
    assert Layer(3, 2, seed=1).weights != Layer(3, 2, seed=2).weights


def test_explicit_seed_does_not_advance_the_global_stream() -> None:
    random.seed(7)
    expected = random.random()
    random.seed(7)
    Layer(3, 2, seed=99)
    assert random.random() == expected


def test_unseeded_layer_still_follows_the_global_stream() -> None:
    random.seed(7)
    first = Layer(3, 2)
    random.seed(7)
    second = Layer(3, 2)
    assert first.weights == second.weights


def test_mlp_training_is_reproducible_with_seeded_layers() -> None:
    def build() -> MLP:
        return MLP(
            [
                Layer(2, 3, activation="relu", seed=11),
                Layer(3, 1, activation="sigmoid", seed=12),
            ]
        )

    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    targets = [[0.0], [1.0], [1.0], [0.0]]
    first, second = build(), build()
    first.train(inputs, targets, epochs=20)
    second.train(inputs, targets, epochs=20)
    assert [first.predict(x) for x in inputs] == [second.predict(x) for x in inputs]
