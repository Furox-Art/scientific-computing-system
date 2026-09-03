# Machine Learning Tutorial

`cds.ml` ships a small, dependency-free multilayer perceptron (`MLP`) built on a
stack of `Layer` objects. Training uses the Adam optimizer from
[`cds.optimization`](optimization_demo.md), so the whole pipeline, forward
pass, backprop, and the optimizer step, is pure Python.

## 1. Build and inspect a network

Construct a layer with `(in_features, out_features, activation)`. Supported
activations are `relu`, `sigmoid`, and `tanh`. Stack them into an `MLP`.

```python
from cds.ml import MLP, Layer

# 2 inputs -> 4 hidden (ReLU) -> 1 output (Sigmoid)
net = MLP(
    [
        Layer(2, 4, activation="relu"),
        Layer(4, 1, activation="sigmoid"),
    ]
)
```

## 2. Predict before training

`predict` takes a feature vector and returns the network's output.

```python
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]  # OR-gate inputs
y = [[0.0], [1.0], [1.0], [1.0]]

for xi in X:
    print(f"  In: {xi} -> Out: {net.predict(xi)[0]:.4f}")
# Untrained network outputs ≈ 0.5 everywhere:
#   In: [0.0, 0.0] -> Out: 0.5000
#   In: [0.0, 1.0] -> Out: 0.5159
#   In: [1.0, 0.0] -> Out: 0.4839
#   In: [1.0, 1.0] -> Out: 0.5249
```

## 3. Train with Adam

`train` runs Adam for `epochs` iterations over the dataset and returns a history
dict with the final loss, iteration count, and convergence flag.

```python
history = net.train(X, y, epochs=200, lr=0.1)
print(history)
# {'final_loss': 0.000403..., 'iterations': 200, 'converged': False}

for xi in X:
    print(f"  In: {xi} -> Out: {net.predict(xi)[0]:.4f}")
# After training the network has learned the OR function:
#   In: [0.0, 0.0] -> Out: 0.0394
#   In: [0.0, 1.0] -> Out: 0.9967
#   In: [1.0, 0.0] -> Out: 0.9927
#   In: [1.0, 1.0] -> Out: 1.0000
```

## 4. Resumable optimizer state

Internally, `train` stores Adam's moment estimates on the `MLP` instance
(`net.optimizer_state`). Calling `train` again resumes from that checkpoint
rather than restarting, so you can train in stages or lower the learning rate
between phases.

---

Run the full demo with `python examples/ml_and_viz_demo.py`.
