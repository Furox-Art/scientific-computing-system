# Neural Networks + Visualization Tutorial

`cds.ml` is a from-scratch neural network library (MLP, dense layers, Adam
optimizer) — no PyTorch, no NumPy. Pair it with `cds.data_analysis` for
terminal-native ASCII charts to inspect results without a plotting backend.

## 1. Build and train an MLP

Define the network as a list of `Layer`s with explicit input/output sizes and
activations.

```python
from cds.ml import MLP, Layer

# XOR-like logic: input [x1, x2] -> output [x1 OR x2]
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [1.0]]

net = MLP(
    [
        Layer(2, 4, activation="relu"),
        Layer(4, 1, activation="sigmoid"),
    ]
)
```

Before training the outputs sit near 0.5 (random init):

```
Initial predictions (Untrained):
  In: [0.0, 0.0] -> Out: 0.5000
  In: [0.0, 1.0] -> Out: 0.3825
  ...
```

Train with the built-in Adam optimizer (momentum state persisted across steps):

```python
history = net.train(X, y, epochs=200, lr=0.1)
print(f"Final loss: {history['final_loss']}")
```

After 200 epochs the network has learned the OR function:

```
Final predictions (Trained):
  In: [0.0, 0.0] -> Out: 0.0406
  In: [0.0, 1.0] -> Out: 0.9908
  In: [1.0, 0.0] -> Out: 0.9915
  In: [1.0, 1.0] -> Out: 1.0000
```

`history` carries `final_loss`, `iterations`, and a `converged` flag so you can
decide programmatically whether to keep training.

## 2. Visualize results in the terminal

`cds.data_analysis` ships `plot_bar` and `plot_line` — scale-aware ASCII charts
that need no matplotlib.

```python
from cds.data_analysis import plot_bar, plot_line
import math

stats = {"Quantum": 95.5, "Signals": 88.2, "Math": 92.0, "ML": 99.1}
print(plot_bar(stats, title="Module Readiness Score"))

wave = [math.sin(x * 0.2) for x in range(50)]
print(plot_line(wave, title="Generated Sine Wave (ASCII)"))
```

```
Module Readiness Score
──────────────────────
Quantum  | ████████████████████████████████████████████████ (+95.50)
Signals  | ████████████████████████████████████████████ (+88.20)
...
```

The line plot auto-scales to `[min, max]` and renders with `•` markers, so you
can eyeball loss curves or signal shapes straight from a terminal.

## 3. The pure-Python advantage

Because every layer, optimizer step, and chart is readable Python, this is an
ideal setup for **teaching** how backpropagation and Adam actually work — you
can set breakpoints inside the training loop and watch gradients flow.

Run the full demo:

```bash
python examples/ml_and_viz_demo.py
```
