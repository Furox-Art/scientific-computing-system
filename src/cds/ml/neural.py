"""Pure Python Neural Network module."""

from __future__ import annotations

import math
import random

from cds.core._numeric import GD_DEFAULT_LR
from cds.optimization.minimize import AdamState, adam


class Layer:
    """A basic dense (fully-connected) neural network layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        seed: int | None = None,
    ):
        """Initialise Xavier/Glorot weights, zero biases, and backprop state buffers.

        Args:
            input_size: Width of the input vector (number of features).
            output_size: Number of neurons in this layer.
            activation: Nonlinearity name — ``"relu"``, ``"sigmoid"``,
                or ``"identity"`` (passthrough).
            seed: RNG seed for the weight initialisation. ``None`` keeps the
                historical behaviour of drawing from the process-global
                :mod:`random` stream, so ``random.seed(...)`` still makes a
                whole model reproducible; an explicit seed gives this layer a
                private stream that other callers cannot perturb.
        """
        # Xavier/Glorot initialization for weights
        limit = (6.0 / (input_size + output_size)) ** 0.5
        rng = random if seed is None else random.Random(seed)
        self.weights = [
            [rng.uniform(-limit, limit) for _ in range(input_size)] for _ in range(output_size)
        ]
        self.biases = [0.0] * output_size
        self.activation = activation

        # State for backpropagation
        self.last_x: list[float] = []
        self.last_z: list[float] = []
        self.last_a: list[float] = []
        self.grad_weights = [[0.0 for _ in range(input_size)] for _ in range(output_size)]
        self.grad_biases = [0.0] * output_size

    def forward(self, x: list[float]) -> list[float]:
        """Compute layer output for a single input vector and store state for backward pass."""
        self.last_x = x
        self.last_z = []
        self.last_a = []
        for i in range(len(self.weights)):
            z = sum(w * xi for w, xi in zip(self.weights[i], x)) + self.biases[i]
            self.last_z.append(z)
            self.last_a.append(self._activate(z))
        return self.last_a

    def backward(self, grad_out: list[float]) -> list[float]:
        """Backpropagate error gradient through the layer."""
        # dL/dz = dL/da * da/dz
        grad_z = [
            go * self._activate_derivative(z, a)
            for go, z, a in zip(grad_out, self.last_z, self.last_a)
        ]

        # dL/dw_ij = dL/dz_i * x_j
        for i in range(len(self.weights)):
            gz_i = grad_z[i]
            for j in range(len(self.weights[i])):
                self.grad_weights[i][j] += gz_i * self.last_x[j]

        # dL/db_i = dL/dz_i
        for i in range(len(self.biases)):
            self.grad_biases[i] += grad_z[i]

        # dL/dx_j = sum_i (dL/dz_i * w_ij)
        grad_in = [0.0] * len(self.last_x)
        for j in range(len(self.last_x)):
            grad_in[j] = sum(grad_z[i] * self.weights[i][j] for i in range(len(self.weights)))

        return grad_in

    def _activate(self, z: float) -> float:
        if self.activation == "relu":
            return max(0.0, z)
        if self.activation == "sigmoid":
            # Numerically stable logistic sigmoid. The two branches keep the
            # argument to exp() non-positive so it never overflows; for very
            # large |z| the exp() underflows to 0.0, which we map to the
            # asymptotic limits 1.0 (z -> +inf) / 0.0 (z -> -inf). The
            # OverflowError guard is kept defensively for platforms whose
            # libm raises on subnormal results rather than returning 0.0.
            if z >= 0:
                try:
                    return 1.0 / (1.0 + math.exp(-z))
                except (OverflowError, ValueError):  # pragma: no cover - non-CPython libm
                    return 1.0
            else:
                try:
                    ez = math.exp(z)
                except (OverflowError, ValueError):  # pragma: no cover - non-CPython libm
                    return 0.0
                return ez / (1.0 + ez)
        return z  # identity

    def _activate_derivative(self, z: float, a: float) -> float:
        if self.activation == "relu":
            return 1.0 if z > 0 else 0.0
        if self.activation == "sigmoid":
            return a * (1.0 - a)
        return 1.0  # identity


class MLP:
    """Multi-Layer Perceptron (Pure Python)."""

    def __init__(self, layers: list[Layer]):
        """Store the ordered :class:`Layer` stack; optimizer state starts unset.

        Args:
            layers: The :class:`Layer` instances, input layer first.
        """
        self.layers = layers
        self.optimizer_state: AdamState | None = None

    def predict(self, x: list[float]) -> list[float]:
        """Compute the network output."""
        curr = x
        for layer in self.layers:
            curr = layer.forward(curr)
        return curr

    def get_parameters(self) -> list[float]:
        """Flatten all weights and biases into a single list."""
        params = []
        for layer in self.layers:
            for row in layer.weights:
                params.extend(row)
            params.extend(layer.biases)
        return params

    def set_parameters(self, params: list[float]) -> None:
        """Unflatten parameters back into weights and biases."""
        idx = 0
        for layer in self.layers:
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    layer.weights[i][j] = params[idx]
                    idx += 1
            for i in range(len(layer.biases)):
                layer.biases[i] = params[idx]
                idx += 1

    def get_gradients(self) -> list[float]:
        """Flatten all accumulated gradients into a single list."""
        grads = []
        for layer in self.layers:
            for row in layer.grad_weights:
                grads.extend(row)
            grads.extend(layer.grad_biases)
        return grads

    def zero_grads(self) -> None:
        """Reset all parameter gradients to zero."""
        for layer in self.layers:
            for i in range(len(layer.grad_weights)):
                for j in range(len(layer.grad_weights[i])):
                    layer.grad_weights[i][j] = 0.0
            for i in range(len(layer.grad_biases)):
                layer.grad_biases[i] = 0.0

    def train(
        self,
        X: list[list[float]],
        y: list[list[float]],
        epochs: int = 100,
        lr: float = GD_DEFAULT_LR,
    ) -> dict[str, float | bool]:
        """Train the network using the Adam optimizer with backpropagation and state persistence."""

        def loss_fn(params: list[float]) -> float:
            """Mean squared error over the training set for these parameters."""
            self.set_parameters(params)
            total_loss = 0.0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                total_loss += sum((p - target) ** 2 for p, target in zip(pred, yi))
            return total_loss / len(X)

        def grad_fn(params: list[float]) -> list[float]:
            """Parameter gradients via backpropagation over the training set."""
            self.set_parameters(params)
            self.zero_grads()
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                # MSE gradient: dL/dp = 2/N * (p - y)
                grad_out = [2.0 * (p - target) / len(X) for p, target in zip(pred, yi)]
                curr_grad = grad_out
                for layer in reversed(self.layers):
                    curr_grad = layer.backward(curr_grad)
            return self.get_gradients()

        p0 = self.get_parameters()
        res = adam(loss_fn, p0, lr=lr, max_iter=epochs, state=self.optimizer_state, grad_f=grad_fn)

        # adam()'s overload for a list x0 returns OptResult[list[float]], so
        # res.x is statically a list[float] — no cast or isinstance narrowing.
        final_params = res.x
        self.set_parameters(final_params)
        self.optimizer_state = res.state  # Store state for next training call

        return {"final_loss": res.value, "iterations": res.iterations, "converged": res.converged}
