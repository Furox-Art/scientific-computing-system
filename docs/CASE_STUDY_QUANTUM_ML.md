# Case Study: Integrating Quantum Simulation with Machine Learning

## Overview
This case study combines quantum circuit simulation with classical machine learning using the Cognitive Discovery System (CDS). It uses `cds.quantum` for feature generation and `cds.ml` for predictive modeling to illustrate a hybrid quantum-classical workflow.

## 1. Building a Quantum Circuit with `cds.quantum`

The first step involves creating a quantum circuit that prepares a state based on physical or mathematical parameters. In this example, we use a rotation-based circuit to encode information into a qubit.

```python
from cds.quantum import QuantumCircuit, phase_gate, hadamard


def create_quantum_feature_circuit(theta: float):
    """Creates a quantum circuit with a phase rotation."""
    circuit = QuantumCircuit()
    circuit.add(hadamard())  # Put qubit in superposition
    circuit.add(phase_gate(theta))  # Apply parameter-dependent rotation
    return circuit


# Example: Encode an angle of 0.75 radians
circuit = create_quantum_feature_circuit(0.75)
```

## 2. Collecting Measurements with `cds.quantum.simulate`

Quantum states are probabilistic. To use them in classical ML, we must "measure" the state multiple times to collect statistics. These statistics (measurement frequencies) serve as our features.

```python
from cds.quantum import simulate


def get_quantum_features(theta: float, shots: int = 1000):
    """Generates features from quantum simulation."""
    circuit = create_quantum_feature_circuit(theta)
    counts = simulate(circuit, shots=shots)

    # Calculate probability of state |1>
    prob_1 = counts.get(1, 0) / shots
    return [prob_1]


# Collect features for a range of angles
X = [get_quantum_features(a / 10.0) for a in range(10)]
y = [[(a / 10.0) ** 2] for a in range(10)]  # Target: predict theta^2
```

## 3. Training a `cds.ml` Neural Network

With our quantum-derived features, we can now train a classical neural network to perform regression or classification.

```python
from cds.ml import MLP, Layer

# Define a simple MLP: 1 Input (Quantum Feature) -> 5 Hidden -> 1 Output
mlp = MLP([Layer(1, 5, activation="relu"), Layer(5, 1, activation="identity")])

# Train the network
history = mlp.train(X, y, epochs=200, lr=0.05)

print(f"Final Loss: {history['final_loss']:.6f}")

# Predict using the trained model
sample_feature = get_quantum_features(0.5)
prediction = mlp.predict(sample_feature)
print(f"Input: 0.5, Prediction: {prediction[0]:.4f}")
```

## 4. The 'Discovery' Value of Integration

The integration of `cds.quantum` and `cds.ml` offers significant value for scientific discovery:

1.  **Quantum-Enhanced Feature Spaces**: Quantum circuits can represent complex probability distributions that are hard to model classically. Using these as inputs to ML allows researchers to explore "Quantum Kernels" and non-linear mappings.
2.  **Hybrid Optimization**: This workflow enables the development of Variational Quantum Algorithms (VQAs), where a classical optimizer (like the Adam optimizer in `cds.ml`) tunes quantum parameters to minimize a physical objective function.
3.  **Noise Analysis**: By simulating quantum noise and feeding the results into an MLP, researchers can train models to identify and mitigate errors in real quantum hardware.
4.  **Scientific Insight**: The ability to bridge the gap between quantum mechanics and neural architectures allows for the discovery of new patterns in quantum materials, chemistry simulations, and complex system dynamics.

By combining the probabilistic nature of quantum mechanics with the predictive power of neural networks, this workflow offers a way to explore problems that sit between quantum simulation and classical modeling.
