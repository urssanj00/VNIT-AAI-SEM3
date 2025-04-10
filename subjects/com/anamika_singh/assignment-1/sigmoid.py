import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# ---------------- Part 1: Sigmoid Response ----------------


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Visualize sigmoid for different weights and biases


def plot_sigmoid_variations():
    sns.set(style="whitegrid")
    x = np.linspace(-10, 10, 500)
    weights = [0.5, 1, 2, 5]
    biases = [-4, 0, 4]

    fig, axes = plt.subplots(len(biases), len(
        weights), figsize=(16, 10), sharex=True, sharey=True)
    fig.suptitle(
        "Sigmoid Activation Response to Varying Weights and Biases", fontsize=16)

    for i, b in enumerate(biases):
        for j, w in enumerate(weights):
            z = w * x + b
            y = sigmoid(z)
            ax = axes[i, j]
            ax.plot(x, y, color='blue')
            ax.set_title(f"w={w}, b={b}")
            ax.set_ylim([-0.1, 1.1])
            if i == len(biases) - 1:
                ax.set_xlabel("Input x")
            if j == 0:
                ax.set_ylabel("Sigmoid Output")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('plot_sigmoid_variations.png')
    plt.close()

# ---------------- Part 2: 2D Activation Tower ----------------


def sigmoid_layer(inputs, weights, biases):
    return sigmoid(np.dot(inputs, weights) + biases)


def plot_2d_activation_tower():
    # Create 2D input grid
    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)
    X1, X2 = np.meshgrid(x1, x2)
    input_grid = np.c_[X1.ravel(), X2.ravel()]

    # Define network layers
    np.random.seed(42)
    layers = [
        {"weights": np.random.randn(
            2, 3), "biases": np.random.randn(3)},  # 2 -> 3
        {"weights": np.random.randn(
            3, 2), "biases": np.random.randn(2)},  # 3 -> 2
        {"weights": np.random.randn(
            2, 1), "biases": np.random.randn(1)}   # 2 -> 1
    ]

    # Forward pass
    activations = [input_grid]
    for layer in layers:
        inputs = activations[-1]
        output = sigmoid_layer(inputs, layer["weights"], layer["biases"])
        activations.append(output)

    # Normalize activations
    scaler = MinMaxScaler()
    normalized_activations = [
        scaler.fit_transform(act) for act in activations[1:]]

    # Plot activations
    fig, axes = plt.subplots(1, len(normalized_activations), figsize=(18, 5))
    fig.suptitle("Multi-Layer 2D Activation Tower with Sigmoid", fontsize=16)

    for i, activation in enumerate(normalized_activations):
        ax = axes[i]
        sc = ax.scatter(X1, X2, c=activation[:, 0], cmap="viridis", s=5)
        ax.set_title(f"Layer {i+1} Activation (Neuron 1)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('plot_2d_activation_tower.png')
    plt.close()

# -------- Run Both Parts --------
plot_sigmoid_variations()
plot_2d_activation_tower()
