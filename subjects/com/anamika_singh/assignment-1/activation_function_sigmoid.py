import numpy as np
import matplotlib.pyplot as plt

"""
    LossPlotter Class
    Author: Sanjeev Kumar Pandey
    Date: April 10, 2025
    Description: 
    - Initializes random weights and biases,
    - Computes predictions using the sigmoid function,
    - Plots the loss function vs (w, b),
    - Visualizes activations across a multi-layer sigmoid network.
"""

class LossPlotter:
    def __init__(self, count=10):
        self.count = count
        self.weights, self.biases = self.generate_weights_and_biases()
        self.x_values = np.array([0.5, 2.5])  # Input values
        self.y_true = np.array([0.2, 0.9])  # Corresponding true values
        self.loss_values = None

    def generate_weights_and_biases(self):
        np.random.seed(42)
        weights = np.random.randn(self.count)
        biases = np.random.randn(self.count)
        return weights, biases

    def sigmoid(self, x, w, b):
        return 1 / (1 + np.exp(-(w * x + b)))

    def compute_predictions(self):
        y_preds = []  # Initialize an empty list
        for i in range(self.count):
            predictions = []  # Temporary list for each (w, b) pair
            for x in self.x_values:
                print(f"weights[{i}]: {self.weights[i]}, biases[{i}]: {self.biases[i]}")
                predictions.append(self.sigmoid(x, self.weights[i], self.biases[i]))  # Compute sigmoid
            print(f"predictions[{i}]: {predictions}")
            y_preds.append(predictions)  # Store predictions for the current (w, b)
        return np.array(y_preds)  # Convert list to numpy array

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def compute_loss(self):
        y_preds = self.compute_predictions()
        self.loss_values = np.zeros(self.count)  # Initialize an array
        for i in range(self.count):
            self.loss_values[i] = self.loss(self.y_true, y_preds[i])  # Compute loss iteratively
            print(f"loss_values[{i}]: {self.loss_values[i]}")


    def plot_loss(self):
        if self.loss_values is None:
            self.compute_loss()

        plt.figure(figsize=(8, 5))
        plt.plot(range(self.count), self.loss_values, 'ro-', linewidth=2, label="Loss Function")

        # Annotate each red dot with (w, b)
        for i in range(self.count):
            plt.annotate(f"w={self.weights[i]:.2f}, b={self.biases[i]:.2f}",
                         (i, self.loss_values[i]),
                         textcoords="offset points",
                         xytext=(-20, 10),
                         ha='center', fontsize=10, color='black')

        plt.xlabel("Point Index")
        plt.ylabel("Loss")
        plt.title("Loss Function Plot with (w, b) Annotations")
        plt.legend()
        plt.grid()
        plt.savefig('loss_plot.png')
        plt.close()

        # Plot 2: Sigmoid Tower (all sigmoid curves together)
        x = np.linspace(-10, 10, 500)
        plt.figure(figsize=(10, 6))

        for i in range(self.count):
            y = self.sigmoid(x, self.weights[i], self.biases[i])
            plt.plot(x, y, label=f'#{i + 1}: w={self.weights[i]:.2f}, b={self.biases[i]:.2f}')

            # Find top of the curve (x where y is max)
            max_idx = np.argmax(y)
            x_max = x[max_idx]
            y_max = y[max_idx]

            # Place label slightly above the curve
            plt.text(x_max, y_max + 0.03, f"#{i + 1}", fontsize=9, ha='center', va='bottom', fontweight='bold')

        plt.title("Sigmoid Activation Tower (All (w, b) pairs)")
        plt.xlabel("x")
        plt.ylabel("sigmoid(wx + b)")
        plt.ylim(-0.1, 1.1)
        plt.grid(True)
        plt.legend(fontsize='small', loc='best', ncol=2)
        plt.savefig('sigmoid_tower.png')
        plt.close()

    def visualize_activation_tower(self, layers=5):
        x = np.linspace(-10, 10, 400).reshape(-1, 1)
        np.random.seed(42)

        weights = [np.random.randn(1, 1) for _ in range(layers)]
        biases = [np.random.randn(1) for _ in range(layers)]

        activations = [x]
        for i in range(layers):
            z = np.dot(activations[-1], weights[i]) + biases[i]
            a = 1 / (1 + np.exp(-z))
            activations.append(a)

        plt.figure(figsize=(10, 6))
        for i, act in enumerate(activations[1:], 1):
            plt.plot(x, act, label=f"Layer {i}")

        plt.title("Multi-Layer Sigmoid Activation Tower")
        plt.xlabel("Input x")
        plt.ylabel("Activation")
        plt.legend()
        plt.grid(True)
        plt.savefig("activation_tower.png")
        plt.close()
# Create an instance and run the plot
plotter = LossPlotter(count=10)
plotter.plot_loss()
plotter.visualize_activation_tower(layers=5)
