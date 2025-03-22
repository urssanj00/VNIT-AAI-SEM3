import numpy as np
import matplotlib.pyplot as plt


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
        plt.show()


# Create an instance and run the plot
plotter = LossPlotter(count=10)
plotter.plot_loss()
