import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def plot_sigmoid_towers():
    # Create a 2D grid
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Define tower centers
    a1, b1 = -2, -2
    a2, b2 = 2, 2

    # Combine two sigmoid towers
    Z = sigmoid(-((X1 - a1)**2 + (X2 - b1)**2)) + sigmoid(-((X1 - a2)**2 + (X2 - b2)**2))

    # Plot
    fig = plt.figure(figsize=(10, 5))

    # Heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(X1, X2, Z, cmap='plasma')
    fig.colorbar(contour, ax=ax1)
    ax1.set_title("Sigmoid Towers - Heatmap")
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")

    # 3D Surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X1, X2, Z, cmap='plasma')
    ax2.set_title("Sigmoid Towers - 3D Surface")
    ax2.set_xlabel("X1")
    ax2.set_ylabel("X2")
    ax2.set_zlabel("Activation")

    plt.tight_layout()
    plt.savefig("sigmoid_towers.png")
    plt.show()

# Call the tower plotting function
plot_sigmoid_towers()
