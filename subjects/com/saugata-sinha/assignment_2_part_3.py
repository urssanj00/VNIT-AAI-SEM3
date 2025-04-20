import numpy as np
from assignment_2_part_1 import ConvLayer
from assignment_2_part_2 import PoolingLayer


class DenseLayer:
    def __init__(self, input_size, output_size):
        """
        Initialize dense layer with He initialization for weights

        Parameters:
        - input_size: int - size of input features (flattened dimension)
        - output_size: int - number of neurons in the layer
        """
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def forward(self, x):
        """
        Perform forward propagation through the dense layer

        Parameters:
        - x: numpy array - input data (any dimension, will be flattened)

        Returns:
        - numpy array: activated output of shape (output_size,)
        """
        # Flatten input while preserving batch dimension (if present)
        original_shape = x.shape
        if x.ndim > 1:
            x = x.reshape(original_shape[0], -1)  # (batch_size, features)
        else:
            x = x.reshape(1, -1)  # single sample

        # Linear transformation: Wx + b
        linear_output = np.dot(x, self.weights) + self.bias

        # Apply ReLU activation
        activated = self.relu(linear_output)

        return activated.squeeze()  # remove batch dimension for single sample


# Example usage
if __name__ == "__main__":
    # Example workflow (not in original code)
    input_image = np.random.randn(32, 32, 3)  # Example input

    # Conv -> Pool -> Dense
    conv_layer = ConvLayer(kernel=np.random.randn(3, 3), stride=1, padding=1)
    pool_layer = PoolingLayer(kernel_size=2, stride=2)
    dense_layer = DenseLayer(input_size=16 * 16 * 64, output_size=10)  # After flattening

    # Forward pass
    x = conv_layer.forward(input_image)  # Shape: (32,32,num_filters)
    x = pool_layer.forward(x)  # Shape: (16,16,num_filters)
    x = x.flatten()  # Shape: (16*16*num_filters,)
    output = dense_layer.forward(x)  # Shape: (10,)

