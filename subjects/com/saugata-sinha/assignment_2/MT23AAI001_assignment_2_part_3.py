import numpy as np
from MT23AAI001_assignment_2_part_1 import ConvLayer
from MT23AAI001_assignment_2_part_2 import PoolingLayer

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        original_shape = x.shape
        if x.ndim > 1:
            x = x.reshape(original_shape[0], -1)
        else:
            x = x.reshape(1, -1)

        linear_output = np.dot(x, self.weights) + self.bias
        activated = self.relu(linear_output)
        return activated.squeeze()

if __name__ == "__main__":
    input_image = np.random.randn(32, 32, 3)

    conv_layer = ConvLayer(num_filters=8, kernel_size=3, stride=1, padding=1)
    pool_layer = PoolingLayer(kernel_size=2, stride=2, pooling_type='max')

    x = conv_layer.forward(input_image)
    x = pool_layer.forward(x)
    x = x.flatten()

    dense_layer = DenseLayer(input_size=x.size, output_size=10)

    output = dense_layer.forward(x)
    print("Output vector:", output)