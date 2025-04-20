# Testing ConvLayer with small 4x4 input
import numpy as np
from MT23AAI001_assignment_2_part_1 import ConvLayer

# Create a simple input
input_matrix = np.array([
    [1, 2, 0, 1],
    [3, 1, 2, 2],
    [0, 1, 3, 1],
    [2, 2, 1, 0]
])

# Create ConvLayer (1 filter, 2x2 kernel, stride 1, padding 0)
conv_layer = ConvLayer(num_filters=1, kernel_size=2, stride=1, padding=0)

# Forward pass
output = conv_layer.forward(input_matrix)

print("Convolution Output:")
print(output)  # remove last dimension for clarity
print(f"final ConvLayer output {output[:, :, 0]}")  # remove last dimension for clarity
