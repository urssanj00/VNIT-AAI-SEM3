# Testing PoolingLayer with small 4x4x1 input
import numpy as np
from MT23AAI001_assignment_2_part_2 import PoolingLayer
from MT23AAI001_assignment_2_part_1_usage import output
# Create a simple input (adding channel dimension)
input_matrix = output

# Create PoolingLayer (2x2 kernel, stride 2, max pooling)
pool_layer = PoolingLayer(kernel_size=2, stride=2, pooling_type='max')

# Forward pass
output = pool_layer.forward(input_matrix)

print("Pooling Output:")
print(output[:, :, 0])  # remove last dimension for clarity
