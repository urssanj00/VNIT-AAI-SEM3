# Testing DenseLayer with small input
import numpy as np
from MT23AAI001_assignment_2_part_3 import DenseLayer
from MT23AAI001_assignment_2_part_1 import ConvLayer
from MT23AAI001_assignment_2_part_2 import PoolingLayer

# Create a small flattened input (say 6 features)
input_image = np.array([
    [1, 2, 0, 1],
    [3, 1, 2, 2],
    [0, 1, 3, 1],
    [2, 2, 1, 0]
])

conv_layer = ConvLayer(num_filters=8, kernel_size=3, stride=1, padding=1)
pool_layer = PoolingLayer(kernel_size=2, stride=2, pooling_type='max')

x = conv_layer.forward(input_image)
print(f'conv output : {x}')
x = pool_layer.forward(x)
print(f'maxpool output : {x}')

x = x.flatten()

dense_layer = DenseLayer(input_size=x.size, output_size=10)

output = dense_layer.forward(x)
print(f"Dense layer Output vector:{output}")