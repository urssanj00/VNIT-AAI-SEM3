import numpy as np
from assignment_2_part_1 import ConvLayer

class PoolingLayer:
    def __init__(self, kernel_size=2, stride=2, pooling_type='max', padding=0):
        """
        Initialize pooling layer

        Parameters:
        - kernel_size: int or tuple (height, width)
        - stride: int or tuple (vertical, horizontal)
        - pooling_type: 'max' or 'average'
        - padding: int (zeros to add around input)
        """
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.pooling_type = pooling_type
        self.padding = padding

    def forward(self, input_matrix):
        """
        Perform forward pass through pooling layer

        Parameters:
        - input_matrix: 2D numpy array

        Returns:
        - 2D numpy array of pooled values
        """
        # Apply padding
        if self.padding > 0:
            input_padded = np.pad(input_matrix,
                                  ((self.padding, self.padding),
                                   (self.padding, self.padding)),
                                  mode='constant')
        else:
            input_padded = input_matrix

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        input_h, input_w = input_padded.shape

        # Calculate output dimensions
        out_h = (input_h - kernel_h) // stride_h + 1
        out_w = (input_w - kernel_w) // stride_w + 1

        # Initialize output matrix
        output = np.zeros((out_h, out_w))

        # Perform pooling operation
        for y in range(out_h):
            for x in range(out_w):
                y_start = y * stride_h
                y_end = y_start + kernel_h
                x_start = x * stride_w
                x_end = x_start + kernel_w

                window = input_padded[y_start:y_end, x_start:x_end]
                print(f'window: \n{window}')

                if self.pooling_type == 'max':
                    output[y, x] = np.max(window)
                elif self.pooling_type == 'average':
                    output[y, x] = np.mean(window)
                else:
                    raise ValueError("Pooling type must be 'max' or 'average'")
                print(f'output[{y} {x}]:\n {output[y, x]}')

        return output





# Example usage with ConvLayer
if __name__ == "__main__":
    # Original input
    input_matrix = np.array([[1, 2, 3, 0], [4, 5, 6, 1], [7, 8, 9, 0], [0, 1, 2, 3]])

    # Convolution layer
    kernel = np.array([[1, 0], [0, -1]])
    conv = ConvLayer(kernel, stride=1, padding=1)
    conv_output = conv.forward(input_matrix)

    # Pooling layer
    pool = PoolingLayer(kernel_size=2, stride=2, pooling_type='max')
    pool_output = pool.forward(conv_output)

    print(f"Convolution output:\n {conv_output}")
    print(f"\nPooling output:\n {pool_output}")
