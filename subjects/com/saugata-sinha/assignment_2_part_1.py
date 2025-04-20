import numpy as np


class ConvLayer:
    def __init__(self, kernel, stride=1, padding=0):
        """
        Initializes the convolution layer.

        Parameters:
        - kernel (np.ndarray): 2D convolution kernel.
        - stride (int): Stride of the convolution.
        - padding (int): Amount of zero-padding around the input.
        """
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        print(f'kernal: \n{kernel}')
        print(f'stride: \n{stride}')
        print(f'padding: \n{padding}')

    def relu(self, x):
        """Applies the ReLU activation function."""
        return np.maximum(0, x)

    def forward(self, input_matrix):
        """
        Performs the forward pass of the convolution layer.

        Parameters:
        - input_matrix (np.ndarray): 2D input array.

        Returns:
        - np.ndarray: Output after convolution and ReLU activation.
        """
		
		# Handle 3D inputs (height, width, channels)
        if input_matrix.ndim == 3:
            # Apply padding to spatial dimensions only
            pad_width = (
                (self.padding, self.padding), 
                (self.padding, self.padding),
                (0, 0)  # No padding for channels
            )
        else:
            pad_width = ((self.padding, self.padding), (self.padding, self.padding))
			
        # Apply padding if needed
        if self.padding > 0:
            input_padded = np.pad(
                input_matrix,
                pad_width,
                mode='constant', 
                constant_values=0

            )
        else:
            input_padded = input_matrix

        print(f'input_padded : \n{input_padded}')
        kernel_height, kernel_width = self.kernel.shape
        print(f'kernel_height: {kernel_height}, kernel_width: {kernel_width}')

        # Handle 2D/3D inputs
        if input_padded.ndim == 3:
            input_height, input_width, _ = input_padded.shape  # Ignore channels
        else:
            input_height, input_width = input_padded.shape
        print(f'input_height: {input_height}, input_width: {input_width}')

        # Calculate output dimensions
        out_height = (input_height - kernel_height) // self.stride + 1
        out_width = (input_width - kernel_width) // self.stride + 1
        print(f'out_height: {out_height}, out_width: {out_width}')

        # Initialize output
        output = np.zeros((out_height, out_width))
        print(f'output: \n{output}')

        # Perform cross-correlation
        for y in range(out_height):
            for x in range(out_width):
                y_start = y * self.stride
                y_end = y_start + kernel_height
                x_start = x * self.stride
                x_end = x_start + kernel_width

                region = input_padded[y_start:y_end, x_start:x_end]
                print(f'region: \n{region}')

                output[y, x] = np.sum(region * self.kernel)
                print(f'output[{y} {x}]:\n {output[y, x]}')

        print(f'output:\n {output}')
        output = self.relu(output)
        print(f'output after applying RelU:\n {output}')

        # Apply ReLU
        return output


# Example usage
if __name__ == "__main__":
    input_matrix = np.array([
        [1, 2, 3, 0],
        [4, 5, 6, 1],
        [7, 8, 9, 0],
        [0, 1, 2, 3]
    ])
    kernel = np.array([
        [1, 0],
        [0, -1]
    ])

    conv = ConvLayer(kernel, stride=1, padding=1)
    output = conv.forward(input_matrix)
    print(output)
