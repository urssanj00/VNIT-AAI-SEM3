import numpy as np

class ConvLayer:
    def __init__(self, num_filters, kernel_size=3, stride=1, padding=0):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernels = np.random.randn(num_filters, kernel_size, kernel_size) * 0.1


    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        print(f'ConvLayer : Input Matrix:\n{x}')
        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)
            
        input_h, input_w, input_c = x.shape

        if self.padding > 0:
            x_padded = np.pad(x, 
                              ((self.padding, self.padding),
                               (self.padding, self.padding),
                               (0, 0)), 
                              mode='constant')
        else:
            x_padded = x

        output_h = (input_h + 2*self.padding - self.kernel_size) // self.stride + 1
        output_w = (input_w + 2*self.padding - self.kernel_size) // self.stride + 1

        output = np.zeros((output_h, output_w, self.num_filters))

        for f in range(self.num_filters):
            kernel = self.kernels[f]
            for y in range(output_h):
                for x_pos in range(output_w):
                    y_start = y * self.stride
                    y_end = y_start + self.kernel_size
                    x_start = x_pos * self.stride
                    x_end = x_start + self.kernel_size

                    region = x_padded[y_start:y_end, x_start:x_end, :]
                 #   print(f'region : {region}')
                    output[y, x_pos, f] = np.sum(region * kernel[:, :, np.newaxis])
                 #   print(f'output[{y}, {x_pos}, {f}] : {output[y, x_pos, f]}')

        print(f'ConvLayer before ReLU output : {output}')
        return self.relu(output)