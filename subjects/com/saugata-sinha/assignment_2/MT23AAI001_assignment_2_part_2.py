import numpy as np

class PoolingLayer:
    def __init__(self, kernel_size=2, stride=2, pooling_type='max', padding=0):
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.pooling_type = pooling_type
        self.padding = padding

    def forward(self, input_matrix):
        if self.padding > 0:
            input_padded = np.pad(input_matrix,
                                  ((self.padding, self.padding),
                                   (self.padding, self.padding),
                                   (0, 0)),
                                  mode='constant')
        else:
            input_padded = input_matrix

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        input_h, input_w, input_c = input_padded.shape

        out_h = (input_h - kernel_h) // stride_h + 1
        out_w = (input_w - kernel_w) // stride_w + 1

        output = np.zeros((out_h, out_w, input_c))

        for c in range(input_c):
            for y in range(out_h):
                for x in range(out_w):
                    y_start = y * stride_h
                    y_end = y_start + kernel_h
                    x_start = x * stride_w
                    x_end = x_start + kernel_w

                    window = input_padded[y_start:y_end, x_start:x_end, c]

                    if self.pooling_type == 'max':
                        output[y, x, c] = np.max(window)
                    elif self.pooling_type == 'average':
                        output[y, x, c] = np.mean(window)
                    else:
                        raise ValueError("Pooling type must be 'max' or 'average'")
                    
        return output