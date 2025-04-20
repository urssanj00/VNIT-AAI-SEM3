import numpy as np
from MT23AAI001_assignment_2_part_1 import ConvLayer
from MT23AAI001_assignment_2_part_2 import PoolingLayer
from MT23AAI001_assignment_2_part_3 import DenseLayer

class AlexNet:
    def __init__(self, num_classes=10):
        """
        Build the AlexNet architecture
        """
        self.conv1 = ConvLayer(num_filters=96, kernel_size=11, stride=4, padding=0)
        self.pool1 = PoolingLayer(kernel_size=3, stride=2, pooling_type='max')

        self.conv2 = ConvLayer(num_filters=256, kernel_size=5, stride=1, padding=2)
        self.pool2 = PoolingLayer(kernel_size=3, stride=2, pooling_type='max')

        self.conv3 = ConvLayer(num_filters=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvLayer(num_filters=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvLayer(num_filters=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = PoolingLayer(kernel_size=3, stride=2, pooling_type='max')

        # Dense layers
        self.fc1 = DenseLayer(input_size=6*6*256, output_size=4096)
        self.fc2 = DenseLayer(input_size=4096, output_size=4096)
        self.fc3 = DenseLayer(input_size=4096, output_size=num_classes)

    def forward(self, x):
        """
        Forward pass through AlexNet
        """
        x = self.conv1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.pool2.forward(x)

        x = self.conv3.forward(x)
        x = self.conv4.forward(x)
        x = self.conv5.forward(x)
        x = self.pool3.forward(x)

        x = x.flatten()

        x = self.fc1.forward(x)
        x = self.fc2.forward(x)
        x = self.fc3.forward(x)

        return x

if __name__ == "__main__":
    # Example input: (227, 227, 3) image as in original AlexNet
    input_image = np.random.randn(227, 227, 3)

    model = AlexNet(num_classes=10)
    output = model.forward(input_image)

    print("Output vector:", output)
