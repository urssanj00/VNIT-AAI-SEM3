from NaiveDense import NaiveDense
import tensorflow as tf
from logger_config import logger

class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


model = NaiveSequential([NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
                         NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
                         ])
logger.info(f"model : {model}")
assert len(model.weights) == 4
