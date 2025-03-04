from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f"train_images.shape: {train_images.shape}")
print(f"train_label: {train_labels}")

print(f"test_images.shape: {test_images.shape}")
print(f"test_labels: {test_labels}")


model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])