from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
from logger_config import logger

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

logger.info(f"train_images.shape: {train_images.shape}")
logger.info(f"train_label: {train_labels}")

logger.info(f"test_images.shape: {test_images.shape}")
logger.info(f"test_labels: {test_labels}")

# our model consists of a sequence of two Dense layers, which are densely connected (also called fully connected)
# neural layers. The second (and last) layer is a 10-way softmax classification layer, which means it will return
# an array of 10 probability scores (summing to 1). Each score will be the probability that the current digit image
# belongs to one of our 10 digit classes.

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
logger.info(f"model: {model}")

# 1. An optimizer—The mechanism through which the model will update itself based on the training data it sees,
# so as to improve its performance.
# 2. A loss function—How the model will be able to measure its performance on the training data, and thus how
# it will be able to steer itself in the right direction.
# 3. Metrics to monitor during training and testing—Here, we’ll only care about accuracy (the fraction of the
# images that were correctly classified).

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
logger.info(f"model.compile ")

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

model.fit(train_images, train_labels, epochs=5, batch_size=128)

logger.info(f"model.fit done ")

test_digits = test_images[0:10]
logger.info(f"test_digits : {test_digits}")

predictions = model.predict(test_digits)
logger.info(f"predictions : {predictions}")
logger.info(f"predictions[0] : {predictions[0]}")

logger.info(f"predictions[0].argmax() : {predictions[0].argmax()}")
logger.info(f"predictions[0][7] : {predictions[0][7]}")

logger.info(f"test_labels[0] : {test_labels[0]}")

test_loss, test_acc = model.evaluate(test_images, test_labels)
logger.info(f"test_acc: {test_acc}")
