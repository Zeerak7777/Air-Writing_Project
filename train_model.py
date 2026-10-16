"""
train_model.py
---------------
Run this ONCE, before starting app.py, to train and save the letter
classifier: `python3 train_model.py`

Uses TensorFlow Datasets (tfds) to fetch EMNIST's 'letters' split. This is
a more reliable source than the older `emnist` pip package, whose original
download link has become unstable.

Output: model/emnist_cnn.h5
"""

import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

NUM_CLASSES = 26  # A-Z

print("Loading EMNIST 'letters' dataset (first run downloads ~44MB)...")
ds_train = tfds.load("emnist/letters", split="train", as_supervised=True)
ds_test = tfds.load("emnist/letters", split="test", as_supervised=True)


def valid_label(image, label):
    return tf.logical_and(label >= 1, label <= 26)


def preprocess(image, label):
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.cast(image, tf.float32) / 255.0
    label = label - 1
    return image, label


ds_train = (
    ds_train.filter(valid_label)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(20000)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)
ds_test = (
    ds_test.filter(valid_label)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

print("Training...")
model.fit(ds_train, validation_data=ds_test, epochs=10)

test_loss, test_acc = model.evaluate(ds_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

os.makedirs("model", exist_ok=True)
model.save("model/emnist_cnn.h5")
print("Saved model to model/emnist_cnn.h5")
