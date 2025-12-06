from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def build_mnist_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Build a small CNN for MNIST classification.
    """
    model = models.Sequential(
        [
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def build_adult_mlp(input_dim: int, num_classes: int) -> tf.keras.Model:
    """
    Build a small MLP for Adult Income classification.
    """
    model = models.Sequential(
        [
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model
