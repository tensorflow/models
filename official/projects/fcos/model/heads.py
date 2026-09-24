"""implamentation for the classification and bounding box genertation 
head for the FCOS model which is equal to the heads of RetinaNet"""

import tensorflow as tf
import keras


def head(output_classes, bias):
    """the output head for the FCOS model from the FPN

    Args:
        output_classes (int): number of classes in the output layer
        bias: the initial bias value
    Output:
        model (keras.model): the head model
    """
    kernel_initial = tf.initializers.RandomNormal(0.0, 0.01)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=[None, None, 256]),
            tf.keras.layers.Conv2D(
                256, 3, padding="same", kernel_initializer=kernel_initial
            ),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.GroupNormalization(),
            tf.keras.layers.Conv2D(
                256, 3, padding="same", kernel_initializer=kernel_initial
            ),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.GroupNormalization(),
            tf.keras.layers.Conv2D(
                256, 3, padding="same", kernel_initializer=kernel_initial
            ),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.GroupNormalization(),
            tf.keras.layers.Conv2D(
                256, 3, padding="same", kernel_initializer=kernel_initial
            ),
            tf.keras.layers.ReLU(),
            # tf.keras.layers.GroupNormalization(),
            tf.keras.layers.Conv2D(
                output_classes,
                3,
                1,
                padding="same",
                kernel_initializer=kernel_initial,
                bias_initializer=bias,
            ),
        ]
    )

    return model
