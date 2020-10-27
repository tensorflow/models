"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks


@ks.utils.register_keras_serializable(package='yolo')
class Identity(ks.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input):
        return input
