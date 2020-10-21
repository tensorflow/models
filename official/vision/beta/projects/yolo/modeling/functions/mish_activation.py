import tensorflow as tf
import tensorflow.keras as ks

class mish(ks.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def call(self, x):
        return x * tf.math.tanh(ks.activations.softplus(x))
