"""Contains common building blocks for Mesh R-CNN."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='mesh_rcnn')
class GraphConv(tf.keras.layers.Layer):
  def __init__(self,):
    pass

  def build(self, input_shape):
    pass

  def call(self, x):
    pass

  def get_config(self):
    layer_config = {

    }

    layer_config.update(super().get_config())
    return layer_config

@tf.keras.utils.register_keras_serializable(package='mesh_rcnn')
class ResGraphConv(tf.keras.layers.Layer):
  def __init__(self,):
    pass

  def build(self, input_shape):
    pass

  def call(self, x):
    pass

  def get_config(self):
    layer_config = {

    }

    layer_config.update(super().get_config())
    return layer_config
