from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.layers import position_embedding

hidden_size = 32

pos_layer = position_embedding.PositionEmbeddingRelative(
    hidden_size=hidden_size)
input_tensor = tf.keras.Input(shape=(None, 32))
output = pos_layer(input_tensor)
print ('input shape', input_tensor.shape.as_list())
print ('output shape', output.shape.as_list())

pos_layer = position_embedding.PositionEmbeddingRelative(
    hidden_size=hidden_size)
input_tensor = tf.keras.Input(shape=(1, 32))
output = pos_layer(input_tensor)
print ('input shape', input_tensor.shape.as_list())
print ('output shape', output.shape.as_list())


pos_layer = position_embedding.PositionEmbeddingRelative(
    hidden_size=hidden_size)
input_tensor = tf.keras.Input(shape=(97, 32))
output = pos_layer(input_tensor)
print ('input shape', input_tensor.shape.as_list())
print ('output shape', output.shape.as_list())

pos_layer = position_embedding.PositionEmbeddingRelative(
    hidden_size=hidden_size)
input_tensor = tf.constant([[[0,0,0], [1,1,1]]])
output = pos_layer(input_tensor)
tf.print('output', output)
