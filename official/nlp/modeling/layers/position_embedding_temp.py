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
input_tensor = tf.keras.Input(shape=(1, 32))
output = pos_layer(input_tensor)
print (output)
print ('test shape', output.shape.as_list())
