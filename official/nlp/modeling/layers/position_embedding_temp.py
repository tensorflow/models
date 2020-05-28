from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.layers import position_embedding

pos_layer = position_embedding.PositionEmbedding(
    use_relative=True)
input_tensor = tf.zeros([1, 32])
pos_encoding = pos_layer(input_tensor)
print (pos_encoding)
tf.print(pos_encoding)
