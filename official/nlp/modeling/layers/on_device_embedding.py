# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras-based one-hot embedding layer."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Text")
class OnDeviceEmbedding(tf.keras.layers.Layer):
  """Performs an embedding lookup suitable for accelerator devices.

  This layer uses either tf.gather or tf.one_hot to translate integer indices to
  float embeddings.

  Arguments:
    vocab_size: Number of elements in the vocabulary.
    embedding_width: Output size of the embedding layer.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    use_one_hot: Whether to use tf.one_hot over tf.gather for the embedding
      lookup. Defaults to False (that is, using tf.gather). Setting this option
      to True may improve performance, especially on small vocabulary sizes,
      but will generally require more memory.
  """

  def __init__(self,
               vocab_size,
               embedding_width,
               initializer="glorot_uniform",
               use_one_hot=False,
               **kwargs):
    # We need to have a default dtype of float32, since the inputs (which Keras
    # usually uses to infer the dtype) will always be int32.
    if "dtype" not in kwargs:
      kwargs["dtype"] = "float32"

    super(OnDeviceEmbedding, self).__init__(**kwargs)
    self._vocab_size = vocab_size
    self._embedding_width = embedding_width
    self._initializer = initializer
    self._use_one_hot = use_one_hot

  def get_config(self):
    config = {
        "vocab_size": self._vocab_size,
        "embedding_width": self._embedding_width,
        "initializer": self._initializer,
        "use_one_hot": self._use_one_hot,
    }
    base_config = super(OnDeviceEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    self.embeddings = self.add_weight(
        "embeddings",
        shape=[self._vocab_size, self._embedding_width],
        initializer=self._initializer)

    super(OnDeviceEmbedding, self).build(input_shape)

  def call(self, inputs):
    flat_inputs = tf.reshape(inputs, [-1])
    if self._use_one_hot:
      one_hot_data = tf.one_hot(
          flat_inputs, depth=self._vocab_size, dtype=self._dtype)
      embeddings = tf.matmul(one_hot_data, self.embeddings)
    else:
      embeddings = tf.gather(self.embeddings, flat_inputs)
    embeddings = tf.reshape(
        embeddings,
        # Work around b/142213824: prefer concat to shape over a Python list.
        tf.concat([tf.shape(inputs), [self._embedding_width]], axis=0))
    embeddings.set_shape(inputs.shape.as_list() + [self._embedding_width])
    return embeddings
