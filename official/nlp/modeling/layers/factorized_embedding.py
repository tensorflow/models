# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""A factorized embedding layer."""
# pylint: disable=g-classes-have-attributes

import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.nlp.modeling.layers import on_device_embedding


@tf_keras.utils.register_keras_serializable(package='Text')
class FactorizedEmbedding(on_device_embedding.OnDeviceEmbedding):
  """A factorized embeddings layer for supporting larger embeddings.

  Arguments:
    vocab_size: Number of elements in the vocabulary.
    embedding_width: Width of word embeddings.
    output_dim: The output dimension of this layer.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    use_one_hot: Whether to use tf.one_hot over tf.gather for the embedding
      lookup. Defaults to False (that is, using tf.gather). Setting this option
      to True may improve performance, especially on small vocabulary sizes, but
      will generally require more memory.
    scale_factor: Whether to scale the output embeddings. Defaults to None (that
      is, not to scale). Setting this option to a float will let values in
      output embeddings multiplied by scale_factor.
  """

  def __init__(self,
               vocab_size: int,
               embedding_width: int,
               output_dim: int,
               initializer='glorot_uniform',
               use_one_hot=False,
               scale_factor=None,
               **kwargs):
    super().__init__(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        use_one_hot=use_one_hot,
        scale_factor=scale_factor,
        **kwargs)
    self._output_dim = output_dim

  def get_config(self):
    config = {'output_dim': self._output_dim}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    self._embedding_projection = tf_keras.layers.EinsumDense(
        '...x,xy->...y',
        output_shape=self._output_dim,
        bias_axes=None,
        kernel_initializer=tf_utils.clone_initializer(self._initializer),
        name='embedding_projection')
    super().build(input_shape)

  def call(self, inputs):
    output = super().call(inputs)
    return self._embedding_projection(output)
