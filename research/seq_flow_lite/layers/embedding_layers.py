# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
"""Layers for embedding."""
import tensorflow as tf

from layers import base_layers # import seq_flow_lite module
from layers import quantization_layers # import seq_flow_lite module


class EmbeddingLayer(base_layers.BaseLayer):
  """Embedding layer."""

  def __init__(self,
               shape,
               num_bits=8,
               initializer=None,
               trainable=True,
               **kwargs):
    self.shape = shape
    self.quantizer = quantization_layers.ActivationQuantization(
        num_bits=num_bits, **kwargs)
    super(EmbeddingLayer, self).__init__(**kwargs)
    if initializer is None:
      initializer = tf.keras.initializers.GlorotUniform()
    self.initializer = initializer
    self.trainable = trainable

  def build(self, input_shapes):
    self.embedding_table = self.add_weight(
        name="embedding_table",
        shape=self.shape,
        initializer=self.initializer,
        trainable=self.trainable,
        dtype=tf.float32)
    if self.trainable:
      self.add_reg_loss(self.embedding_table)

  def call(self, indices):
    assert indices.dtype in [tf.int64, tf.int32]
    outputs = tf.nn.embedding_lookup(self.embedding_table, indices)
    return self.quantizer(outputs)


class EmbeddingFullyConnected(EmbeddingLayer):
  """Uses embedding table as weights in a fully connected op."""

  def __init__(self, **kwargs):
    shape = kwargs.pop("shape", None)
    initializer = kwargs.pop("initializer", None)
    self.qoutput = quantization_layers.ActivationQuantization(**kwargs)
    super(EmbeddingFullyConnected, self).__init__(
        shape=shape, initializer=initializer, **kwargs)

  def fully_connected(self, inputs, bias=None, weights_scale_factor=None):
    # This method can only be called after a call to "call" method in this class
    self._assert_rank_and_type(inputs, 2)
    weights = self.embedding_table
    if weights_scale_factor is not None:
      weights = weights * weights_scale_factor
    outputs = tf.matmul(inputs, weights, transpose_b=True)
    if bias is not None:
      outputs = tf.nn.bias_add(outputs, bias)
    return self.qoutput(outputs)
