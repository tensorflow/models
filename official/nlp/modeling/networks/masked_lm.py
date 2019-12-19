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
"""Masked language model network."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.engine import network  # pylint: disable=g-direct-tensorflow-import
from official.modeling import tf_utils


@tf.keras.utils.register_keras_serializable(package='Text')
class MaskedLM(network.Network):
  """Masked language model network head for BERT modeling.

  This network implements a masked language model based on the provided network.
  It assumes that the network being passed has a "get_embedding_table()" method.

  Attributes:
    input_width: The innermost dimension of the input tensor to this network.
    num_predictions: The number of predictions to make per sequence.
    source_network: The network with the embedding layer to use for the
      embedding layer.
    activation: The activation, if any, for the dense layer in this network.
    initializer: The intializer for the dense layer in this network. Defaults to
      a Glorot uniform initializer.
    output: The output style for this network. Can be either 'logits' or
      'predictions'.
  """

  def __init__(self,
               input_width,
               num_predictions,
               source_network,
               activation=None,
               initializer='glorot_uniform',
               output='logits',
               **kwargs):

    embedding_table = source_network.get_embedding_table()
    vocab_size, hidden_size = embedding_table.shape

    sequence_data = tf.keras.layers.Input(
        shape=(None, input_width), name='sequence_data', dtype=tf.float32)
    masked_lm_positions = tf.keras.layers.Input(
        shape=(num_predictions,), name='masked_lm_positions', dtype=tf.int32)

    masked_lm_input = tf.keras.layers.Lambda(
        lambda x: self._gather_indexes(x[0], x[1]))(
            [sequence_data, masked_lm_positions])
    lm_data = (
        tf.keras.layers.Dense(
            hidden_size,
            activation=activation,
            kernel_initializer=initializer,
            name='cls/predictions/transform/dense')(masked_lm_input))
    lm_data = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name='cls/predictions/transform/LayerNorm')(
            lm_data)
    lm_data = tf.keras.layers.Lambda(
        lambda x: tf.matmul(x, embedding_table, transpose_b=True))(
            lm_data)
    logits = Bias(
        initializer=tf.keras.initializers.Zeros(),
        name='cls/predictions/output_bias')(
            lm_data)

    # We can't use the standard Keras reshape layer here, since it expects
    # the input and output batch size to be the same.
    reshape_layer = tf.keras.layers.Lambda(
        lambda x: tf.reshape(x, [-1, num_predictions, vocab_size]))

    self.logits = reshape_layer(logits)
    predictions = tf.keras.layers.Activation(tf.nn.log_softmax)(self.logits)

    if output == 'logits':
      output_tensors = self.logits
    elif output == 'predictions':
      output_tensors = predictions
    else:
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)

    super(MaskedLM, self).__init__(
        inputs=[sequence_data, masked_lm_positions],
        outputs=output_tensors,
        **kwargs)

  def get_config(self):
    raise NotImplementedError('MaskedLM cannot be directly serialized at this '
                              'time. Please use it only in Layers or '
                              'functionally subclassed Models/Networks.')

  def _gather_indexes(self, sequence_tensor, positions):
    """Gathers the vectors at the specific positions.

    Args:
        sequence_tensor: Sequence output of `BertModel` layer of shape
          (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
          hidden units of `BertModel` layer.
        positions: Positions ids of tokens in sequence to mask for pretraining
          of with dimension (batch_size, num_predictions) where
          `num_predictions` is maximum number of tokens to mask out and predict
          per each sequence.

    Returns:
        Masked out sequence tensor of shape (batch_size * num_predictions,
        num_hidden).
    """
    sequence_shape = tf_utils.get_shape_list(
        sequence_tensor, name='sequence_output_tensor')
    batch_size, seq_length, width = sequence_shape

    flat_offsets = tf.keras.backend.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.keras.backend.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.keras.backend.reshape(
        sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

    return output_tensor


@tf.keras.utils.register_keras_serializable(package='Text')
# Temporary until we can create a Dense layer that ties the embedding.
class Bias(tf.keras.layers.Layer):
  """Adds a bias term to an input."""

  def __init__(self,
               initializer='zeros',
               regularizer=None,
               constraint=None,
               activation=None,
               **kwargs):
    super(Bias, self).__init__(**kwargs)
    self._initializer = tf.keras.initializers.get(initializer)
    self._regularizer = tf.keras.regularizers.get(regularizer)
    self._constraint = tf.keras.constraints.get(constraint)
    self._activation = tf.keras.activations.get(activation)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self._bias = self.add_weight(
        'bias',
        shape=input_shape[1:],
        initializer=self._initializer,
        regularizer=self._regularizer,
        constraint=self._constraint,
        dtype=self._dtype,
        trainable=True)

    super(Bias, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'activation': tf.keras.activations.serialize(self._activation),
        'initializer': tf.keras.initializers.serialize(self._initializer),
        'regularizer': tf.keras.regularizers.serialize(self._regularizer),
        'constraint': tf.keras.constraints.serialize(self._constraint)
    }
    base_config = super(Bias, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    outputs = tf.nn.bias_add(inputs, self._bias)
    if self._activation is not None:
      return self._activation(outputs)  # pylint: disable=not-callable
    else:
      return outputs
