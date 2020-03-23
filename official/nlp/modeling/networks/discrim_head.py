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
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras.engine import network  # pylint: disable=g-direct-tensorflow-import
from official.modeling import tf_utils


@tf.keras.utils.register_keras_serializable(package='Text')
class Discriminator(network.Network):
  """Masked language model network head for BERT modeling.

  This network implements a masked language model based on the provided network.
  It assumes that the network being passed has a "get_embedding_table()" method.

  Arguments:
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
        shape=(num_predictions,), name='input_masks', dtype=tf.int32)

    masked_lm_input = tf.keras.layers.Lambda(
        lambda x: self._gather_indexes(x[0], x[1]))(
            [sequence_data, masked_lm_positions])
    lm_data = (
        tf.keras.layers.Dense(
            hidden_size,
            activation=activation,
            kernel_initializer=initializer,
            name='cls/predictions/transform/dense')(masked_lm_input))
    self.logits = tf.squeeze(tf.keras.layers.Dense(1)(lm_data), axis=-1)

    output_tensors = self.logits
    super(Discriminator, self).__init__(
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

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

    return output_tensor
