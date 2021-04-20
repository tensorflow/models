# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Masked language model network."""
# pylint: disable=g-classes-have-attributes
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='keras_nlp')
class MaskedLM(tf.keras.layers.Layer):
  """Masked language model network head for BERT modeling.

  This layer implements a masked language model based on the provided
  transformer based encoder. It assumes that the encoder network being passed
  has a "get_embedding_table()" method.

  Example:
  ```python
  encoder=keras_nlp.BertEncoder(...)
  lm_layer=MaskedLM(embedding_table=encoder.get_embedding_table())
  ```

  Args:
    embedding_table: The embedding table from encoder network.
    activation: The activation, if any, for the dense layer.
    initializer: The initializer for the dense layer. Defaults to a Glorot
      uniform initializer.
    output: The output style for this layer. Can be either 'logits' or
      'predictions'.
  """

  def __init__(self,
               embedding_table,
               activation=None,
               initializer='glorot_uniform',
               output='logits',
               name=None,
               **kwargs):
    super(MaskedLM, self).__init__(name=name, **kwargs)
    self.embedding_table = embedding_table
    self.activation = activation
    self.initializer = tf.keras.initializers.get(initializer)

    if output not in ('predictions', 'logits'):
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)
    self._output_type = output

  def build(self, input_shape):
    self._vocab_size, hidden_size = self.embedding_table.shape
    self.dense = tf.keras.layers.Dense(
        hidden_size,
        activation=self.activation,
        kernel_initializer=self.initializer,
        name='transform/dense')
    self.layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name='transform/LayerNorm')
    self.bias = self.add_weight(
        'output_bias/bias',
        shape=(self._vocab_size,),
        initializer='zeros',
        trainable=True)

    super(MaskedLM, self).build(input_shape)

  def call(self, sequence_data, masked_positions):
    masked_lm_input = self._gather_indexes(sequence_data, masked_positions)
    lm_data = self.dense(masked_lm_input)
    lm_data = self.layer_norm(lm_data)
    lm_data = tf.matmul(lm_data, self.embedding_table, transpose_b=True)
    logits = tf.nn.bias_add(lm_data, self.bias)
    masked_positions_length = masked_positions.shape.as_list()[1] or tf.shape(
        masked_positions)[1]
    logits = tf.reshape(logits,
                        [-1, masked_positions_length, self._vocab_size])
    if self._output_type == 'logits':
      return logits
    return tf.nn.log_softmax(logits)

  def get_config(self):
    raise NotImplementedError('MaskedLM cannot be directly serialized because '
                              'it has variable sharing logic.')

  def _gather_indexes(self, sequence_tensor, positions):
    """Gathers the vectors at the specific positions, for performance.

    Args:
        sequence_tensor: Sequence output of shape
          (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
          hidden units.
        positions: Positions ids of tokens in sequence to mask for pretraining
          of with dimension (batch_size, num_predictions) where
          `num_predictions` is maximum number of tokens to mask out and predict
          per each sequence.

    Returns:
        Masked out sequence tensor of shape (batch_size * num_predictions,
        num_hidden).
    """
    sequence_shape = tf.shape(sequence_tensor)
    batch_size, seq_length = sequence_shape[0], sequence_shape[1]
    width = sequence_tensor.shape.as_list()[2] or sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

    return output_tensor
