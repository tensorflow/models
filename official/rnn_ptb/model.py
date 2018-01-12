# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Defines the PTB model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Embedding(tf.layers.Layer):
  """An embedding layer."""
  def __init__(self, vocab_size, embedding_size, max_init_value, **kwargs):
    super(Embedding, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.initializer = tf.random_uniform_initializer(
        -max_init_value, max_init_value)

  def build(self, _):
    self.embedding = self.add_variable(
        'embedding',
        shape=[self.vocab_size, self.embedding_size],
        dtype=tf.float32,
        initializer=self.initializer)
    self.built = True

  def call(self, inputs):
    return tf.nn.embedding_lookup(self.embedding, inputs)


class PTBModel(object):
  """Recurrent language model.

  Applies dropout when training, as described in https://arxiv.org/abs/1409.2329
  """

  def __init__(self, mode, embedding_size, hidden_size, keep_prob, num_layers,
      batch_size, vocab_size, max_init_value, max_init_value_emb):
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.keep_prob = keep_prob

    # Only use dropout when in training mode.
    self.use_dropout = (mode == tf.estimator.ModeKeys.TRAIN) and (keep_prob < 1)

    # Create embedding layer.
    self.embedding = Embedding(vocab_size, embedding_size, max_init_value_emb)

    # Add rnn and dense layers with weights initialized as defined in the params
    initializer = tf.random_uniform_initializer(-max_init_value, max_init_value)
    with tf.variable_scope('rnn_dense_layers', initializer=initializer):
      self.rnn = self._create_rnn()
      self.dense = tf.layers.Dense(vocab_size)

    # Save the state as a tensor with shape
    # [num_layers, 2, batch_size, hidden_size]
    self.initial_state = tf.convert_to_tensor(
        self.rnn.zero_state(batch_size, tf.float32))

  @staticmethod
  def from_params(mode, params):
    # batch size = 1 if in predict mode.
    batch_size = params.batch_size
    if mode == tf.estimator.ModeKeys.PREDICT:
      batch_size = 1
    return PTBModel(mode=mode,
                    embedding_size=params.embedding_size,
                    hidden_size=params.hidden_size,
                    keep_prob=params.keep_prob,
                    num_layers=params.num_layers,
                    batch_size=batch_size,
                    vocab_size=params.vocab_size,
                    max_init_value=params.max_init_value,
                    max_init_value_emb=params.max_init_value_emb)

  @property
  def state(self):
    """Hidden state of the LSTM cells."""
    # Use variable scope to allow the variable to be reused.
    with tf.variable_scope('state', reuse=tf.AUTO_REUSE):
      return tf.get_local_variable('rnn_state', initializer=self.initial_state)

  def __call__(self, inputs, reset_state=None):
    """Compute logits from input features.

    Args:
      inputs: feature tensor with shape [batch_size, unrolled_count]
      reset_state: A boolean tensor indicating whether to calculate the logits
          with the initial state or the current hidden state.

    Returns:
      logits: tensor with the same shape as inputs.
      state: tuple of size num_layers. Elements in the tuple are LSTMStateTuple.
    """
    # Get the embedding of the inputs
    # New shape of inputs: [batch_size, unrolled_count, embedding_size]
    inputs = self.embedding(inputs)

    if self.use_dropout:
      inputs = tf.nn.dropout(inputs, self.keep_prob)

    # Set the state as the current or initial hidden state.
    state = tf.cond(reset_state,
                    lambda: self.initial_state,
                    lambda: self.state)

    # Get the output and new state of the RNN.
    # First, unstack the inputs --> inputs is now a list of size unrolled_count.
    # Each item in the list has shape: [batch_size, embedding_size]
    inputs = tf.unstack(inputs, axis=1)

    # Run the inputs and state through the rnn to obtain the outputs and final
    # hidden state. outputs is a list of size unrolled_count.
    # Each item in the list has shape: [batch_size, hidden_size]
    outputs, state = tf.nn.static_rnn(
        self.rnn, inputs, initial_state=self._convert_to_state_tuple(state))

    # Stack outputs --> new shape: [batch_size, unrolled_count, hidden_size]
    outputs = tf.stack(outputs, axis=1)

    # Update the state and use control dependency to ensure that the op runs.
    update_state_op = tf.assign(self.state, state)
    with tf.control_dependencies([update_state_op]):
      return self.dense(outputs), state  # Return the logits and state

  def _create_rnn(self):
    """Returns a MultiRNNCell made up of LSTM cells."""
    cells = []
    for layer in range(self.num_layers):
      with tf.variable_scope('layer_%d' % layer):
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=0.)

        if self.use_dropout:
          cell = tf.nn.rnn_cell.DropoutWrapper(
              cell, output_keep_prob=self.keep_prob)
      cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

  def _convert_to_state_tuple(self, state):
    """Converts a tensor to a tuple of LSTMStateTuples.

    Args:
      state: A tensor with shape: [num_layers, 2, batch_size, hidden_size].
          The `2` refers to the memory cell(c) and hidden state(h) of LSTM cells

    Returns:
      A tuple with length of num_layers. Each element is a LSTMStateTuple.
    """
    state_tuple = []
    for i in range(self.num_layers):
      state_tuple.append(
          tf.nn.rnn_cell.LSTMStateTuple(c=state[i, 0], h=state[i, 1]))
    return tuple(state_tuple)

  def predict_next(self, logits, state, num_predictions):
    """Predict the next n words, given the initial logits and state."""
    # Randomly choose the next word based on the logits.
    next_word = self._get_next_possible_word(logits)

    # Create a list of predicted words (includes the input sequence)
    predictions = [next_word]
    for i in range(num_predictions - 1):
      # Obtain the logits for the next prediction
      embedded_input = self.embedding([next_word])
      output, state = self.rnn(embedded_input, state)
      logits = self.dense(output)

      next_word = self._get_next_possible_word(logits)
      predictions.append(next_word)
    return tf.convert_to_tensor([predictions])

  def _get_next_possible_word(self, logits, skip=3):
    """Randomly select a word id from the given logits.

    Args:
      logits: Log probabilities of each word
      skip: Number of symbols to skip from the start of the logits list.
            (By default, skip=3, which skips the first 3 elements in the list:
            '<unk>', 'N', and '$'.)

    Returns:
      A randomly selected word id based on the logits.
    """
    logits = logits[0, skip:]

    next_word = tf.multinomial([logits], 1)[0][0] + skip
    return next_word
