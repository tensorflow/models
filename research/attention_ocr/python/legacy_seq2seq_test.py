# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for functional style sequence-to-sequence models."""

import numpy as np

import legacy_seq2seq

import tensorflow.compat.v1 as tf

from tensorflow.python.ops import rnn_cell_impl


class OutputProjectionWrapper(tf.nn.rnn_cell.RNNCell):
  """Operator adding an output projection to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your outputs in time,
  do the projection on this batch-concatenated sequence, then split it
  if needed or directly feed into a softmax.
  """

  def __init__(self, cell, output_size, activation=None, reuse=None):
    """Create a cell with output projection.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      output_size: integer, the size of the output after projection.
      activation: (optional) an optional activation function.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if output_size is not positive.
    """
    super(OutputProjectionWrapper, self).__init__(_reuse=reuse)
    rnn_cell_impl.assert_like_rnncell("cell", cell)
    if output_size < 1:
      raise ValueError("Parameter output_size must be > 0: %d." % output_size)
    self._cell = cell
    self._output_size = output_size
    self._activation = activation
    self._linear = None

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._output_size

  def zero_state(self, batch_size, dtype):
    with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def call(self, inputs, state):
    """Run the cell and output projection on inputs, starting from state."""
    output, res_state = self._cell(inputs, state)
    if self._linear is None:
      self._linear = legacy_seq2seq._Linear(output, self._output_size, True)
    projected = self._linear(output)
    if self._activation:
      projected = self._activation(projected)
    return projected, res_state


class Seq2SeqTest(tf.test.TestCase):

  def testRNNDecoder(self):
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        _, enc_state = tf.nn.static_rnn(
            tf.nn.rnn_cell.GRUCell(2), inp, dtype=tf.float32)
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3
        cell = OutputProjectionWrapper(tf.nn.rnn_cell.GRUCell(2), 4)
        dec, mem = legacy_seq2seq.rnn_decoder(dec_inp, enc_state, cell)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testAttentionDecoder1(self):
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        def cell_fn(): return tf.nn.rnn_cell.GRUCell(2)
        cell = cell_fn()
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        enc_outputs, enc_state = tf.nn.static_rnn(cell, inp, dtype=tf.float32)
        attn_states = tf.concat([
            tf.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs
        ], 1)
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3

        # Create a new cell instance for the decoder, since it uses a
        # different variable scope
        dec, mem = legacy_seq2seq.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(), output_size=4)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testAttentionDecoder2(self):
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        def cell_fn(): return tf.nn.rnn_cell.GRUCell(2)
        cell = cell_fn()
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        enc_outputs, enc_state = tf.nn.static_rnn(cell, inp, dtype=tf.float32)
        attn_states = tf.concat([
            tf.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs
        ], 1)
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3

        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = legacy_seq2seq.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(),
            output_size=4, num_heads=2)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testDynamicAttentionDecoder1(self):
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        def cell_fn(): return tf.nn.rnn_cell.GRUCell(2)
        cell = cell_fn()
        inp = tf.constant(0.5, shape=[2, 2, 2])
        enc_outputs, enc_state = tf.nn.dynamic_rnn(
            cell, inp, dtype=tf.float32)
        attn_states = enc_outputs
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3

        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = legacy_seq2seq.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(), output_size=4)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testDynamicAttentionDecoder2(self):
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        def cell_fn(): return tf.nn.rnn_cell.GRUCell(2)
        cell = cell_fn()
        inp = tf.constant(0.5, shape=[2, 2, 2])
        enc_outputs, enc_state = tf.nn.dynamic_rnn(
            cell, inp, dtype=tf.float32)
        attn_states = enc_outputs
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3

        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = legacy_seq2seq.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(),
            output_size=4, num_heads=2)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testAttentionDecoderStateIsTuple(self):
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        def single_cell(): return tf.nn.rnn_cell.BasicLSTMCell(  # pylint: disable=g-long-lambda
            2, state_is_tuple=True)
        def cell_fn(): return tf.nn.rnn_cell.MultiRNNCell(  # pylint: disable=g-long-lambda
            cells=[single_cell() for _ in range(2)], state_is_tuple=True)
        cell = cell_fn()
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        enc_outputs, enc_state = tf.nn.static_rnn(cell, inp, dtype=tf.float32)
        attn_states = tf.concat([
            tf.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs
        ], 1)
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3

        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = legacy_seq2seq.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(), output_size=4)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual(2, len(res[0]))
        self.assertEqual((2, 2), res[0][0].c.shape)
        self.assertEqual((2, 2), res[0][0].h.shape)
        self.assertEqual((2, 2), res[0][1].c.shape)
        self.assertEqual((2, 2), res[0][1].h.shape)

  def testDynamicAttentionDecoderStateIsTuple(self):
    with self.cached_session() as sess:
      with tf.variable_scope(
          "root", initializer=tf.constant_initializer(0.5)):
        def cell_fn(): return tf.nn.rnn_cell.MultiRNNCell(  # pylint: disable=g-long-lambda
            cells=[tf.nn.rnn_cell.BasicLSTMCell(2) for _ in range(2)])
        cell = cell_fn()
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        enc_outputs, enc_state = tf.nn.static_rnn(cell, inp, dtype=tf.float32)
        attn_states = tf.concat([
            tf.reshape(e, [-1, 1, cell.output_size])
            for e in enc_outputs
        ], 1)
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3

        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = legacy_seq2seq.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(), output_size=4)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual(2, len(res[0]))
        self.assertEqual((2, 2), res[0][0].c.shape)
        self.assertEqual((2, 2), res[0][0].h.shape)
        self.assertEqual((2, 2), res[0][1].c.shape)
        self.assertEqual((2, 2), res[0][1].h.shape)

  def testSequenceLoss(self):
    with self.cached_session() as sess:
      logits = [tf.constant(i + 0.5, shape=[2, 5]) for i in range(3)]
      targets = [
          tf.constant(
              i, tf.int32, shape=[2]) for i in range(3)
      ]
      weights = [tf.constant(1.0, shape=[2]) for i in range(3)]

      average_loss_per_example = legacy_seq2seq.sequence_loss(
          logits,
          targets,
          weights,
          average_across_timesteps=True,
          average_across_batch=True)
      res = sess.run(average_loss_per_example)
      self.assertAllClose(1.60944, res)

      average_loss_per_sequence = legacy_seq2seq.sequence_loss(
          logits,
          targets,
          weights,
          average_across_timesteps=False,
          average_across_batch=True)
      res = sess.run(average_loss_per_sequence)
      self.assertAllClose(4.828314, res)

      total_loss = legacy_seq2seq.sequence_loss(
          logits,
          targets,
          weights,
          average_across_timesteps=False,
          average_across_batch=False)
      res = sess.run(total_loss)
      self.assertAllClose(9.656628, res)

  def testSequenceLossByExample(self):
    with self.cached_session() as sess:
      output_classes = 5
      logits = [
          tf.constant(
              i + 0.5, shape=[2, output_classes]) for i in range(3)
      ]
      targets = [
          tf.constant(
              i, tf.int32, shape=[2]) for i in range(3)
      ]
      weights = [tf.constant(1.0, shape=[2]) for i in range(3)]

      average_loss_per_example = (legacy_seq2seq.sequence_loss_by_example(
          logits, targets, weights, average_across_timesteps=True))
      res = sess.run(average_loss_per_example)
      self.assertAllClose(np.asarray([1.609438, 1.609438]), res)

      loss_per_sequence = legacy_seq2seq.sequence_loss_by_example(
          logits, targets, weights, average_across_timesteps=False)
      res = sess.run(loss_per_sequence)
      self.assertAllClose(np.asarray([4.828314, 4.828314]), res)


if __name__ == "__main__":
  tf.disable_eager_execution()
  tf.test.main()
