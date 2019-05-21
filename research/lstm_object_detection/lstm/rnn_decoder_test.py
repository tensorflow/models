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

"""Tests for lstm_object_detection.lstm.rnn_decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from lstm_object_detection.lstm import rnn_decoder


class MockRnnCell(tf.contrib.rnn.RNNCell):

  def __init__(self, input_size, num_units):
    self._input_size = input_size
    self._num_units = num_units
    self._filter_size = [3, 3]

  def __call__(self, inputs, state_tuple):
    outputs = tf.concat([inputs, state_tuple[0]], axis=3)
    new_state_tuple = (tf.multiply(state_tuple[0], 2), state_tuple[1])
    return outputs, new_state_tuple

  def state_size(self):
    return self._num_units

  def output_size(self):
    return self._input_size + self._num_units

  def pre_bottleneck(self, inputs, state, input_index):
    with tf.variable_scope('bottleneck_%d' % input_index, reuse=tf.AUTO_REUSE):
      inputs = tf.contrib.layers.separable_conv2d(
          tf.concat([inputs, state], 3),
          self._input_size,
          self._filter_size,
          depth_multiplier=1,
          activation_fn=tf.nn.relu6,
          normalizer_fn=None)
    return inputs


class RnnDecoderTest(tf.test.TestCase):

  def test_rnn_decoder_single_unroll(self):
    batch_size = 2
    num_unroll = 1
    num_units = 64
    width = 8
    height = 10
    input_channels = 128

    initial_state = tf.random_normal((batch_size, width, height, num_units))
    inputs = tf.random_normal([batch_size, width, height, input_channels])

    rnn_cell = MockRnnCell(input_channels, num_units)
    outputs, states = rnn_decoder.rnn_decoder(
        decoder_inputs=[inputs] * num_unroll,
        initial_state=(initial_state, initial_state),
        cell=rnn_cell)

    self.assertEqual(len(outputs), num_unroll)
    self.assertEqual(len(states), num_unroll)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      results = sess.run((outputs, states, inputs, initial_state))
      outputs_results = results[0]
      states_results = results[1]
      inputs_results = results[2]
      initial_states_results = results[3]
      self.assertEqual(outputs_results[0].shape,
                       (batch_size, width, height, input_channels + num_units))
      self.assertAllEqual(
          outputs_results[0],
          np.concatenate((inputs_results, initial_states_results), axis=3))
      self.assertEqual(states_results[0][0].shape,
                       (batch_size, width, height, num_units))
      self.assertEqual(states_results[0][1].shape,
                       (batch_size, width, height, num_units))
      self.assertAllEqual(states_results[0][0],
                          np.multiply(initial_states_results, 2.0))
      self.assertAllEqual(states_results[0][1], initial_states_results)

  def test_rnn_decoder_multiple_unroll(self):
    batch_size = 2
    num_unroll = 3
    num_units = 64
    width = 8
    height = 10
    input_channels = 128

    initial_state = tf.random_normal((batch_size, width, height, num_units))
    inputs = tf.random_normal([batch_size, width, height, input_channels])

    rnn_cell = MockRnnCell(input_channels, num_units)
    outputs, states = rnn_decoder.rnn_decoder(
        decoder_inputs=[inputs] * num_unroll,
        initial_state=(initial_state, initial_state),
        cell=rnn_cell)

    self.assertEqual(len(outputs), num_unroll)
    self.assertEqual(len(states), num_unroll)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      results = sess.run((outputs, states, inputs, initial_state))
      outputs_results = results[0]
      states_results = results[1]
      inputs_results = results[2]
      initial_states_results = results[3]
      for i in range(num_unroll):
        previous_state = ([initial_states_results, initial_states_results]
                          if i == 0 else states_results[i - 1])
        self.assertEqual(
            outputs_results[i].shape,
            (batch_size, width, height, input_channels + num_units))
        self.assertAllEqual(
            outputs_results[i],
            np.concatenate((inputs_results, previous_state[0]), axis=3))
        self.assertEqual(states_results[i][0].shape,
                         (batch_size, width, height, num_units))
        self.assertEqual(states_results[i][1].shape,
                         (batch_size, width, height, num_units))
        self.assertAllEqual(states_results[i][0],
                            np.multiply(previous_state[0], 2.0))
        self.assertAllEqual(states_results[i][1], previous_state[1])


class MultiInputRnnDecoderTest(tf.test.TestCase):

  def test_rnn_decoder_single_unroll(self):
    batch_size = 2
    num_unroll = 1
    num_units = 12
    width = 8
    height = 10
    input_channels_large = 24
    input_channels_small = 12
    bottleneck_channels = 20

    initial_state_c = tf.random_normal((batch_size, width, height, num_units))
    initial_state_h = tf.random_normal((batch_size, width, height, num_units))
    initial_state = (initial_state_c, initial_state_h)
    inputs_large = tf.random_normal(
        [batch_size, width, height, input_channels_large])
    inputs_small = tf.random_normal(
        [batch_size, width, height, input_channels_small])

    rnn_cell = MockRnnCell(bottleneck_channels, num_units)
    outputs, states = rnn_decoder.multi_input_rnn_decoder(
        decoder_inputs=[[inputs_large] * num_unroll,
                        [inputs_small] * num_unroll],
        initial_state=initial_state,
        cell=rnn_cell,
        sequence_step=tf.zeros([batch_size]),
        pre_bottleneck=True)

    self.assertEqual(len(outputs), num_unroll)
    self.assertEqual(len(states), num_unroll)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      results = sess.run(
          (outputs, states, inputs_large, inputs_small, initial_state))
      outputs_results = results[0]
      states_results = results[1]
      inputs_large_results = results[2]
      inputs_small_results = results[3]
      initial_states_results = results[4]
      self.assertEqual(
          outputs_results[0].shape,
          (batch_size, width, height, bottleneck_channels + num_units))
      self.assertEqual(states_results[0][0].shape,
                       (batch_size, width, height, num_units))
      self.assertEqual(states_results[0][1].shape,
                       (batch_size, width, height, num_units))
      # The first step should always update state.
      self.assertAllEqual(states_results[0][0],
                              np.multiply(initial_states_results[0], 2))
      self.assertAllEqual(states_results[0][1], initial_states_results[1])

  def test_rnn_decoder_multiple_unroll(self):
    batch_size = 2
    num_unroll = 3
    num_units = 12
    width = 8
    height = 10
    input_channels_large = 24
    input_channels_small = 12
    bottleneck_channels = 20

    initial_state_c = tf.random_normal((batch_size, width, height, num_units))
    initial_state_h = tf.random_normal((batch_size, width, height, num_units))
    initial_state = (initial_state_c, initial_state_h)
    inputs_large = tf.random_normal(
        [batch_size, width, height, input_channels_large])
    inputs_small = tf.random_normal(
        [batch_size, width, height, input_channels_small])

    rnn_cell = MockRnnCell(bottleneck_channels, num_units)
    outputs, states = rnn_decoder.multi_input_rnn_decoder(
        decoder_inputs=[[inputs_large] * num_unroll,
                        [inputs_small] * num_unroll],
        initial_state=initial_state,
        cell=rnn_cell,
        sequence_step=tf.zeros([batch_size]),
        pre_bottleneck=True)

    self.assertEqual(len(outputs), num_unroll)
    self.assertEqual(len(states), num_unroll)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      results = sess.run(
          (outputs, states, inputs_large, inputs_small, initial_state))
      outputs_results = results[0]
      states_results = results[1]
      inputs_large_results = results[2]
      inputs_small_results = results[3]
      initial_states_results = results[4]

      # The first step should always update state.
      self.assertAllEqual(states_results[0][0],
                          np.multiply(initial_states_results[0], 2))
      self.assertAllEqual(states_results[0][1], initial_states_results[1])
      for i in range(num_unroll):
        self.assertEqual(
            outputs_results[i].shape,
            (batch_size, width, height, bottleneck_channels + num_units))
        self.assertEqual(states_results[i][0].shape,
                         (batch_size, width, height, num_units))
        self.assertEqual(states_results[i][1].shape,
                         (batch_size, width, height, num_units))

  def test_rnn_decoder_multiple_unroll_with_skip(self):
    batch_size = 2
    num_unroll = 5
    num_units = 12
    width = 8
    height = 10
    input_channels_large = 24
    input_channels_small = 12
    bottleneck_channels = 20
    skip = 2

    initial_state_c = tf.random_normal((batch_size, width, height, num_units))
    initial_state_h = tf.random_normal((batch_size, width, height, num_units))
    initial_state = (initial_state_c, initial_state_h)
    inputs_large = tf.random_normal(
        [batch_size, width, height, input_channels_large])
    inputs_small = tf.random_normal(
        [batch_size, width, height, input_channels_small])

    rnn_cell = MockRnnCell(bottleneck_channels, num_units)
    outputs, states = rnn_decoder.multi_input_rnn_decoder(
        decoder_inputs=[[inputs_large] * num_unroll,
                        [inputs_small] * num_unroll],
        initial_state=initial_state,
        cell=rnn_cell,
        sequence_step=tf.zeros([batch_size]),
        pre_bottleneck=True,
        selection_strategy='SKIP%d' % skip)

    self.assertEqual(len(outputs), num_unroll)
    self.assertEqual(len(states), num_unroll)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      results = sess.run(
          (outputs, states, inputs_large, inputs_small, initial_state))
      outputs_results = results[0]
      states_results = results[1]
      inputs_large_results = results[2]
      inputs_small_results = results[3]
      initial_states_results = results[4]

      for i in range(num_unroll):
        self.assertEqual(
            outputs_results[i].shape,
            (batch_size, width, height, bottleneck_channels + num_units))
        self.assertEqual(states_results[i][0].shape,
                         (batch_size, width, height, num_units))
        self.assertEqual(states_results[i][1].shape,
                         (batch_size, width, height, num_units))

        previous_state = (
            initial_states_results if i == 0 else states_results[i - 1])
        # State only updates during key frames
        if i % (skip + 1) == 0:
          self.assertAllEqual(states_results[i][0],
                              np.multiply(previous_state[0], 2))
          self.assertAllEqual(states_results[i][1], previous_state[1])
        else:
          self.assertAllEqual(states_results[i][0], previous_state[0])
          self.assertAllEqual(states_results[i][1], previous_state[1])


if __name__ == '__main__':
  tf.test.main()
