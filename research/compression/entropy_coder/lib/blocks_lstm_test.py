# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Tests for LSTM tensorflow blocks."""
from __future__ import division

import numpy as np
import tensorflow as tf

import block_base
import blocks_std
import blocks_lstm


class BlocksLSTMTest(tf.test.TestCase):

  def CheckUnary(self, y, op_type):
    self.assertEqual(op_type, y.op.type)
    self.assertEqual(1, len(y.op.inputs))
    return y.op.inputs[0]

  def CheckBinary(self, y, op_type):
    self.assertEqual(op_type, y.op.type)
    self.assertEqual(2, len(y.op.inputs))
    return y.op.inputs

  def testLSTM(self):
    lstm = blocks_lstm.LSTM(10)
    lstm.hidden = tf.zeros(shape=[10, 10], dtype=tf.float32)
    lstm.cell = tf.zeros(shape=[10, 10], dtype=tf.float32)
    x = tf.placeholder(dtype=tf.float32, shape=[10, 11])
    y = lstm(x)

    o, tanhc = self.CheckBinary(y, 'Mul')
    self.assertEqual(self.CheckUnary(o, 'Sigmoid').name, 'LSTM/split:3')

    self.assertIs(lstm.cell, self.CheckUnary(tanhc, 'Tanh'))
    fc, ij = self.CheckBinary(lstm.cell, 'Add')

    f, _ = self.CheckBinary(fc, 'Mul')
    self.assertEqual(self.CheckUnary(f, 'Sigmoid').name, 'LSTM/split:0')

    i, j = self.CheckBinary(ij, 'Mul')
    self.assertEqual(self.CheckUnary(i, 'Sigmoid').name, 'LSTM/split:1')
    j = self.CheckUnary(j, 'Tanh')
    self.assertEqual(j.name, 'LSTM/split:2')

  def testLSTMBiasInit(self):
    lstm = blocks_lstm.LSTM(9)
    x = tf.placeholder(dtype=tf.float32, shape=[15, 7])
    lstm(x)
    b = lstm._nn._bias

    with self.test_session():
      tf.global_variables_initializer().run()
      bias_var = b._bias.eval()

      comp = ([1.0] * 9) + ([0.0] * 27)
      self.assertAllEqual(bias_var, comp)

  def testConv2DLSTM(self):
    lstm = blocks_lstm.Conv2DLSTM(depth=10,
                                  filter_size=[1, 1],
                                  hidden_filter_size=[1, 1],
                                  strides=[1, 1],
                                  padding='SAME')
    lstm.hidden = tf.zeros(shape=[10, 11, 11, 10], dtype=tf.float32)
    lstm.cell = tf.zeros(shape=[10, 11, 11, 10], dtype=tf.float32)
    x = tf.placeholder(dtype=tf.float32, shape=[10, 11, 11, 1])
    y = lstm(x)

    o, tanhc = self.CheckBinary(y, 'Mul')
    self.assertEqual(self.CheckUnary(o, 'Sigmoid').name, 'Conv2DLSTM/split:3')

    self.assertIs(lstm.cell, self.CheckUnary(tanhc, 'Tanh'))
    fc, ij = self.CheckBinary(lstm.cell, 'Add')

    f, _ = self.CheckBinary(fc, 'Mul')
    self.assertEqual(self.CheckUnary(f, 'Sigmoid').name, 'Conv2DLSTM/split:0')

    i, j = self.CheckBinary(ij, 'Mul')
    self.assertEqual(self.CheckUnary(i, 'Sigmoid').name, 'Conv2DLSTM/split:1')
    j = self.CheckUnary(j, 'Tanh')
    self.assertEqual(j.name, 'Conv2DLSTM/split:2')

  def testConv2DLSTMBiasInit(self):
    lstm = blocks_lstm.Conv2DLSTM(9, 1, 1, [1, 1], 'SAME')
    x = tf.placeholder(dtype=tf.float32, shape=[1, 7, 7, 7])
    lstm(x)
    b = lstm._bias

    with self.test_session():
      tf.global_variables_initializer().run()
      bias_var = b._bias.eval()

      comp = ([1.0] * 9) + ([0.0] * 27)
      self.assertAllEqual(bias_var, comp)


if __name__ == '__main__':
  tf.test.main()
