# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for mixing.py."""

import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import mixing


class MixingTest(tf.test.TestCase):

  def test_base_mixing_layer(self):
    inputs = tf.random.uniform((3, 8, 16),
                               minval=0,
                               maxval=10,
                               dtype=tf.float32)

    with self.assertRaisesRegex(NotImplementedError, "Abstract method"):
      _ = mixing.MixingLayer()(query=inputs, value=inputs)

  def test_fourier_layer(self):
    batch_size = 4
    max_seq_length = 8
    hidden_dim = 16

    inputs = tf.random.uniform((batch_size, max_seq_length, hidden_dim),
                               minval=0,
                               maxval=10,
                               dtype=tf.float32)
    outputs = mixing.FourierTransformLayer(use_fft=True)(
        query=inputs, value=inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, hidden_dim))

  def test_hartley_layer(self):
    batch_size = 3
    max_seq_length = 16
    hidden_dim = 4

    inputs = tf.random.uniform((batch_size, max_seq_length, hidden_dim),
                               minval=0,
                               maxval=12,
                               dtype=tf.float32)
    outputs = mixing.HartleyTransformLayer(use_fft=True)(
        query=inputs, value=inputs)
    self.assertEqual(outputs.shape, (batch_size, max_seq_length, hidden_dim))

  def test_linear_mixing_layer(self):
    batch_size = 2
    max_seq_length = 4
    hidden_dim = 3

    inputs = tf.ones((batch_size, max_seq_length, hidden_dim), dtype=tf.float32)
    outputs = mixing.LinearTransformLayer(
        kernel_initializer=tf_keras.initializers.Ones())(
            query=inputs, value=inputs)

    # hidden_dim * (max_seq_length * 1) = 12.
    expected_outputs = [
        [
            [12., 12., 12.],
            [12., 12., 12.],
            [12., 12., 12.],
            [12., 12., 12.],
        ],
        [
            [12., 12., 12.],
            [12., 12., 12.],
            [12., 12., 12.],
            [12., 12., 12.],
        ],
    ]
    np.testing.assert_allclose(outputs, expected_outputs, rtol=1e-6, atol=1e-6)

  def test_pick_fourier_transform(self):
    # Ensure we don't hit an edge case which exceeds the fixed numerical error.
    tf.random.set_seed(1)
    np.random.seed(1)

    batch_size = 3
    max_seq_length = 4
    hidden_dim = 8

    fft = mixing._pick_fourier_transform(
        use_fft=True, max_seq_length=max_seq_length, hidden_dim=hidden_dim)
    dft_matmul = mixing._pick_fourier_transform(
        use_fft=False, max_seq_length=max_seq_length, hidden_dim=hidden_dim)

    inputs = tf.random.uniform([batch_size, max_seq_length, hidden_dim])
    inputs = tf.cast(inputs, tf.complex64)

    np.testing.assert_allclose(
        fft(inputs), dft_matmul(inputs), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  tf.test.main()
