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

# Lint as: python3
"""Tests for nn_layers."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.modeling.layers import nn_layers


class NNLayersTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    nn_layers.LEGACY_PADDING = False

  def test_hard_swish(self):
    activation = tf.keras.layers.Activation('hard_swish')
    output = activation(tf.constant([-3, -1.5, 0, 3]))
    self.assertAllEqual(output, [0., -0.375, 0., 3.])

  def test_scale(self):
    scale = nn_layers.Scale(initializer=tf.keras.initializers.constant(10.))
    output = scale(3.)
    self.assertAllEqual(output, 30.)

  def test_temporal_softmax_pool(self):
    inputs = tf.range(4, dtype=tf.float32) + 1.
    inputs = tf.reshape(inputs, [1, 4, 1, 1, 1])
    layer = nn_layers.TemporalSoftmaxPool()
    output = layer(inputs)
    self.assertAllClose(
        output,
        [[[[[0.10153633]]],
          [[[0.33481020]]],
          [[[0.82801306]]],
          [[[1.82021690]]]]])

  def test_positional_encoding(self):
    pos_encoding = nn_layers.PositionalEncoding(
        initializer='ones', cache_encoding=False)
    pos_encoding_cached = nn_layers.PositionalEncoding(
        initializer='ones', cache_encoding=True)

    inputs = tf.ones([1, 4, 1, 1, 3])
    outputs, _ = pos_encoding(inputs)
    outputs_cached, _ = pos_encoding_cached(inputs)

    expected = tf.constant(
        [[[[[1.0000000, 1.0000000, 2.0000000]]],
          [[[1.8414710, 1.0021545, 1.5403023]]],
          [[[1.9092975, 1.0043088, 0.5838531]]],
          [[[1.1411200, 1.0064633, 0.0100075]]]]])

    self.assertEqual(outputs.shape, expected.shape)
    self.assertAllClose(outputs, expected)

    self.assertEqual(outputs.shape, outputs_cached.shape)
    self.assertAllClose(outputs, outputs_cached)

    inputs = tf.ones([1, 5, 1, 1, 3])
    _ = pos_encoding(inputs)

  def test_positional_encoding_bfloat16(self):
    pos_encoding = nn_layers.PositionalEncoding(initializer='ones')

    inputs = tf.ones([1, 4, 1, 1, 3], dtype=tf.bfloat16)
    outputs, _ = pos_encoding(inputs)

    expected = tf.constant(
        [[[[[1.0000000, 1.0000000, 2.0000000]]],
          [[[1.8414710, 1.0021545, 1.5403023]]],
          [[[1.9092975, 1.0043088, 0.5838531]]],
          [[[1.1411200, 1.0064633, 0.0100075]]]]])

    self.assertEqual(outputs.shape, expected.shape)
    self.assertAllClose(outputs, expected)

  def test_global_average_pool_basic(self):
    pool = nn_layers.GlobalAveragePool3D(keepdims=True)

    inputs = tf.ones([1, 2, 3, 4, 1])
    outputs = pool(inputs, output_states=False)

    expected = tf.ones([1, 1, 1, 1, 1])

    self.assertEqual(outputs.shape, expected.shape)
    self.assertAllEqual(outputs, expected)

  def test_positional_encoding_stream(self):
    pos_encoding = nn_layers.PositionalEncoding(
        initializer='ones', cache_encoding=False)

    inputs = tf.range(4, dtype=tf.float32) + 1.
    inputs = tf.reshape(inputs, [1, 4, 1, 1, 1])
    inputs = tf.tile(inputs, [1, 1, 1, 1, 3])
    expected, _ = pos_encoding(inputs)

    for num_splits in [1, 2, 4]:
      frames = tf.split(inputs, num_splits, axis=1)
      states = {}
      predicted = []
      for frame in frames:
        output, states = pos_encoding(frame, states=states)
        predicted.append(output)
      predicted = tf.concat(predicted, axis=1)

      self.assertEqual(predicted.shape, expected.shape)
      self.assertAllClose(predicted, expected)
      self.assertAllClose(predicted, [[[[[1.0000000, 1.0000000, 2.0000000]]],
                                       [[[2.8414710, 2.0021544, 2.5403023]]],
                                       [[[3.9092975, 3.0043090, 2.5838532]]],
                                       [[[4.1411200, 4.0064630, 3.0100074]]]]])

  def test_global_average_pool_keras(self):
    pool = nn_layers.GlobalAveragePool3D(keepdims=False)
    keras_pool = tf.keras.layers.GlobalAveragePooling3D()

    inputs = 10 * tf.random.normal([1, 2, 3, 4, 1])

    outputs = pool(inputs, output_states=False)
    keras_output = keras_pool(inputs)

    self.assertAllEqual(outputs.shape, keras_output.shape)
    self.assertAllClose(outputs, keras_output)

  def test_stream_global_average_pool(self):
    gap = nn_layers.GlobalAveragePool3D(keepdims=True, causal=False)

    inputs = tf.range(4, dtype=tf.float32) + 1.
    inputs = tf.reshape(inputs, [1, 4, 1, 1, 1])
    inputs = tf.tile(inputs, [1, 1, 2, 2, 3])
    expected, _ = gap(inputs)

    for num_splits in [1, 2, 4]:
      frames = tf.split(inputs, num_splits, axis=1)
      states = {}
      predicted = None
      for frame in frames:
        predicted, states = gap(frame, states=states)

      self.assertEqual(predicted.shape, expected.shape)
      self.assertAllClose(predicted, expected)
      self.assertAllClose(
          predicted,
          [[[[[2.5, 2.5, 2.5]]]]])

  def test_causal_stream_global_average_pool(self):
    gap = nn_layers.GlobalAveragePool3D(keepdims=True, causal=True)

    inputs = tf.range(4, dtype=tf.float32) + 1.
    inputs = tf.reshape(inputs, [1, 4, 1, 1, 1])
    inputs = tf.tile(inputs, [1, 1, 2, 2, 3])
    expected, _ = gap(inputs)

    for num_splits in [1, 2, 4]:
      frames = tf.split(inputs, num_splits, axis=1)
      states = {}
      predicted = []
      for frame in frames:
        x, states = gap(frame, states=states)
        predicted.append(x)
      predicted = tf.concat(predicted, axis=1)

      self.assertEqual(predicted.shape, expected.shape)
      self.assertAllClose(predicted, expected)
      self.assertAllClose(
          predicted,
          [[[[[1.0, 1.0, 1.0]]],
            [[[1.5, 1.5, 1.5]]],
            [[[2.0, 2.0, 2.0]]],
            [[[2.5, 2.5, 2.5]]]]])

  def test_spatial_average_pool(self):
    pool = nn_layers.SpatialAveragePool3D(keepdims=True)

    inputs = tf.range(64, dtype=tf.float32) + 1.
    inputs = tf.reshape(inputs, [1, 4, 4, 4, 1])

    output = pool(inputs)

    self.assertEqual(output.shape, [1, 4, 1, 1, 1])
    self.assertAllClose(
        output,
        [[[[[8.50]]],
          [[[24.5]]],
          [[[40.5]]],
          [[[56.5]]]]])

  def test_conv2d_causal(self):
    conv2d = nn_layers.Conv2D(
        filters=3,
        kernel_size=(3, 3),
        strides=(1, 2),
        padding='causal',
        use_buffered_input=True,
        kernel_initializer='ones',
        use_bias=False,
    )

    inputs = tf.ones([1, 4, 2, 3])

    paddings = [[0, 0], [2, 0], [0, 0], [0, 0]]
    padded_inputs = tf.pad(inputs, paddings)
    predicted = conv2d(padded_inputs)

    expected = tf.constant(
        [[[[6.0, 6.0, 6.0]],
          [[12., 12., 12.]],
          [[18., 18., 18.]],
          [[18., 18., 18.]]]])

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

    conv2d.use_buffered_input = False
    predicted = conv2d(inputs)

    self.assertFalse(conv2d.use_buffered_input)
    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

  def test_depthwise_conv2d_causal(self):
    conv2d = nn_layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='causal',
        use_buffered_input=True,
        depthwise_initializer='ones',
        use_bias=False,
    )

    inputs = tf.ones([1, 2, 2, 3])

    paddings = [[0, 0], [2, 0], [0, 0], [0, 0]]
    padded_inputs = tf.pad(inputs, paddings)
    predicted = conv2d(padded_inputs)

    expected = tf.constant(
        [[[[2., 2., 2.],
           [2., 2., 2.]],
          [[4., 4., 4.],
           [4., 4., 4.]]]])

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

    conv2d.use_buffered_input = False
    predicted = conv2d(inputs)

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

  def test_conv3d_causal(self):
    conv3d = nn_layers.Conv3D(
        filters=3,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        padding='causal',
        use_buffered_input=True,
        kernel_initializer='ones',
        use_bias=False,
    )

    inputs = tf.ones([1, 2, 4, 4, 3])

    paddings = [[0, 0], [2, 0], [0, 0], [0, 0], [0, 0]]
    padded_inputs = tf.pad(inputs, paddings)
    predicted = conv3d(padded_inputs)

    expected = tf.constant(
        [[[[[27., 27., 27.],
            [18., 18., 18.]],
           [[18., 18., 18.],
            [12., 12., 12.]]],
          [[[54., 54., 54.],
            [36., 36., 36.]],
           [[36., 36., 36.],
            [24., 24., 24.]]]]])

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

    conv3d.use_buffered_input = False
    predicted = conv3d(inputs)

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

  def test_depthwise_conv3d_causal(self):
    conv3d = nn_layers.Conv3D(
        filters=3,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        padding='causal',
        use_buffered_input=True,
        kernel_initializer='ones',
        use_bias=False,
        groups=3,
    )

    inputs = tf.ones([1, 2, 4, 4, 3])

    paddings = [[0, 0], [2, 0], [0, 0], [0, 0], [0, 0]]
    padded_inputs = tf.pad(inputs, paddings)
    predicted = conv3d(padded_inputs)

    expected = tf.constant(
        [[[[[9.0, 9.0, 9.0],
            [6.0, 6.0, 6.0]],
           [[6.0, 6.0, 6.0],
            [4.0, 4.0, 4.0]]],
          [[[18.0, 18.0, 18.0],
            [12., 12., 12.]],
           [[12., 12., 12.],
            [8., 8., 8.]]]]])

    output_shape = conv3d._spatial_output_shape([4, 4, 4])
    self.assertAllClose(output_shape, [2, 2, 2])

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

    conv3d.use_buffered_input = False
    predicted = conv3d(inputs)

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

  def test_conv3d_causal_padding_2d(self):
    """Test to ensure causal padding works like standard padding."""
    conv3d = nn_layers.Conv3D(
        filters=1,
        kernel_size=(1, 3, 3),
        strides=(1, 2, 2),
        padding='causal',
        use_buffered_input=False,
        kernel_initializer='ones',
        use_bias=False,
    )

    keras_conv3d = tf.keras.layers.Conv3D(
        filters=1,
        kernel_size=(1, 3, 3),
        strides=(1, 2, 2),
        padding='same',
        kernel_initializer='ones',
        use_bias=False,
    )

    inputs = tf.ones([1, 1, 4, 4, 1])

    predicted = conv3d(inputs)
    expected = keras_conv3d(inputs)

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

    self.assertAllClose(predicted,
                        [[[[[9.],
                            [6.]],
                           [[6.],
                            [4.]]]]])

  def test_conv3d_causal_padding_1d(self):
    """Test to ensure causal padding works like standard padding."""
    conv3d = nn_layers.Conv3D(
        filters=1,
        kernel_size=(3, 1, 1),
        strides=(2, 1, 1),
        padding='causal',
        use_buffered_input=False,
        kernel_initializer='ones',
        use_bias=False,
    )

    keras_conv1d = tf.keras.layers.Conv1D(
        filters=1,
        kernel_size=3,
        strides=2,
        padding='causal',
        kernel_initializer='ones',
        use_bias=False,
    )

    inputs = tf.ones([1, 4, 1, 1, 1])

    predicted = conv3d(inputs)
    expected = keras_conv1d(tf.squeeze(inputs, axis=[2, 3]))
    expected = tf.reshape(expected, [1, 2, 1, 1, 1])

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

    self.assertAllClose(predicted,
                        [[[[[1.]]],
                          [[[3.]]]]])

if __name__ == '__main__':
  tf.test.main()
