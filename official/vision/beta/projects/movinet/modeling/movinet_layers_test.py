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
"""Tests for movinet_layers.py."""

from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.modeling.layers import nn_layers
from official.vision.beta.projects.movinet.modeling import movinet_layers


class MovinetLayersTest(parameterized.TestCase, tf.test.TestCase):

  def test_squeeze3d(self):
    squeeze = movinet_layers.Squeeze3D()

    inputs = tf.ones([5, 1, 1, 1, 3])
    predicted = squeeze(inputs)
    expected = tf.ones([5, 3])

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllEqual(predicted, expected)

  def test_mobile_conv2d(self):
    conv2d = movinet_layers.MobileConv2D(
        filters=3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer='ones',
        use_bias=False,
        use_depthwise=False,
        use_temporal=False,
        use_buffered_input=True,
    )

    inputs = tf.ones([1, 2, 2, 2, 3])

    predicted = conv2d(inputs)

    expected = tf.constant(
        [[[[[12., 12., 12.],
            [12., 12., 12.]],
           [[12., 12., 12.],
            [12., 12., 12.]]],
          [[[12., 12., 12.],
            [12., 12., 12.]],
           [[12., 12., 12.],
            [12., 12., 12.]]]]])

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

  def test_mobile_conv2d_temporal(self):
    conv2d = movinet_layers.MobileConv2D(
        filters=3,
        kernel_size=(3, 1),
        strides=(1, 1),
        padding='causal',
        kernel_initializer='ones',
        use_bias=False,
        use_depthwise=True,
        use_temporal=True,
        use_buffered_input=True,
    )

    inputs = tf.ones([1, 2, 2, 1, 3])

    paddings = [[0, 0], [2, 0], [0, 0], [0, 0], [0, 0]]
    padded_inputs = tf.pad(inputs, paddings)
    predicted = conv2d(padded_inputs)

    expected = tf.constant(
        [[[[[1., 1., 1.]],
           [[1., 1., 1.]]],
          [[[2., 2., 2.]],
           [[2., 2., 2.]]]]])

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected)

  def test_stream_buffer(self):
    conv3d_stream = nn_layers.Conv3D(
        filters=3,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        padding='causal',
        kernel_initializer='ones',
        use_bias=False,
        use_buffered_input=True,
    )
    buffer = movinet_layers.StreamBuffer(buffer_size=2)

    conv3d = nn_layers.Conv3D(
        filters=3,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        padding='causal',
        kernel_initializer='ones',
        use_bias=False,
        use_buffered_input=False,
    )

    inputs = tf.ones([1, 4, 2, 2, 3])
    expected = conv3d(inputs)

    for num_splits in [1, 2, 4]:
      frames = tf.split(inputs, inputs.shape[1] // num_splits, axis=1)
      states = {}
      predicted = []
      for frame in frames:
        x, states = buffer(frame, states=states)
        x = conv3d_stream(x)
        predicted.append(x)
      predicted = tf.concat(predicted, axis=1)

      self.assertEqual(predicted.shape, expected.shape)
      self.assertAllClose(predicted, expected)
      self.assertAllClose(
          predicted,
          [[[[[12., 12., 12.]]],
            [[[24., 24., 24.]]],
            [[[36., 36., 36.]]],
            [[[36., 36., 36.]]]]])

  def test_stream_conv_block_2plus1d(self):
    conv_block = movinet_layers.ConvBlock(
        filters=3,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        causal=True,
        kernel_initializer='ones',
        use_bias=False,
        activation='relu',
        conv_type='2plus1d',
    )

    stream_conv_block = movinet_layers.StreamConvBlock(
        filters=3,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        causal=True,
        kernel_initializer='ones',
        use_bias=False,
        activation='relu',
        conv_type='2plus1d',
    )

    inputs = tf.ones([1, 4, 2, 2, 3])
    expected = conv_block(inputs)

    predicted_disabled, _ = stream_conv_block(inputs)

    self.assertEqual(predicted_disabled.shape, expected.shape)
    self.assertAllClose(predicted_disabled, expected)

    for num_splits in [1, 2, 4]:
      frames = tf.split(inputs, inputs.shape[1] // num_splits, axis=1)
      states = {}
      predicted = []
      for frame in frames:
        x, states = stream_conv_block(frame, states=states)
        predicted.append(x)
      predicted = tf.concat(predicted, axis=1)

      self.assertEqual(predicted.shape, expected.shape)
      self.assertAllClose(predicted, expected)
      self.assertAllClose(
          predicted,
          [[[[[35.9640400, 35.9640400, 35.9640400]]],
            [[[71.9280700, 71.9280700, 71.9280700]]],
            [[[107.892105, 107.892105, 107.892105]]],
            [[[107.892105, 107.892105, 107.892105]]]]])

  def test_stream_conv_block_3d_2plus1d(self):
    conv_block = movinet_layers.ConvBlock(
        filters=3,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        causal=True,
        kernel_initializer='ones',
        use_bias=False,
        activation='relu',
        conv_type='3d_2plus1d',
    )

    stream_conv_block = movinet_layers.StreamConvBlock(
        filters=3,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        causal=True,
        kernel_initializer='ones',
        use_bias=False,
        activation='relu',
        conv_type='3d_2plus1d',
    )

    inputs = tf.ones([1, 4, 2, 2, 3])
    expected = conv_block(inputs)

    predicted_disabled, _ = stream_conv_block(inputs)

    self.assertEqual(predicted_disabled.shape, expected.shape)
    self.assertAllClose(predicted_disabled, expected)

    for num_splits in [1, 2, 4]:
      frames = tf.split(inputs, inputs.shape[1] // num_splits, axis=1)
      states = {}
      predicted = []
      for frame in frames:
        x, states = stream_conv_block(frame, states=states)
        predicted.append(x)
      predicted = tf.concat(predicted, axis=1)

      self.assertEqual(predicted.shape, expected.shape)
      self.assertAllClose(predicted, expected)
      self.assertAllClose(
          predicted,
          [[[[[35.9640400, 35.9640400, 35.9640400]]],
            [[[71.9280700, 71.9280700, 71.9280700]]],
            [[[107.892105, 107.892105, 107.892105]]],
            [[[107.892105, 107.892105, 107.892105]]]]])

  def test_stream_conv_block(self):
    conv_block = movinet_layers.ConvBlock(
        filters=3,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        causal=True,
        kernel_initializer='ones',
        use_bias=False,
        activation='relu',
    )

    stream_conv_block = movinet_layers.StreamConvBlock(
        filters=3,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        causal=True,
        kernel_initializer='ones',
        use_bias=False,
        activation='relu',
    )

    inputs = tf.ones([1, 4, 2, 2, 3])
    expected = conv_block(inputs)

    predicted_disabled, _ = stream_conv_block(inputs)

    self.assertEqual(predicted_disabled.shape, expected.shape)
    self.assertAllClose(predicted_disabled, expected)

    for num_splits in [1, 2, 4]:
      frames = tf.split(inputs, inputs.shape[1] // num_splits, axis=1)
      states = {}
      predicted = []
      for frame in frames:
        x, states = stream_conv_block(frame, states=states)
        predicted.append(x)
      predicted = tf.concat(predicted, axis=1)

      self.assertEqual(predicted.shape, expected.shape)
      self.assertAllClose(predicted, expected)
      self.assertAllClose(
          predicted,
          [[[[[11.994005, 11.994005, 11.994005]]],
            [[[23.988010, 23.988010, 23.988010]]],
            [[[35.982014, 35.982014, 35.982014]]],
            [[[35.982014, 35.982014, 35.982014]]]]])

  def test_stream_squeeze_excitation(self):
    se = movinet_layers.StreamSqueezeExcitation(
        3, causal=True, kernel_initializer='ones')

    inputs = tf.range(4, dtype=tf.float32) + 1.
    inputs = tf.reshape(inputs, [1, 4, 1, 1, 1])
    inputs = tf.tile(inputs, [1, 1, 2, 1, 3])
    expected, _ = se(inputs)

    for num_splits in [1, 2, 4]:
      frames = tf.split(inputs, inputs.shape[1] // num_splits, axis=1)
      states = {}
      predicted = []
      for frame in frames:
        x, states = se(frame, states=states)
        predicted.append(x)
      predicted = tf.concat(predicted, axis=1)

      self.assertEqual(predicted.shape, expected.shape)
      self.assertAllClose(predicted, expected, 1e-5, 1e-5)

      self.assertAllClose(
          predicted,
          [[[[[0.9998109, 0.9998109, 0.9998109]],
             [[0.9998109, 0.9998109, 0.9998109]]],
            [[[1.9999969, 1.9999969, 1.9999969]],
             [[1.9999969, 1.9999969, 1.9999969]]],
            [[[3., 3., 3.]],
             [[3., 3., 3.]]],
            [[[4., 4., 4.]],
             [[4., 4., 4.]]]]],
          1e-5, 1e-5)

  def test_stream_squeeze_excitation_2plus3d(self):
    se = movinet_layers.StreamSqueezeExcitation(
        3,
        se_type='2plus3d',
        causal=True,
        activation='hard_swish',
        gating_activation='hard_sigmoid',
        kernel_initializer='ones')

    inputs = tf.range(4, dtype=tf.float32) + 1.
    inputs = tf.reshape(inputs, [1, 4, 1, 1, 1])
    inputs = tf.tile(inputs, [1, 1, 2, 1, 3])
    expected, _ = se(inputs)

    for num_splits in [1, 2, 4]:
      frames = tf.split(inputs, inputs.shape[1] // num_splits, axis=1)
      states = {}
      predicted = []
      for frame in frames:
        x, states = se(frame, states=states)
        predicted.append(x)
      predicted = tf.concat(predicted, axis=1)

      self.assertEqual(predicted.shape, expected.shape)
      self.assertAllClose(predicted, expected)

      self.assertAllClose(
          predicted,
          [[[[[1., 1., 1.]],
             [[1., 1., 1.]]],
            [[[2., 2., 2.]],
             [[2., 2., 2.]]],
            [[[3., 3., 3.]],
             [[3., 3., 3.]]],
            [[[4., 4., 4.]],
             [[4., 4., 4.]]]]])

  def test_stream_movinet_block(self):
    block = movinet_layers.MovinetBlock(
        out_filters=3,
        expand_filters=6,
        kernel_size=(3, 3, 3),
        strides=(1, 2, 2),
        causal=True,
    )

    inputs = tf.range(4, dtype=tf.float32) + 1.
    inputs = tf.reshape(inputs, [1, 4, 1, 1, 1])
    inputs = tf.tile(inputs, [1, 1, 2, 1, 3])
    expected, _ = block(inputs)

    for num_splits in [1, 2, 4]:
      frames = tf.split(inputs, inputs.shape[1] // num_splits, axis=1)
      states = {}
      predicted = []
      for frame in frames:
        x, states = block(frame, states=states)
        predicted.append(x)
      predicted = tf.concat(predicted, axis=1)

      self.assertEqual(predicted.shape, expected.shape)
      self.assertAllClose(predicted, expected)

  def test_stream_classifier_head(self):
    head = movinet_layers.Head(project_filters=5)
    classifier_head = movinet_layers.ClassifierHead(
        head_filters=10, num_classes=4)

    inputs = tf.range(4, dtype=tf.float32) + 1.
    inputs = tf.reshape(inputs, [1, 4, 1, 1, 1])
    inputs = tf.tile(inputs, [1, 1, 2, 1, 3])
    x, _ = head(inputs)
    expected = classifier_head(x)

    for num_splits in [1, 2, 4]:
      frames = tf.split(inputs, inputs.shape[1] // num_splits, axis=1)
      states = {}
      for frame in frames:
        x, states = head(frame, states=states)
        predicted = classifier_head(x)

      self.assertEqual(predicted.shape, expected.shape)
      self.assertAllClose(predicted, expected)


if __name__ == '__main__':
  tf.test.main()
