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

"""Tests for custom_layers."""

import itertools

from absl.testing import parameterized
import tensorflow as tf, tf_keras
from official.projects.edgetpu.vision.modeling import custom_layers

GROUPS = [2, 4]
INPUT_CHANNEL = [8, 16]
OUTPUT_CHANNEL = [8, 16]
USE_BATCH_NORM = [True, False]
ACTIVATION = ['relu', 'linear']
BATCH_NORM_LAYER = tf_keras.layers.BatchNormalization

# 2 functionally identical group conv implementations.
GROUP_CONV_IMPL = {
    'layer': custom_layers.GroupConv2D,
    'model': custom_layers.GroupConv2DKerasModel
}


def _get_random_inputs(input_shape):
  return tf.random.uniform(shape=input_shape)


class GroupConv2DTest(tf.test.TestCase, parameterized.TestCase):

  # Test for combinations of groups, input_channel, output_channel, and
  # whether to use batch_norm
  @parameterized.parameters(
      itertools.product(GROUPS, INPUT_CHANNEL, OUTPUT_CHANNEL, USE_BATCH_NORM))
  def test_construction(self, groups, input_channel, output_channel,
                        use_batch_norm):
    batch_norm_layer = BATCH_NORM_LAYER if use_batch_norm else None
    l = custom_layers.GroupConv2D(
        output_channel,
        3,
        groups=groups,
        use_bias=True,
        batch_norm_layer=batch_norm_layer)
    inputs = _get_random_inputs(input_shape=(1, 4, 4, output_channel))
    _ = l(inputs)
    # kernel and bias for each group. When using batch norm, 2 additional
    # trainable weights per group for batchnorm layers: gamma and beta.
    expected_num_trainable_weights = groups * (2 + 2 * use_batch_norm)
    self.assertLen(l.trainable_weights, expected_num_trainable_weights)

  @parameterized.parameters(
      itertools.product(GROUPS, INPUT_CHANNEL, OUTPUT_CHANNEL))
  def test_kernel_shapes(self, groups, input_channel, output_channel):
    l = custom_layers.GroupConv2D(
        output_channel, 3, groups=groups, use_bias=False)
    _ = l(_get_random_inputs(input_shape=(1, 32, 32, input_channel)))
    expected_kernel_shapes = [(3, 3, int(input_channel / groups),
                               int(output_channel / groups))
                              for _ in range(groups)]
    kernel_shapes = [
        l.trainable_weights[i].get_shape()
        for i in range(len(l.trainable_weights))
    ]
    self.assertListEqual(kernel_shapes, expected_kernel_shapes)

  @parameterized.parameters(
      itertools.product(GROUPS, INPUT_CHANNEL, OUTPUT_CHANNEL))
  def test_output_shapes(self, groups, input_channel, output_channel):
    l = custom_layers.GroupConv2D(
        output_channel, 3, groups=groups, use_bias=False, padding='same')
    outputs = l(_get_random_inputs(input_shape=[2, 32, 32, input_channel]))
    self.assertListEqual(outputs.get_shape().as_list(),
                         [2, 32, 32, output_channel])

  @parameterized.parameters(
      itertools.product(GROUPS, USE_BATCH_NORM, ACTIVATION))
  def test_serialization_deserialization(self, groups, use_batch_norm,
                                         activation):
    batch_norm_layer = BATCH_NORM_LAYER if use_batch_norm else None
    l = custom_layers.GroupConv2D(
        filters=8,
        kernel_size=1,
        groups=groups,
        use_bias=False,
        padding='same',
        batch_norm_layer=batch_norm_layer,
        activation=activation)
    config = l.get_config()
    # New layer from config
    new_l = custom_layers.GroupConv2D.from_config(config)
    # Copy the weights too.
    l.build(input_shape=(1, 1, 4))
    new_l.build(input_shape=(1, 1, 4))
    new_l.set_weights(l.get_weights())
    inputs = _get_random_inputs((1, 1, 1, 4))
    self.assertNotEqual(l, new_l)
    self.assertAllEqual(l(inputs), new_l(inputs))

  @parameterized.parameters(
      itertools.product(GROUPS, INPUT_CHANNEL, OUTPUT_CHANNEL, USE_BATCH_NORM,
                        ACTIVATION))
  def test_equivalence(self, groups, input_channel, output_channel,
                       use_batch_norm, activation):
    batch_norm_layer = BATCH_NORM_LAYER if use_batch_norm else None
    kwargs = dict(
        filters=output_channel,
        groups=groups,
        kernel_size=1,
        use_bias=False,
        batch_norm_layer=batch_norm_layer,
        activation=activation)
    gc_layer = tf_keras.Sequential([custom_layers.GroupConv2D(**kwargs)])
    gc_model = custom_layers.GroupConv2DKerasModel(**kwargs)
    gc_layer.build(input_shape=(None, 3, 3, input_channel))
    gc_model.build(input_shape=(None, 3, 3, input_channel))

    inputs = _get_random_inputs((2, 3, 3, input_channel))
    gc_layer.set_weights(gc_model.get_weights())

    self.assertAllEqual(gc_layer(inputs), gc_model(inputs))

  @parameterized.parameters(('layer', 1, 4), ('layer', 4, 4), ('model', 1, 4),
                            ('model', 4, 4))
  def test_invalid_groups_raises_value_error(self, gc_type, groups,
                                             output_channel):

    with self.assertRaisesRegex(ValueError, r'^(Number of groups)'):
      _ = GROUP_CONV_IMPL[gc_type](
          filters=output_channel, groups=groups, kernel_size=3)

  @parameterized.parameters(('layer', 3, 4), ('layer', 4, 6), ('model', 3, 4),
                            ('model', 4, 6))
  def test_non_group_divisible_raises_value_error(self, gc_type, groups,
                                                  input_channel):
    with self.assertRaisesRegex(ValueError, r'^(Number of input channels)'):
      l = GROUP_CONV_IMPL[gc_type](
          filters=groups * 4, groups=groups, kernel_size=3)
      l.build(input_shape=(4, 4, input_channel))

  @parameterized.parameters(('layer'), ('model'))
  def test_non_supported_data_format_raises_value_error(self, gc_type):
    with self.assertRaisesRegex(ValueError, r'^(.*(channels_last).*)'):
      _ = GROUP_CONV_IMPL[gc_type](
          filters=4, groups=2, kernel_size=1, data_format='channels_first')

  @parameterized.parameters(('layer'), ('model'))
  def test_invalid_batch_norm_raises_value_error(self, gc_type):

    def my_batch_norm(x):
      return x**2

    with self.assertRaisesRegex(ValueError, r'^(.*(not a class).*)'):
      _ = GROUP_CONV_IMPL[gc_type](
          filters=4, groups=2, kernel_size=1, batch_norm_layer=my_batch_norm)

  @parameterized.parameters(('layer'), ('model'))
  def test_invalid_padding_raises_value_error(self, gc_type):
    with self.assertRaisesRegex(ValueError, r'^(.*(same, or valid).*)'):
      _ = GROUP_CONV_IMPL[gc_type](
          filters=4, groups=2, kernel_size=1, padding='causal')


class ArgmaxTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(([16, 32, 64], tf.dtypes.float32, tf.dtypes.int32),
                            ([255, 19], tf.dtypes.int32, tf.dtypes.int64))
  def test_reference_match(self, shape, input_type, output_type):
    random_inputs = tf.random.uniform(shape=shape, maxval=10, dtype=input_type)
    for axis in range(-len(shape) + 1, len(shape)):
      control_output = tf.math.argmax(
          random_inputs, axis=axis, output_type=output_type)
      test_output = custom_layers.argmax(
          random_inputs, axis=axis, output_type=output_type)
      self.assertAllEqual(control_output, test_output)
