# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
import numpy as np
import tensorflow as tf
from official.projects.edgetpu.vision.modeling import custom_layers

GROUPS = [2, 4]
INPUT_CHANNEL = [8, 16]
OUTPUT_CHANNEL = [8, 16]
USE_BATCH_NORM = [True, False]
ACTIVATION = ['relu', 'linear']
BATCH_NORM_LAYER = tf.keras.layers.BatchNormalization

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
    gc_layer = tf.keras.Sequential([custom_layers.GroupConv2D(**kwargs)])
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


def random_boxes(n):
  a = tf.random.uniform(shape=[n, 2])
  b = tf.random.uniform(shape=[n, 2])
  l = tf.minimum(a, b)
  u = tf.maximum(a, b)
  return tf.concat([l, u], axis=-1)


class NonMaxSuppressionTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((16, 8, 500, 0.016), (31, 17, 300, 0.033),
                            (71, 41, 300, 0.065), (150, 100, 250, 0.137),
                            (300, 300, 250, 0.126), (600, 600, 100, 0.213))
  def test_reference_match(self, n, top, runs, max_deviation):
    """Compares that new optimized method is close to reference method.

    Runs two algorithms with same sets of input boxes and scores, and measures
    deviation between returned sets of prunned boxes.
    (*) Avoid flakiness with safe boundary (go/python-tips/048): deviation
    between two sets is a positive number, which may vary from test to test.
    Doing multiple runs expected to reduce average deviation variation following
    LLN theorem. Therefore by having first test run we know upper deviation
    bound which algorithm would not exceed until broken (in any feasible amount
    of time in the future). Use of this safe boundary makes test non-flaky.
    (**) Parametrized inputs description. See safe deviation choice is higher
    than absolute deviation to avoid flaky tesing.
    in # | out # | deflake # | test time | deviation | safe threshold
    ---- | ----- | --------- | --------- | --------- | --------------
    18   | 8     | 500       | 6 sec     | 0.4%      | 1.6%
    31   | 17    | 300       | 7 sec     | 1.0%      | 3.3%
    71   | 41    | 300       | 7 sec     | 3.4%      | 6.5%
    150  | 100   | 250       | 7 sec     | 8.2%      | 13.7%
    300  | 300   | 250       | 10 sec    | 7.4%      | 12.6%
    600  | 600   | 100       | 9 sec     | 9.6%      | 21.3%

    Args:
      n: number of boxes and scores on input of the algorithm.
      top: limit of output boxes count.
      runs: for the statistical testing number of runs to performs to avoid
        tests flakiness.
      max_deviation: mean limit on deviation between optimized and reference
        algorithms. Please read notes why this number may be set higher to avoid
        flaky testing.
    """
    deviation_rate = 0
    for _ in range(runs):
      boxes = random_boxes(n)
      scores = tf.random.uniform(shape=[n])
      optimized = custom_layers.non_max_suppression_padded(boxes, scores, top)
      optimized = {*optimized.numpy().astype(int).tolist()} - {-1}
      reference = tf.image.non_max_suppression(boxes, scores, top)
      reference = {*reference.numpy().tolist()}
      deviation_rate += len(optimized ^ reference) / len(optimized | reference)
    deviation_rate = deviation_rate / runs
    # six sigma estimate via LLN theorem
    safe_margin = 6 * (deviation_rate / np.sqrt(runs) + 1 / runs)
    self.assertLess(
        deviation_rate,
        max_deviation,
        msg='Deviation rate between optimized and reference implementations is '
        'higher than expected. If you are tuning the test, recommended safe '
        'deviation rate is '
        f'{deviation_rate} + {safe_margin} = {deviation_rate + safe_margin}')


if __name__ == '__main__':
  tf.test.main()
