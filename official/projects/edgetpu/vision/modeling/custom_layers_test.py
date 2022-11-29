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
from typing import List

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


def random_boxes(shape):
  a = tf.random.uniform(shape=shape+[2])
  b = tf.random.uniform(shape=shape+[2])
  l = tf.minimum(a, b)
  u = tf.maximum(a, b)
  return tf.concat([l, u], axis=-1)


def _maximum_activation_size(model):
  max_size = 0
  for layer in model.layers:
    outputs = layer.output
    if not isinstance(outputs, list):
      outputs = [outputs]
    for output in outputs:
      if hasattr(output, 'shape'):
        size = np.prod(output.shape)
        max_size = max(max_size, size)
        print('Layer', size, output.shape, layer.name)
  return max_size


class NonMaxSuppressionTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((16, 8, 200, 0.009), (31, 17, 100, 0.013),
                            (71, 41, 100, 0.045), (150, 100, 100, 0.129),
                            (300, 300, 100, 0.116), (600, 600, 50, 0.176))
  def test_reference_match(self, n, top, runs, max_deviation):
    """Compares that new optimized method is close to reference method.

    Runs two algorithms with same sets of input boxes and scores, and measures
    deviation between returned sets of prunned boxes.
    Read more about test results at ./g3doc/non_max_suppression.md
    (*) Avoid flakiness with safe boundary (go/python-tips/048): deviation
    between two sets is a positive number, which may vary from test to test.
    Doing multiple runs expected to reduce average deviation variation following
    LLN theorem. Therefore by having first test run we know upper deviation
    bound which algorithm would not exceed until broken (in any feasible amount
    of time in the future). Use of this safe boundary makes test non-flaky.

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
    min_union = 2*n
    boxes = random_boxes([runs, n])
    scores = tf.random.uniform(shape=[runs, n])
    test = custom_layers.non_max_suppression_padded(boxes, scores, top)
    for run in range(runs):
      reference = tf.image.non_max_suppression(boxes[run], scores[run], top)
      reference = {*reference.numpy().tolist()}
      optimized = {*test[run].numpy().astype(int).tolist()} - {-1}
      union_size = len(optimized | reference)
      deviation_rate += len(optimized ^ reference) / union_size
      min_union = min(min_union, union_size)
    deviation_rate = deviation_rate / runs
    # six sigma estimate via LLN theorem
    safe_margin = 6 * (deviation_rate / np.sqrt(runs) + 1/(runs*min_union))
    self.assertLess(
        deviation_rate,
        max_deviation,
        msg='Deviation rate between optimized and reference implementations is '
        'higher than expected. If you are tuning the test, recommended safe '
        'deviation rate is '
        f'{deviation_rate} + {safe_margin} = {deviation_rate + safe_margin}')

  @parameterized.parameters(([16], 8), ([91, 150], 100), ([20, 20, 200], 10))
  def test_sharded_match(self, shape: List[int], top: int):
    boxes = random_boxes(shape)
    scores = tf.random.uniform(shape=shape)
    optimized = custom_layers.non_max_suppression_padded(boxes, scores, top)
    reference = custom_layers._non_max_suppression_as_is(boxes, scores, top)
    self.assertAllEqual(optimized, reference)

  _sharded_nms = custom_layers.non_max_suppression_padded
  _stright_nms = custom_layers._non_max_suppression_as_is

  @parameterized.parameters(([16], 8, _sharded_nms, True),
                            ([16], 8, _stright_nms, True),
                            ([91, 150], 100, _sharded_nms, True),
                            ([91, 150], 100, _stright_nms, False),
                            ([20, 20, 200], 10, _sharded_nms, True),
                            ([20, 20, 200], 10, _stright_nms, False))
  def test_sharded_size(self, shape: List[int], top: int, algorithm,
                        fits_as_is: bool):
    scores = tf.keras.Input(shape=shape, batch_size=1)
    boxes = tf.keras.Input(shape=shape + [4], batch_size=1)
    optimized = algorithm(boxes, scores, top)
    model = tf.keras.Model(inputs=[boxes, scores], outputs=optimized)
    max_size = _maximum_activation_size(model)
    if fits_as_is:
      # Sharding done or not needed.
      self.assertLessEqual(max_size, custom_layers._RECOMMENDED_NMS_MEMORY)
    else:
      # Sharding needed.
      self.assertGreater(max_size, custom_layers._RECOMMENDED_NMS_MEMORY)

  def test_shard_tensors(self):
    a: tf.Tensor = tf.constant([[0, 1, 2, 3, 4]])
    b: tf.Tensor = tf.constant([[
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
    ]])
    for i, (a_i, b_i) in enumerate(custom_layers.shard_tensors(1, 3, a, b)):
      self.assertAllEqual(a_i, a[:, i * 3:i * 3 + 3])
      self.assertAllEqual(b_i, b[:, i * 3:i * 3 + 3, :])

if __name__ == '__main__':
  tf.test.main()
