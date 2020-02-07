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
"""Tests for lstm_object_detection.lstm.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lstm_object_detection.lstm import utils


class QuantizableUtilsTest(tf.test.TestCase):

  def test_quantizable_concat_is_training(self):
    inputs_1 = tf.zeros([4, 10, 10, 1], dtype=tf.float32)
    inputs_2 = tf.ones([4, 10, 10, 2], dtype=tf.float32)
    concat_in_train = utils.quantizable_concat([inputs_1, inputs_2],
                                               axis=3,
                                               is_training=True)
    self.assertAllEqual([4, 10, 10, 3], concat_in_train.shape.as_list())
    self._check_min_max_ema(tf.get_default_graph())
    self._check_min_max_vars(tf.get_default_graph())

  def test_quantizable_concat_inference(self):
    inputs_1 = tf.zeros([4, 10, 10, 1], dtype=tf.float32)
    inputs_2 = tf.ones([4, 10, 10, 2], dtype=tf.float32)
    concat_in_train = utils.quantizable_concat([inputs_1, inputs_2],
                                               axis=3,
                                               is_training=False)
    self.assertAllEqual([4, 10, 10, 3], concat_in_train.shape.as_list())
    self._check_no_min_max_ema(tf.get_default_graph())
    self._check_min_max_vars(tf.get_default_graph())

  def test_quantizable_concat_not_quantized_is_training(self):
    inputs_1 = tf.zeros([4, 10, 10, 1], dtype=tf.float32)
    inputs_2 = tf.ones([4, 10, 10, 2], dtype=tf.float32)
    concat_in_train = utils.quantizable_concat([inputs_1, inputs_2],
                                               axis=3,
                                               is_training=True,
                                               is_quantized=False)
    self.assertAllEqual([4, 10, 10, 3], concat_in_train.shape.as_list())
    self._check_no_min_max_ema(tf.get_default_graph())
    self._check_no_min_max_vars(tf.get_default_graph())

  def test_quantizable_concat_not_quantized_inference(self):
    inputs_1 = tf.zeros([4, 10, 10, 1], dtype=tf.float32)
    inputs_2 = tf.ones([4, 10, 10, 2], dtype=tf.float32)
    concat_in_train = utils.quantizable_concat([inputs_1, inputs_2],
                                               axis=3,
                                               is_training=False,
                                               is_quantized=False)
    self.assertAllEqual([4, 10, 10, 3], concat_in_train.shape.as_list())
    self._check_no_min_max_ema(tf.get_default_graph())
    self._check_no_min_max_vars(tf.get_default_graph())

  def test_quantize_op_is_training(self):
    inputs = tf.zeros([4, 10, 10, 128], dtype=tf.float32)
    outputs = utils.quantize_op(inputs)
    self.assertAllEqual(inputs.shape.as_list(), outputs.shape.as_list())
    self._check_min_max_ema(tf.get_default_graph())
    self._check_min_max_vars(tf.get_default_graph())

  def test_quantize_op_inferene(self):
    inputs = tf.zeros([4, 10, 10, 128], dtype=tf.float32)
    outputs = utils.quantize_op(inputs, is_training=False)
    self.assertAllEqual(inputs.shape.as_list(), outputs.shape.as_list())
    self._check_no_min_max_ema(tf.get_default_graph())
    self._check_min_max_vars(tf.get_default_graph())

  def _check_min_max_vars(self, graph):
    op_types = [op.type for op in graph.get_operations()]
    self.assertTrue(
        any('FakeQuantWithMinMaxVars' in op_type for op_type in op_types))

  def _check_min_max_ema(self, graph):
    op_names = [op.name for op in graph.get_operations()]
    self.assertTrue(any('AssignMinEma' in name for name in op_names))
    self.assertTrue(any('AssignMaxEma' in name for name in op_names))
    self.assertTrue(any('SafeQuantRangeMin' in name for name in op_names))
    self.assertTrue(any('SafeQuantRangeMax' in name for name in op_names))

  def _check_no_min_max_vars(self, graph):
    op_types = [op.type for op in graph.get_operations()]
    self.assertFalse(
        any('FakeQuantWithMinMaxVars' in op_type for op_type in op_types))

  def _check_no_min_max_ema(self, graph):
    op_names = [op.name for op in graph.get_operations()]
    self.assertFalse(any('AssignMinEma' in name for name in op_names))
    self.assertFalse(any('AssignMaxEma' in name for name in op_names))
    self.assertFalse(any('SafeQuantRangeMin' in name for name in op_names))
    self.assertFalse(any('SafeQuantRangeMax' in name for name in op_names))


class QuantizableSeparableConv2dTest(tf.test.TestCase):

  def test_quantizable_separable_conv2d(self):
    inputs = tf.zeros([4, 10, 10, 128], dtype=tf.float32)
    num_outputs = 64
    kernel_size = [3, 3]
    scope = 'QuantSeparable'
    outputs = utils.quantizable_separable_conv2d(
        inputs, num_outputs, kernel_size, scope=scope)
    self.assertAllEqual([4, 10, 10, num_outputs], outputs.shape.as_list())
    self._check_depthwise_bias_add(tf.get_default_graph(), scope)

  def test_quantizable_separable_conv2d_not_quantized(self):
    inputs = tf.zeros([4, 10, 10, 128], dtype=tf.float32)
    num_outputs = 64
    kernel_size = [3, 3]
    scope = 'QuantSeparable'
    outputs = utils.quantizable_separable_conv2d(
        inputs, num_outputs, kernel_size, is_quantized=False, scope=scope)
    self.assertAllEqual([4, 10, 10, num_outputs], outputs.shape.as_list())
    self._check_no_depthwise_bias_add(tf.get_default_graph(), scope)

  def _check_depthwise_bias_add(self, graph, scope):
    op_names = [op.name for op in graph.get_operations()]
    self.assertTrue(
        any('%s_bias/BiasAdd' % scope in name for name in op_names))

  def _check_no_depthwise_bias_add(self, graph, scope):
    op_names = [op.name for op in graph.get_operations()]
    self.assertFalse(
        any('%s_bias/BiasAdd' % scope in name for name in op_names))


if __name__ == '__main__':
  tf.test.main()
